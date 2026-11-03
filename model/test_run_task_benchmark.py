"""Tests for the native DREAM human-scored task benchmark runner."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from dream_transformer import DreamConfig, DreamTransformer, DreamByteTokenizer
from dream_bpe import DreamBPETokenizer

SCRIPT = Path(__file__).with_name("run_task_benchmark.py")


class NativeTaskBenchmarkTests(unittest.TestCase):
    def make_checkpoint(self, root):
        config = DreamConfig(
            vocab_size=256, block_size=64, d_model=16,
            n_heads=4, n_layers=1, dropout=0.0
        )
        model = DreamTransformer(config)
        path = root / "dream-test.pt"
        torch.save({
            "model_state": model.state_dict(),
            "config": config.__dict__,
            "tokenizer": "utf8-byte-v1",
            "training": {"completed_steps": 2}
        }, path)
        return path

    def make_benchmark(self, root, tasks=None):
        if tasks is None:
            tasks = [
                {
                    "id": "math-test",
                    "category": "mathematics",
                    "prompt": "What is 2 plus 2? Explain briefly.",
                    "rubric": ["Gives the answer four", "Explains the addition"]
                },
                {
                    "id": "language-test",
                    "category": "language",
                    "prompt": "Say hello in Kiswahili.",
                    "rubric": ["Uses a suitable Kiswahili greeting"]
                }
            ]
        path = root / "benchmark.jsonl"
        path.write_text(
            "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks),
            encoding="utf-8"
        )
        return path

    def test_generates_reproducible_human_scoring_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.make_checkpoint(root)
            benchmark = self.make_benchmark(root)
            output = root / "report.json"
            result = subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--checkpoint", str(checkpoint),
                    "--benchmark", str(benchmark),
                    "--out", str(output),
                    "--max-new-tokens", "4",
                    "--temperature", "0",
                    "--limit", "2"
                ],
                capture_output=True, text=True, timeout=60
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], "dream-native-task-benchmark-v1")
            self.assertEqual(report["task_count"], 2)
            self.assertFalse(report["data_provenance_verified"])
            self.assertIsNone(report["corpus_manifest_sha256"])
            self.assertIsNone(report["pretraining_corpus_sha256"])
            self.assertIsNone(report["sft_train_sha256"])
            self.assertEqual(len(report["checkpoint_sha256"]), 64)
            self.assertEqual(len(report["benchmark_sha256"]), 64)
            self.assertEqual(
                report["scoring_policy"],
                "Human rubric scoring only; mark each rubric_scores entry 0 or 1, "
                "then record the optional aggregate human_score; no model-based automatic grading."
            )
            self.assertEqual(report["results"][0]["id"], "math-test")
            self.assertEqual(report["results"][0]["prompt_tokens_used"], report["results"][0]["prompt_tokens_original"])
            self.assertFalse(report["results"][0]["prompt_truncated"])
            self.assertIn("rubric", report["results"][0])
            self.assertEqual(
                len(report["results"][0]["rubric_scores"]),
                len(report["results"][0]["rubric"])
            )
            self.assertTrue(all(score is None for score in report["results"][0]["rubric_scores"]))
            self.assertIn("response", report["results"][0])
            self.assertIsNone(report["results"][0]["human_score"])

    def test_runs_native_bpe_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = DreamBPETokenizer.train(
                "DREAM native benchmark tokenizer material " * 20,
                vocab_size=280
            )
            config = DreamConfig(
                vocab_size=tokenizer.vocab_size, block_size=64, d_model=16,
                n_heads=4, n_layers=1, dropout=0.0
            )
            model = DreamTransformer(config)
            checkpoint = root / "dream-bpe-test.pt"
            torch.save({
                "model_state": model.state_dict(),
                "config": config.__dict__,
                "tokenizer": DreamBPETokenizer.format_version,
                "tokenizer_data": tokenizer.to_dict(),
                "training": {"completed_steps": 2}
            }, checkpoint)
            benchmark = self.make_benchmark(root)
            output = root / "bpe-report.json"
            result = subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--checkpoint", str(checkpoint),
                    "--benchmark", str(benchmark),
                    "--out", str(output),
                    "--max-new-tokens", "2",
                    "--temperature", "0",
                    "--limit", "1"
                ],
                capture_output=True, text=True, timeout=60
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["tokenizer"], DreamBPETokenizer.format_version)
            self.assertEqual(report["task_count"], 1)

    def test_rejects_duplicate_task_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.make_checkpoint(root)
            task = {
                "id": "duplicate",
                "category": "reasoning",
                "prompt": "Explain a simple idea.",
                "rubric": ["Provides a coherent explanation"]
            }
            benchmark = self.make_benchmark(root, [task, task])
            result = subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--checkpoint", str(checkpoint),
                    "--benchmark", str(benchmark),
                    "--out", str(root / "report.json"),
                    "--max-new-tokens", "1"
                ],
                capture_output=True, text=True, timeout=30
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Duplicate benchmark task id", result.stderr)


if __name__ == "__main__":
    unittest.main()
