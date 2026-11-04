"""Tests for cached-vs-reference generation benchmarking."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from dream_transformer import DreamConfig, DreamTransformer
from benchmark_generation import benchmark

SCRIPT = Path(__file__).with_name("benchmark_generation.py")


class GenerationBenchmarkTests(unittest.TestCase):
    def test_benchmark_confirms_exact_output_parity_and_reports_latency(self):
        torch.manual_seed(13)
        config = DreamConfig(block_size=64, d_model=16, n_heads=4, n_layers=1, dropout=0.0)
        model = DreamTransformer(config).eval()
        prompt = torch.tensor([[65, 66, 67]], dtype=torch.long)
        result = benchmark(model, prompt, max_new_tokens=4, runs=1, warmup=0)
        self.assertTrue(result["equivalent_outputs"])
        self.assertGreater(result["cached"]["median_seconds"], 0)
        self.assertGreater(result["full_prefix_reference"]["median_seconds"], 0)
        self.assertGreater(result["speedup_ratio"], 0)

    def test_cli_writes_checkpoint_hashed_json_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.manual_seed(19)
            config = DreamConfig(block_size=64, d_model=16, n_heads=4, n_layers=1, dropout=0.0)
            model = DreamTransformer(config)
            checkpoint = root / "tiny.pt"
            torch.save({
                "config": config.__dict__,
                "model_state": model.state_dict(),
                "tokenizer": "utf8-byte-v1",
                "training": {"completed_steps": 1},
            }, checkpoint)
            output = root / "report.json"
            result = subprocess.run([
                sys.executable, str(SCRIPT),
                "--checkpoint", str(checkpoint),
                "--prompt", "USER: hi\nASSISTANT:",
                "--tokens", "3",
                "--runs", "1",
                "--warmup", "0",
                "--out", str(output),
            ], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["equivalent_outputs"])
            self.assertEqual(report["completed_training_steps"], 1)
            self.assertEqual(len(report["checkpoint_sha256"]), 64)
            self.assertIn("speedup_ratio", report)

    def test_benchmark_rejects_context_overflow(self):
        config = DreamConfig(block_size=8, d_model=16, n_heads=4, n_layers=1, dropout=0.0)
        model = DreamTransformer(config).eval()
        with self.assertRaisesRegex(ValueError, "must fit inside block_size"):
            benchmark(model, torch.tensor([[65, 66]], dtype=torch.long), 7, runs=1, warmup=0)


if __name__ == "__main__":
    unittest.main()
