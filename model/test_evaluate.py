"""Held-out evaluation metric tests for byte and BPE tokenizers."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from dream_bpe import DreamBPETokenizer
from dream_transformer import DreamConfig, DreamTransformer

SCRIPT = Path(__file__).with_name("evaluate.py")


class DreamEvaluationTests(unittest.TestCase):
    def test_bpe_cross_entropy_is_normalized_by_utf8_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_text = "DREAM language modeling learns patterns safely. " * 30
            evaluation_text = "DREAM language modeling learns new patterns safely. " * 8
            tokenizer = DreamBPETokenizer.train(training_text, vocab_size=320)
            config = DreamConfig(vocab_size=tokenizer.vocab_size, block_size=8,
                                 d_model=16, n_heads=4, n_layers=1, dropout=0.0)
            model = DreamTransformer(config)
            checkpoint = root / "bpe.pt"
            torch.save({
                "model_state": model.state_dict(),
                "config": config.__dict__,
                "tokenizer": tokenizer.to_dict(),
                "training": {"completed_steps": 1}
            }, checkpoint)
            data = root / "test.txt"
            data.write_text(evaluation_text, encoding="utf-8")
            report_path = root / "report.json"
            result = subprocess.run([
                sys.executable, str(SCRIPT), "--checkpoint", str(checkpoint),
                "--data", str(data), "--batch-size", "2", "--batches", "2",
                "--sample-tokens", "0", "--out", str(report_path)
            ], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["evaluation_bytes"], len(evaluation_text.encode("utf-8")))
            self.assertLess(report["evaluation_tokens"], report["evaluation_bytes"])
            self.assertTrue(torch.isfinite(torch.tensor(
                report["mean_cross_entropy_nats_per_token"]
            )).item())
            self.assertTrue(torch.isfinite(torch.tensor(
                report["mean_cross_entropy_nats_per_byte"]
            )).item())
            self.assertAlmostEqual(
                report["mean_cross_entropy_nats_per_byte"],
                report["mean_cross_entropy_nats_per_token"]
                * report["sampled_target_tokens"] / report["sampled_target_bytes"],
                places=6
            )
            self.assertAlmostEqual(
                report["bits_per_byte"],
                report["mean_cross_entropy_nats_per_byte"] / 0.6931471805599453,
                places=6
            )



    def test_manifest_verified_language_subset_accepts_intact_test_files(self):
        from evaluate import validate_language_subset
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "test.txt"
            subset = root / "test.en.txt"
            parent.write_text("English held out example.\\nKiswahili held out example.\\n", encoding="utf-8")
            subset.write_text("English held out example.\\n", encoding="utf-8")
            parent_hash = hashlib.sha256(parent.read_bytes()).hexdigest()
            subset_hash = hashlib.sha256(subset.read_bytes()).hexdigest()
            manifest = {
                "schema": "dream-corpus-v1",
                "splits": {"test": {
                    "path": parent.name,
                    "sha256": parent_hash,
                    "source_group_ids": ["group-en", "group-sw"],
                    "language_files": {"en": {
                        "path": subset.name,
                        "sha256": subset_hash,
                        "source_group_ids": ["group-en"]
                    }}
                }}
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(
                validate_language_subset(manifest_path, "en", subset, parent),
                parent_hash
            )

    def test_manifest_verified_language_subset_rejects_modified_or_leaking_data(self):
        from evaluate import validate_language_subset
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "test.txt"
            subset = root / "test.en.txt"
            parent.write_text("English held out example.\\nKiswahili held out example.\\n", encoding="utf-8")
            subset.write_text("English held out example.\\n", encoding="utf-8")
            manifest = {
                "schema": "dream-corpus-v1",
                "splits": {"test": {
                    "path": parent.name,
                    "sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
                    "source_group_ids": ["group-en", "group-sw"],
                    "language_files": {"en": {
                        "path": subset.name,
                        "sha256": hashlib.sha256(subset.read_bytes()).hexdigest(),
                        "source_group_ids": ["group-en"]
                    }}
                }}
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            subset.write_text("Tampered evaluation data that is no longer in the manifest.", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash does not match"):
                validate_language_subset(manifest_path, "en", subset, parent)

            subset.write_text("English held out example.\\n", encoding="utf-8")
            manifest["splits"]["test"]["language_files"]["en"]["sha256"] = hashlib.sha256(
                subset.read_bytes()
            ).hexdigest()
            manifest["splits"]["test"]["language_files"]["en"]["source_group_ids"] = ["not-in-parent"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source groups"):
                validate_language_subset(manifest_path, "en", subset, parent)

\nif __name__ == "__main__":
    unittest.main()
