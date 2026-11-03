"""End-to-end smoke tests for DREAM's own training/checkpoint pipeline."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("train.py")


class DreamTrainingSmokeTests(unittest.TestCase):
    def run_training(self, root, checkpoint, extra=()):
        return subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--data", str(root / "train.txt"),
                "--validation-data", str(root / "validation.txt"),
                "--out", str(checkpoint),
                "--steps", "2",
                "--batch-size", "2",
                "--block-size", "8",
                "--d-model", "16",
                "--layers", "1",
                "--heads", "4",
                "--eval-every", "1",
                "--checkpoint-every", "1",
                *extra
            ],
            capture_output=True, text=True, timeout=120
        )

    def test_train_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Deliberately separate corpora for this mechanics-only smoke test.
            (root / "train.txt").write_text(
                ("DREAM authored training sample with varied words and characters. " * 8),
                encoding="utf-8"
            )
            (root / "validation.txt").write_text(
                ("Independent validation material for testing language loss. " * 8),
                encoding="utf-8"
            )
            checkpoint = root / "dream-test.pt"
            first = self.run_training(root, checkpoint)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertTrue(checkpoint.exists())

            import torch
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(saved["training"]["completed_steps"], 2)
            self.assertEqual(saved["tokenizer"], "utf8-byte-v1")
            self.assertTrue(saved["training"]["corpus_sha256"])
            self.assertTrue(saved["training"]["validation_sha256"])
            self.assertIsNotNone(saved["training"]["last_validation_loss"])

            resumed = self.run_training(
                root, checkpoint, extra=("--resume", str(checkpoint))
            )
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            saved_again = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(saved_again["training"]["completed_steps"], 4)
            self.assertIn("Resuming DREAM training at step 2", resumed.stdout)

    def test_rejects_same_train_and_validation_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "same.txt"
            corpus.write_text("DREAM text sample. " * 40, encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--data", str(corpus),
                    "--validation-data", str(corpus),
                    "--out", str(root / "bad.pt"),
                    "--steps", "1",
                    "--block-size", "8"
                ],
                capture_output=True, text=True, timeout=30
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be different", result.stderr)


if __name__ == "__main__":
    unittest.main()
