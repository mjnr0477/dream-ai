import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

TRAIN_SCRIPT = Path(__file__).with_name("train.py")


class TrainingPipelineTests(unittest.TestCase):
    def test_train_save_and_resume_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.txt"
            validation_path = root / "validation.txt"
            checkpoint_path = root / "dream-smoke.pt"
            train_path.write_text(
                ("DREAM trains its own language model from random initialization. " * 20),
                encoding="utf-8"
            )
            validation_path.write_text(
                ("Independent validation examples measure generalization. " * 20),
                encoding="utf-8"
            )

            common = [
                sys.executable, str(TRAIN_SCRIPT),
                "--data", str(train_path),
                "--validation-data", str(validation_path),
                "--out", str(checkpoint_path),
                "--batch-size", "2",
                "--block-size", "8",
                "--d-model", "16",
                "--layers", "1",
                "--heads", "4",
                "--eval-every", "1",
                "--checkpoint-every", "1"
            ]
            first = subprocess.run(
                common + ["--steps", "2"],
                capture_output=True, text=True, timeout=120
            )
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertTrue(checkpoint_path.exists())

            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            self.assertEqual(checkpoint["training"]["completed_steps"], 2)
            self.assertIn("optimizer_state", checkpoint)
            self.assertEqual(checkpoint["training"]["initialization"], "random")

            resumed = subprocess.run(
                common + ["--steps", "1", "--resume", str(checkpoint_path)],
                capture_output=True, text=True, timeout=120
            )
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            final_checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            self.assertEqual(final_checkpoint["training"]["completed_steps"], 3)
            self.assertEqual(
                final_checkpoint["training"]["initialization"],
                "resumed-from-dream-checkpoint"
            )
            self.assertEqual(
                final_checkpoint["training"]["corpus_sha256"],
                checkpoint["training"]["corpus_sha256"]
            )


if __name__ == "__main__":
    unittest.main()
