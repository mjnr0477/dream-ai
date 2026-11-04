"""End-to-end smoke tests for DREAM's own training/checkpoint pipeline."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("train.py")
EVALUATOR = Path(__file__).with_name("evaluate.py")


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
                "--grad-accum-steps", "2",
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
            (root / "train.txt").write_text(
                ("DREAM authored training sample with varied words and characters. " * 8),
                encoding="utf-8"
            )
            (root / "validation.txt").write_text(
                ("Independent validation material for testing language loss. " * 8),
                encoding="utf-8"
            )
            (root / "test.txt").write_text(
                ("Completely held out test material for DREAM model evaluation. " * 8),
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
            self.assertEqual(saved["training"]["eval_batches"], 10)
            self.assertEqual(saved["training"]["grad_accum_steps"], 2)
            self.assertEqual(saved["training"]["effective_batch_size"], 4)
            self.assertIn("torch_rng_state", saved)
            self.assertIn("cuda_rng_state", saved)

            resumed = self.run_training(
                root, checkpoint, extra=("--resume", str(checkpoint))
            )
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            saved_again = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(saved_again["training"]["completed_steps"], 4)
            self.assertIn("Resuming DREAM training at step 2", resumed.stdout)

            report_path = root / "evaluation.json"
            evaluation = subprocess.run(
                [
                    sys.executable, str(EVALUATOR),
                    "--checkpoint", str(checkpoint),
                    "--data", str(root / "test.txt"),
                    "--batch-size", "2",
                    "--batches", "2",
                    "--sample-tokens", "4",
                    "--out", str(report_path)
                ],
                capture_output=True, text=True, timeout=60
            )
            self.assertEqual(evaluation.returncode, 0,
                             evaluation.stdout + evaluation.stderr)
            report = __import__("json").loads(report_path.read_text())
            self.assertEqual(report["schema"], "dream-evaluation-v1")
            self.assertEqual(report["batches"], 2)
            self.assertGreater(report["bits_per_byte"], 0)
            self.assertTrue(report["checkpoint_sha256"])
            self.assertTrue(report["evaluation_corpus_sha256"])
            self.assertIn("does not certify safety", " ".join(report["limitations"]))


    def test_bpe_training_checkpoint_evaluates_and_audits(self):
        import json
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.txt").write_text(
                "DREAM multilingual training text — habari dunia. " * 16,
                encoding="utf-8"
            )
            (root / "validation.txt").write_text(
                "Separate held out validation material — lugha mbalimbali. " * 12,
                encoding="utf-8"
            )
            (root / "test.txt").write_text(
                "Independent evaluation sample — Kiswahili and English. " * 12,
                encoding="utf-8"
            )
            checkpoint = root / "dream-bpe.pt"
            trained = self.run_training(
                root, checkpoint, extra=("--tokenizer", "bpe", "--vocab-size", "300")
            )
            self.assertEqual(trained.returncode, 0, trained.stdout + trained.stderr)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(saved["tokenizer"]["format"], "dream-bpe-v1")
            self.assertEqual(saved["config"]["vocab_size"], saved["tokenizer"]["vocab_size"])
            self.assertGreater(saved["config"]["vocab_size"], 256)

            audit = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("audit_checkpoint.py")),
                 "--checkpoint", str(checkpoint)],
                capture_output=True, text=True, timeout=30
            )
            self.assertEqual(audit.returncode, 0, audit.stdout + audit.stderr)
            self.assertEqual(json.loads(audit.stdout)["tokenizer"], "dream-bpe-v1")

            evaluation = subprocess.run(
                [sys.executable, str(EVALUATOR),
                 "--checkpoint", str(checkpoint),
                 "--data", str(root / "test.txt"),
                 "--batch-size", "2", "--batches", "2",
                 "--sample-tokens", "4", "--out", str(root / "bpe-eval.json")],
                capture_output=True, text=True, timeout=60
            )
            self.assertEqual(evaluation.returncode, 0,
                             evaluation.stdout + evaluation.stderr)

            resumed = self.run_training(
                root, checkpoint,
                extra=("--tokenizer", "bpe", "--vocab-size", "300",
                       "--resume", str(checkpoint))
            )
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            resumed_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(resumed_checkpoint["training"]["completed_steps"], 4)

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

    def test_rejects_identical_content_in_different_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = "DREAM repeated training material for duplicate hash testing. " * 8
            (root / "train.txt").write_text(content, encoding="utf-8")
            (root / "validation.txt").write_text(content, encoding="utf-8")
            result = self.run_training(root, root / "duplicate.pt")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("content hashes must differ", result.stderr)

    def test_evaluator_rejects_training_data_as_test_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.txt").write_text(
                "DREAM training text for evaluator leakage check. " * 8,
                encoding="utf-8"
            )
            (root / "validation.txt").write_text(
                "DREAM validation text for evaluator leakage check. " * 8,
                encoding="utf-8"
            )
            (root / "test.txt").write_text(
                "DREAM held out test text for evaluator leakage check. " * 8,
                encoding="utf-8"
            )
            checkpoint = root / "dream.pt"
            trained = self.run_training(root, checkpoint)
            self.assertEqual(trained.returncode, 0, trained.stdout + trained.stderr)
            (root / "leaked-test.txt").write_text(
                (root / "train.txt").read_text(encoding="utf-8"),
                encoding="utf-8"
            )
            result = subprocess.run(
                [
                    sys.executable, str(EVALUATOR),
                    "--checkpoint", str(checkpoint),
                    "--data", str(root / "leaked-test.txt"),
                    "--sample-tokens", "0",
                    "--out", str(root / "leak.json")
                ],
                capture_output=True, text=True, timeout=30
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("matches training or validation", result.stderr)


if __name__ == "__main__":
    unittest.main()
