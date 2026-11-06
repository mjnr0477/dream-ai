"""End-to-end smoke tests for DREAM's own training/checkpoint pipeline."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from corpus_manifest_test_utils import write_test_corpus_manifest

SCRIPT = Path(__file__).with_name("train.py")
EVALUATOR = Path(__file__).with_name("evaluate.py")


class DreamTrainingSmokeTests(unittest.TestCase):
    def run_training(self, root, checkpoint, extra=(), create_manifest=True,
                     license_name="CC0-1.0",
                     license_url="https://creativecommons.org/publicdomain/zero/1.0/",
                     rights_basis="original-authorship"):
        root = Path(root)
        manifest_path = root / "manifest.json"
        if create_manifest:
            write_test_corpus_manifest(
                root, root / "train.txt", root / "validation.txt",
                license_name=license_name, license_url=license_url,
                rights_basis=rights_basis
            )
        return subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--data", str(root / "train.txt"),
                "--validation-data", str(root / "validation.txt"),
                "--corpus-manifest", str(manifest_path),
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

    def test_pretraining_requires_a_corpus_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.txt").write_text("DREAM training text. " * 20, encoding="utf-8")
            (root / "validation.txt").write_text("DREAM validation text. " * 20, encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--data", str(root / "train.txt"),
                    "--validation-data", str(root / "validation.txt"),
                    "--out", str(root / "missing-manifest.pt"),
                    "--steps", "1", "--block-size", "8"
                ],
                capture_output=True, text=True, timeout=30
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--corpus-manifest", result.stderr)

    def test_pretraining_blocks_noncommercial_license_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.txt").write_text("DREAM training text. " * 30, encoding="utf-8")
            (root / "validation.txt").write_text("DREAM validation text. " * 30, encoding="utf-8")
            result = self.run_training(
                root, root / "blocked.pt",
                license_name="CC-BY-NC-SA-4.0",
                license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
                rights_basis="open-license"
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("noncommercial_restriction", result.stderr)

    def test_pretraining_rejects_text_changed_after_manifest_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.txt").write_text("DREAM training text. " * 30, encoding="utf-8")
            (root / "validation.txt").write_text("DREAM validation text. " * 30, encoding="utf-8")
            write_test_corpus_manifest(root, root / "train.txt", root / "validation.txt")
            with (root / "train.txt").open("a", encoding="utf-8") as handle:
                handle.write("Unmanifested text must never be trained.")
            result = self.run_training(root, root / "tampered.pt", create_manifest=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not match its corpus manifest", result.stderr)

    def test_pretraining_proves_provenance_matches_exact_text_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.txt").write_text("DREAM training text. " * 30, encoding="utf-8")
            (root / "validation.txt").write_text("DREAM validation text. " * 30, encoding="utf-8")
            manifest_path = write_test_corpus_manifest(root, root / "train.txt", root / "validation.txt")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            provenance_meta = manifest["splits"]["train"]["provenance"]
            provenance_path = root / provenance_meta["path"]
            row = json.loads(provenance_path.read_text(encoding="utf-8").splitlines()[0])
            row["text_sha256"] = __import__("hashlib").sha256(
                b"some other text that was never approved"
            ).hexdigest()
            provenance_path.write_text(json.dumps(row) + "\\n", encoding="utf-8")
            provenance_meta["sha256"] = __import__("hashlib").sha256(
                provenance_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = self.run_training(
                root, root / "tampered-provenance.pt", create_manifest=False
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("record text does not match its license provenance", result.stderr)

    def test_resume_requires_the_same_corpus_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.txt").write_text("DREAM training text. " * 30, encoding="utf-8")
            (root / "validation.txt").write_text("DREAM validation text. " * 30, encoding="utf-8")
            checkpoint = root / "manifest-resume.pt"
            first = self.run_training(root, checkpoint)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["post-training-edit"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            resumed = self.run_training(
                root, checkpoint,
                extra=("--resume", str(checkpoint)),
                create_manifest=False
            )
            self.assertNotEqual(resumed.returncode, 0)
            self.assertIn("corpus manifest does not match", resumed.stderr)

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
            self.assertEqual(saved["training"]["eval_every"], 1)
            self.assertEqual(saved["training"]["max_grad_norm"], 1.0)
            self.assertEqual(saved["training"]["grad_accum_steps"], 2)
            self.assertEqual(saved["training"]["effective_batch_size"], 4)
            self.assertIn("torch_rng_state", saved)
            self.assertIn("cuda_rng_state", saved)

            mismatched_schedule = self.run_training(
                root, checkpoint,
                extra=("--resume", str(checkpoint), "--eval-every", "2")
            )
            self.assertNotEqual(mismatched_schedule.returncode, 0)
            self.assertIn("evaluation interval does not match", mismatched_schedule.stderr)

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
            self.assertIn('"training_tokens"', trained.stdout)
            self.assertIn('"validation_tokens"', trained.stdout)
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
            manifest = write_test_corpus_manifest(root, corpus, corpus)
            result = subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--data", str(corpus),
                    "--validation-data", str(corpus),
                    "--corpus-manifest", str(manifest),
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


    def test_activation_checkpointing_is_recorded_and_required_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.txt").write_text(
                "DREAM checkpointed training corpus with varied tokens. " * 12,
                encoding="utf-8"
            )
            (root / "validation.txt").write_text(
                "Independent DREAM validation corpus for checkpointing. " * 12,
                encoding="utf-8"
            )
            checkpoint = root / "checkpointed.pt"
            trained = self.run_training(
                root, checkpoint, extra=("--gradient-checkpointing",)
            )
            self.assertEqual(trained.returncode, 0, trained.stdout + trained.stderr)

            import torch
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertTrue(saved["training"]["gradient_checkpointing"])

            mismatched = self.run_training(
                root, checkpoint,
                extra=("--resume", str(checkpoint))
            )
            self.assertNotEqual(mismatched.returncode, 0)
            self.assertIn(
                "gradient checkpointing setting does not match",
                mismatched.stderr
            )

            resumed = self.run_training(
                root, checkpoint,
                extra=("--resume", str(checkpoint), "--gradient-checkpointing")
            )
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)

    def test_interrupted_resume_matches_uninterrupted_training_exactly(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.txt").write_text(
                "DREAM reproducibility training corpus with varied words. " * 12,
                encoding="utf-8"
            )
            (root / "validation.txt").write_text(
                "Independent DREAM validation corpus with other words. " * 12,
                encoding="utf-8"
            )
            continuous_checkpoint = root / "continuous.pt"
            resumed_checkpoint = root / "resumed.pt"

            continuous = self.run_training(
                root, continuous_checkpoint,
                extra=("--steps", "4", "--gradient-checkpointing")
            )
            self.assertEqual(
                continuous.returncode, 0, continuous.stdout + continuous.stderr
            )

            interrupted = self.run_training(
                root, resumed_checkpoint, extra=("--gradient-checkpointing",)
            )
            self.assertEqual(
                interrupted.returncode, 0, interrupted.stdout + interrupted.stderr
            )
            mismatch = self.run_training(
                root, resumed_checkpoint,
                extra=("--resume", str(resumed_checkpoint), "--steps", "2")
            )
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn("gradient checkpointing setting does not match", mismatch.stderr)

            resumed = self.run_training(
                root, resumed_checkpoint,
                extra=("--resume", str(resumed_checkpoint), "--steps", "2",
                       "--gradient-checkpointing")
            )
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)

            expected = torch.load(
                continuous_checkpoint, map_location="cpu", weights_only=True
            )
            actual = torch.load(
                resumed_checkpoint, map_location="cpu", weights_only=True
            )
            self.assertEqual(expected["training"]["completed_steps"], 4)
            self.assertEqual(actual["training"]["completed_steps"], 4)
            self.assertTrue(actual["training"]["gradient_checkpointing"])
            self.assertEqual(expected["model_state"].keys(), actual["model_state"].keys())
            for name in expected["model_state"]:
                with self.subTest(parameter=name):
                    self.assertTrue(torch.equal(
                        expected["model_state"][name], actual["model_state"][name]
                    ))
            self.assertEqual(
                expected["optimizer_state"]["state"].keys(),
                actual["optimizer_state"]["state"].keys()
            )
            for parameter_id, expected_state in expected["optimizer_state"]["state"].items():
                actual_state = actual["optimizer_state"]["state"][parameter_id]
                self.assertEqual(expected_state.keys(), actual_state.keys())
                for key, expected_value in expected_state.items():
                    with self.subTest(parameter=parameter_id, state=key):
                        if isinstance(expected_value, torch.Tensor):
                            self.assertTrue(torch.equal(
                                expected_value, actual_state[key]
                            ))
                        else:
                            self.assertEqual(expected_value, actual_state[key])
            self.assertTrue(torch.equal(
                expected["torch_rng_state"], actual["torch_rng_state"]
            ))


    def test_gradient_clipping_threshold_is_validated_and_locked_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.txt").write_text(
                "DREAM gradient safety training material with enough tokens. " * 8,
                encoding="utf-8"
            )
            (root / "validation.txt").write_text(
                "Independent validation material for gradient safety tests. " * 8,
                encoding="utf-8"
            )
            checkpoint = root / "gradient-safe.pt"
            trained = self.run_training(
                root, checkpoint, extra=("--max-grad-norm", "0.5")
            )
            self.assertEqual(trained.returncode, 0, trained.stdout + trained.stderr)

            import torch
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(saved["training"]["max_grad_norm"], 0.5)

            mismatched = self.run_training(
                root, checkpoint,
                extra=("--resume", str(checkpoint), "--max-grad-norm", "1.0")
            )
            self.assertNotEqual(mismatched.returncode, 0)
            self.assertIn("max gradient norm does not match", mismatched.stderr)

            for invalid in ("0", "-1", "nan", "inf"):
                with self.subTest(invalid=invalid):
                    rejected = self.run_training(
                        root, root / "invalid.pt",
                        extra=("--max-grad-norm", invalid)
                    )
                    self.assertNotEqual(rejected.returncode, 0)
                    self.assertIn(
                        "max-grad-norm must be finite and positive",
                        rejected.stderr
                    )


if __name__ == "__main__":
    unittest.main()
