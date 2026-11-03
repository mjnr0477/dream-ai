"""End-to-end smoke tests for DREAM's own training/checkpoint pipeline."""
import hashlib
import json
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

    def test_bpe_train_resume_and_evaluate_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.txt").write_text(
                ("DREAM BPE training corpus with repeated Kiswahili and English words. " * 10),
                encoding="utf-8"
            )
            (root / "validation.txt").write_text(
                ("Independent validation corpus for tokenization and model loss. " * 10),
                encoding="utf-8"
            )
            (root / "test.txt").write_text(
                ("Held out evaluation corpus that is never used for tokenizer fitting. " * 10),
                encoding="utf-8"
            )
            checkpoint = root / "dream-bpe.pt"
            trained = self.run_training(
                root, checkpoint,
                extra=("--tokenizer", "bpe", "--tokenizer-vocab-size", "300")
            )
            self.assertEqual(trained.returncode, 0, trained.stdout + trained.stderr)
            import torch
            from dream_bpe import DreamBPETokenizer
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(saved["tokenizer"], DreamBPETokenizer.format_version)
            tokenizer = DreamBPETokenizer.from_dict(saved["tokenizer_data"])
            self.assertEqual(saved["config"]["vocab_size"], tokenizer.vocab_size)
            self.assertGreater(tokenizer.vocab_size, 256)

            resumed = self.run_training(
                root, checkpoint, extra=("--resume", str(checkpoint))
            )
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            saved_again = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(saved_again["training"]["completed_steps"], 4)
            self.assertEqual(saved_again["tokenizer_data"], saved["tokenizer_data"])

            report_path = root / "bpe-evaluation.json"
            evaluation = subprocess.run(
                [
                    sys.executable, str(EVALUATOR),
                    "--checkpoint", str(checkpoint),
                    "--data", str(root / "test.txt"),
                    "--batch-size", "2", "--batches", "2",
                    "--sample-tokens", "4", "--out", str(report_path)
                ],
                capture_output=True, text=True, timeout=60
            )
            self.assertEqual(evaluation.returncode, 0,
                             evaluation.stdout + evaluation.stderr)
            report = __import__("json").loads(report_path.read_text())
            self.assertEqual(report["tokenizer"], DreamBPETokenizer.format_version)
            self.assertGreater(report["bits_per_token"], 0)
            self.assertNotIn("bits_per_byte", report)

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

    def test_manifest_verifies_all_three_split_hashes_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contents = {
                "train": "DREAM licensed train split with enough distinct words. " * 8,
                "validation": "Independent validation split with separate source material. " * 8,
                "test": "Untouched held out test split for final model evaluation. " * 8
            }
            groups = {"train": ["source-train"], "validation": ["source-validation"],
                      "test": ["source-test"]}
            for split, content in contents.items():
                (root / f"{split}.txt").write_text(content, encoding="utf-8")
            manifest = {
                "schema": "dream-corpus-v1",
                "splits": {
                    split: {
                        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                        "source_group_ids": groups[split]
                    }
                    for split, content in contents.items()
                }
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            checkpoint = root / "manifest-verified.pt"
            result = self.run_training(
                root, checkpoint, extra=("--manifest", str(manifest_path))
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            import torch
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertTrue(saved["training"]["data_provenance_verified"])
            self.assertTrue(saved["training"]["corpus_manifest_sha256"])
            evaluation = subprocess.run(
                [
                    sys.executable, str(EVALUATOR),
                    "--checkpoint", str(checkpoint),
                    "--data", str(root / "test.txt"),
                    "--manifest", str(manifest_path),
                    "--batch-size", "1", "--batches", "1",
                    "--sample-tokens", "0",
                    "--out", str(root / "manifest-evaluation.json")
                ],
                capture_output=True, text=True, timeout=60
            )
            self.assertEqual(evaluation.returncode, 0, evaluation.stdout + evaluation.stderr)
            report = json.loads((root / "manifest-evaluation.json").read_text())
            self.assertEqual(report["corpus_manifest_sha256"], hashlib.sha256(manifest_path.read_bytes()).hexdigest())
            self.assertEqual(report["held_out_test_source_groups"], 1)
            resumed = self.run_training(
                root, checkpoint,
                extra=("--resume", str(checkpoint), "--manifest", str(manifest_path))
            )
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            unverified_resume = self.run_training(
                root, checkpoint, extra=("--resume", str(checkpoint))
            )
            self.assertNotEqual(unverified_resume.returncode, 0)
            self.assertIn("corpus manifest does not match", unverified_resume.stderr)

    def test_evaluator_rejects_manifest_mismatch_and_requires_manifest_for_verified_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contents = {
                "train": "DREAM licensed train split with enough distinct words. " * 8,
                "validation": "Independent validation split with separate source material. " * 8,
                "test": "Untouched held out test split for final model evaluation. " * 8
            }
            for split, content in contents.items():
                (root / f"{split}.txt").write_text(content, encoding="utf-8")
            manifest = {
                "schema": "dream-corpus-v1",
                "splits": {
                    split: {
                        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                        "source_group_ids": [f"source-{split}"]
                    }
                    for split, content in contents.items()
                }
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            checkpoint = root / "verified.pt"
            trained = self.run_training(
                root, checkpoint, extra=("--manifest", str(manifest_path))
            )
            self.assertEqual(trained.returncode, 0, trained.stdout + trained.stderr)

            no_manifest = subprocess.run(
                [sys.executable, str(EVALUATOR), "--checkpoint", str(checkpoint),
                 "--data", str(root / "test.txt"), "--sample-tokens", "0"],
                capture_output=True, text=True, timeout=30
            )
            self.assertNotEqual(no_manifest.returncode, 0)
            self.assertIn("requires --manifest", no_manifest.stderr)

            changed = root / "changed-manifest.json"
            manifest["splits"]["test"]["sha256"] = "0" * 64
            changed.write_text(json.dumps(manifest), encoding="utf-8")
            mismatch = subprocess.run(
                [sys.executable, str(EVALUATOR), "--checkpoint", str(checkpoint),
                 "--data", str(root / "test.txt"), "--manifest", str(changed),
                 "--sample-tokens", "0"],
                capture_output=True, text=True, timeout=30
            )
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn("does not match the manifest", mismatch.stderr)

            manifest["splits"]["test"]["sha256"] = hashlib.sha256(contents["test"].encode("utf-8")).hexdigest()
            manifest["splits"]["train"]["source_group_ids"] = []
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            invalid_groups = subprocess.run(
                [sys.executable, str(EVALUATOR), "--checkpoint", str(checkpoint),
                 "--data", str(root / "test.txt"), "--manifest", str(manifest_path),
                 "--sample-tokens", "0"],
                capture_output=True, text=True, timeout=30
            )
            self.assertNotEqual(invalid_groups.returncode, 0)
            self.assertIn("invalid train source-group metadata", invalid_groups.stderr)

    def test_manifest_rejects_split_hash_mismatch_and_source_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contents = {
                "train": "DREAM licensed train split with enough distinct words. " * 8,
                "validation": "Independent validation split with separate source material. " * 8,
                "test": "Untouched held out test split for final model evaluation. " * 8
            }
            for split, content in contents.items():
                (root / f"{split}.txt").write_text(content, encoding="utf-8")
            manifest = {
                "schema": "dream-corpus-v1",
                "splits": {
                    split: {
                        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                        "source_group_ids": [f"source-{split}"]
                    }
                    for split, content in contents.items()
                }
            }
            manifest_path = root / "manifest.json"
            manifest["splits"]["validation"]["source_group_ids"] = ["source-train"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = self.run_training(
                root, root / "leaked.pt", extra=("--manifest", str(manifest_path))
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("source-group leakage", result.stderr)

            manifest["splits"]["validation"]["source_group_ids"] = ["source-validation"]
            manifest["splits"]["validation"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = self.run_training(
                root, root / "tampered.pt", extra=("--manifest", str(manifest_path))
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("hash does not match the manifest", result.stderr)

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


class DreamLanguageSamplingTests(unittest.TestCase):
    def run_training(self, root, checkpoint, extra=()):
        return subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--data", str(root / "train.txt"),
                "--validation-data", str(root / "validation.txt"),
                "--out", str(checkpoint),
                "--steps", "2", "--batch-size", "2",
                "--grad-accum-steps", "2", "--block-size", "8",
                "--d-model", "16", "--layers", "1", "--heads", "4",
                "--eval-every", "1", "--checkpoint-every", "1",
                *extra
            ],
            capture_output=True, text=True, timeout=60
        )

    def test_splits_language_tagged_documents_into_independent_streams(self):
        from train import split_language_streams

        text = (
            "<|language:en|>\nEnglish lesson with enough words for several training windows.\n\n"
            "<|language:sw|>\nSomo la Kiswahili lina maneno ya kutosha kwa mafunzo.\n\n"
            "<|language:en|>\nA second English document adds more training material.\n"
        )
        streams = split_language_streams(text, _ByteTokenizerForSampling(), 4)
        self.assertEqual(set(streams), {"en", "sw"})
        self.assertGreater(len(streams["en"]), len(streams["sw"]))
        self.assertTrue(all(len(tokens) >= 6 for tokens in streams.values()))

    def test_rejects_missing_language_markers_and_short_language_stream(self):
        from train import split_language_streams

        with self.assertRaisesRegex(ValueError, "requires"):
            split_language_streams("Plain text has no language marker.", _ByteTokenizerForSampling(), 4)
        with self.assertRaisesRegex(ValueError, "insufficient"):
            split_language_streams(
                "<|language:en|>\nshort\n\n<|language:sw|>\nThis Kiswahili stream has enough words for training windows.",
                _ByteTokenizerForSampling(), 8
            )

    def test_balanced_batches_return_valid_next_token_pairs(self):
        from train import balanced_batches
        import torch

        streams = {
            "en": torch.tensor(list(range(40)), dtype=torch.long),
            "sw": torch.tensor(list(range(50, 90)), dtype=torch.long)
        }
        x, y = balanced_batches(streams, batch_size=12, block_size=6, device="cpu")
        self.assertEqual(tuple(x.shape), (12, 6))
        self.assertEqual(tuple(y.shape), (12, 6))
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
        self.assertTrue(torch.all(x < 90))

    def test_balanced_training_records_policy_and_rejects_resume_policy_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.txt").write_text(
                "<|language:en|>\n" + ("English training text with varied words. " * 8)
                + "\n\n<|language:sw|>\n"
                + ("Haya ni maandishi ya Kiswahili yenye maneno tofauti. " * 8),
                encoding="utf-8"
            )
            (root / "validation.txt").write_text(
                "Independent validation text with enough tokens. " * 10,
                encoding="utf-8"
            )
            checkpoint = root / "balanced.pt"
            result = self.run_training(
                root, checkpoint, extra=("--language-sampling", "balanced")
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            import torch
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(saved["training"]["language_sampling"], "balanced")
            self.assertIn("python_rng_state", saved)
            self.assertEqual(
                set(saved["training"]["training_tokens_by_language"]), {"en", "sw"}
            )
            mismatch = self.run_training(
                root, checkpoint,
                extra=("--resume", str(checkpoint), "--language-sampling", "proportional")
            )
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn("language sampling mode does not match", mismatch.stderr)


class _ByteTokenizerForSampling:
    def encode(self, text):
        return list(text.encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
