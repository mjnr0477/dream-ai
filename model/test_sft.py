"""Tests for DREAM's assistant-response-only supervised fine-tuning."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from dream_transformer import DreamConfig, DreamTransformer, DreamByteTokenizer
from sft import load_examples, record_signatures

SCRIPT = Path(__file__).with_name("sft.py")
EVALUATOR = Path(__file__).with_name("evaluate.py")


class DreamSFTTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = DreamConfig(
            vocab_size=256, block_size=64, d_model=16,
            n_heads=4, n_layers=1, dropout=0.0
        )
        self.base = self.root / "pretrained.pt"
        model = DreamTransformer(self.config)
        torch.save({
            "model_state": model.state_dict(),
            "config": self.config.__dict__,
            "tokenizer": "utf8-byte-v1",
            "training": {"completed_steps": 2, "initialization": "random"}
        }, self.base)
        self.test_data = self.root / "pretraining-test.txt"
        self.test_data.write_text(
            "This is a separate held out corpus used only to test evaluation isolation. " * 4,
            encoding="utf-8"
        )
        test_hash = hashlib.sha256(self.test_data.read_bytes()).hexdigest()
        self.manifest = self.root / "pretraining-manifest.json"
        self.write_manifest(["pretraining-test-group"], test_hash)

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self, test_groups, test_hash=None):
        if test_hash is None:
            test_hash = hashlib.sha256(self.test_data.read_bytes()).hexdigest()
        self.manifest.write_text(json.dumps({
            "schema": "dream-corpus-v1",
            "splits": {
                "test": {
                    "source_group_ids": test_groups,
                    "sha256": test_hash
                }
            }
        }), encoding="utf-8")

    def write_records(self, path, source_prefix):
        rows = []
        for index in range(3):
            rows.append({
                "id": f"{source_prefix}-{index}",
                "source": f"Reviewed DREAM source {source_prefix}-{index}",
                "source_group": f"{source_prefix}-group-{index}",
                "license": "CC0-1.0",
                "source_date": "2026-01-15",
                "rights_confirmed": True,
                "privacy_reviewed": True,
                "quality_reviewed": True,
                "language": "en",
                "messages": [
                    {"role": "user", "content": f"Question from {source_prefix} number {index}?"},
                    {"role": "assistant", "content": f"Answer from {source_prefix} number {index}."}
                ]
            })
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8"
        )

    def test_masks_user_prompt_and_trains_assistant_answers(self):
        train_path = self.root / "train.jsonl"
        val_path = self.root / "validation.jsonl"
        self.write_records(train_path, "train")
        self.write_records(val_path, "validation")
        examples = load_examples(
            train_path, DreamByteTokenizer(), self.config.block_size
        )
        self.assertEqual(len(examples), 3)
        for input_ids, targets in examples:
            self.assertEqual(len(input_ids), len(targets))
            self.assertIn(-100, targets)
            self.assertTrue(any(target != -100 for target in targets))
            supervised = bytes(target for target in targets if target != -100)
            self.assertIn(b"Answer from train number", supervised)

    def test_rejects_duplicate_dialogue_inside_one_sft_split(self):
        train_path = self.root / "train.jsonl"
        self.write_records(train_path, "train")
        rows = [json.loads(line) for line in train_path.read_text().splitlines()]
        rows[1]["messages"] = rows[0]["messages"]
        train_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8"
        )
        with self.assertRaisesRegex(SystemExit, "Duplicate SFT dialogue content within"):
            record_signatures(train_path)

    def test_rejects_duplicate_id_inside_one_sft_split(self):
        train_path = self.root / "train.jsonl"
        self.write_records(train_path, "train")
        rows = [json.loads(line) for line in train_path.read_text().splitlines()]
        rows[1]["id"] = rows[0]["id"]
        train_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8"
        )
        with self.assertRaisesRegex(SystemExit, "Duplicate SFT record ID within"):
            record_signatures(train_path)

    def test_sft_rejects_missing_or_invalid_source_date(self):
        train_path = self.root / "train.jsonl"
        self.write_records(train_path, "train")
        rows = [json.loads(line) for line in train_path.read_text().splitlines()]
        for value in (None, "15-01-2026", "2026-02-30"):
            candidate = [dict(row) for row in rows]
            if value is None:
                candidate[0].pop("source_date")
            else:
                candidate[0]["source_date"] = value
            path = self.root / "invalid-sft.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in candidate),
                encoding="utf-8"
            )
            with self.subTest(value=value):
                with self.assertRaises(SystemExit) as raised:
                    load_examples(path, DreamByteTokenizer(), self.config.block_size)
                self.assertIn("source_date", str(raised.exception))

    def test_sft_training_creates_resumable_checkpoint(self):
        train_path = self.root / "train.jsonl"
        val_path = self.root / "validation.jsonl"
        self.write_records(train_path, "train")
        self.write_records(val_path, "validation")
        output = self.root / "dream-instruct.pt"
        result = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--checkpoint", str(self.base),
                "--train-data", str(train_path),
                "--validation-data", str(val_path),
                "--pretraining-manifest", str(self.manifest),
                "--pretraining-test-data", str(self.test_data),
                "--out", str(output),
                "--steps", "2",
                "--batch-size", "2",
                "--grad-accum-steps", "2",
                "--eval-every", "1",
                "--eval-batches", "2",
                "--checkpoint-every", "1"
            ],
            capture_output=True, text=True, timeout=120
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(output.exists())
        saved = torch.load(output, map_location="cpu", weights_only=True)
        self.assertEqual(saved["sft_training"]["completed_steps"], 2)
        self.assertEqual(saved["sft_training"]["method"],
                         "assistant-response-only-supervised-finetuning")
        self.assertEqual(saved["sft_training"]["effective_batch_size"], 4)
        self.assertTrue(saved["base_checkpoint_sha256"])
        self.assertTrue(saved["sft_training"]["train_sha256"])
        self.assertTrue(saved["sft_training"]["validation_sha256"])
        self.assertEqual(
            saved["sft_training"]["pretraining_test_sha256"],
            hashlib.sha256(self.test_data.read_bytes()).hexdigest()
        )
        self.assertIn("python_rng_state", saved)
        self.assertIn("torch_rng_state", saved)

        resumed = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--checkpoint", str(self.base),
                "--train-data", str(train_path),
                "--validation-data", str(val_path),
                "--pretraining-manifest", str(self.manifest),
                "--pretraining-test-data", str(self.test_data),
                "--out", str(output),
                "--steps", "1",
                "--batch-size", "2",
                "--grad-accum-steps", "2",
                "--eval-every", "1",
                "--eval-batches", "2",
                "--checkpoint-every", "1",
                "--resume", str(output)
            ],
            capture_output=True, text=True, timeout=120
        )
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        saved = torch.load(output, map_location="cpu", weights_only=True)
        self.assertEqual(saved["sft_training"]["completed_steps"], 3)

        leaked = subprocess.run(
            [
                sys.executable, str(EVALUATOR),
                "--checkpoint", str(output),
                "--data", str(train_path),
                "--sample-tokens", "0",
                "--out", str(self.root / "leaked-evaluation.json")
            ],
            capture_output=True, text=True, timeout=30
        )
        self.assertNotEqual(leaked.returncode, 0)
        self.assertIn("matches training or validation", leaked.stderr)

        valid_eval = subprocess.run(
            [
                sys.executable, str(EVALUATOR),
                "--checkpoint", str(output),
                "--data", str(self.test_data),
                "--sample-tokens", "0",
                "--batches", "1", "--batch-size", "1",
                "--out", str(self.root / "valid-evaluation.json")
            ],
            capture_output=True, text=True, timeout=30
        )
        self.assertEqual(valid_eval.returncode, 0, valid_eval.stdout + valid_eval.stderr)

    def test_rejects_overlapping_source_groups(self):
        train_path = self.root / "train.jsonl"
        val_path = self.root / "validation.jsonl"
        self.write_records(train_path, "train")
        self.write_records(val_path, "validation")
        train_rows = [json.loads(line) for line in train_path.read_text().splitlines()]
        val_rows = [json.loads(line) for line in val_path.read_text().splitlines()]
        for train_row, val_row in zip(train_rows, val_rows):
            val_row["source_group"] = train_row["source_group"]
        val_path.write_text(
            "".join(json.dumps(row) + "\n" for row in val_rows),
            encoding="utf-8"
        )
        result = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--checkpoint", str(self.base),
                "--train-data", str(train_path),
                "--validation-data", str(val_path),
                "--pretraining-manifest", str(self.manifest),
                "--pretraining-test-data", str(self.test_data),
                "--steps", "1"
            ],
            capture_output=True, text=True, timeout=30
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source groups must be disjoint", result.stderr)

    def test_rejects_duplicate_record_ids_across_sft_splits(self):
        train_path = self.root / "train.jsonl"
        val_path = self.root / "validation.jsonl"
        self.write_records(train_path, "train")
        self.write_records(val_path, "validation")
        train_rows = [json.loads(line) for line in train_path.read_text().splitlines()]
        val_rows = [json.loads(line) for line in val_path.read_text().splitlines()]
        val_rows[0]["id"] = train_rows[0]["id"]
        val_path.write_text(
            "".join(json.dumps(row) + "\n" for row in val_rows),
            encoding="utf-8"
        )
        result = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--checkpoint", str(self.base),
                "--train-data", str(train_path),
                "--validation-data", str(val_path),
                "--pretraining-manifest", str(self.manifest),
                "--pretraining-test-data", str(self.test_data),
                "--steps", "1"
            ],
            capture_output=True, text=True, timeout=30
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("record IDs must be disjoint", result.stderr)

    def test_rejects_duplicate_dialogue_content_across_sft_splits(self):
        train_path = self.root / "train.jsonl"
        val_path = self.root / "validation.jsonl"
        self.write_records(train_path, "train")
        self.write_records(val_path, "validation")
        train_rows = [json.loads(line) for line in train_path.read_text().splitlines()]
        val_rows = [json.loads(line) for line in val_path.read_text().splitlines()]
        val_rows[0]["messages"] = train_rows[0]["messages"]
        val_path.write_text(
            "".join(json.dumps(row) + "\n" for row in val_rows),
            encoding="utf-8"
        )
        result = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--checkpoint", str(self.base),
                "--train-data", str(train_path),
                "--validation-data", str(val_path),
                "--pretraining-manifest", str(self.manifest),
                "--pretraining-test-data", str(self.test_data),
                "--steps", "1"
            ],
            capture_output=True, text=True, timeout=30
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Duplicate SFT dialogue content", result.stderr)

    def test_rejects_test_corpus_that_does_not_match_manifest(self):
        train_path = self.root / "train.jsonl"
        val_path = self.root / "validation.jsonl"
        self.write_records(train_path, "train")
        self.write_records(val_path, "validation")
        self.test_data.write_text(
            "Changed after the pretraining manifest was generated. " * 5,
            encoding="utf-8"
        )
        result = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--checkpoint", str(self.base),
                "--train-data", str(train_path),
                "--validation-data", str(val_path),
                "--pretraining-manifest", str(self.manifest),
                "--pretraining-test-data", str(self.test_data),
                "--steps", "1"
            ],
            capture_output=True, text=True, timeout=30
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("hash does not match", result.stderr)

    def test_rejects_sft_sources_used_by_pretraining_test(self):
        train_path = self.root / "train.jsonl"
        val_path = self.root / "validation.jsonl"
        self.write_records(train_path, "train")
        self.write_records(val_path, "validation")
        self.write_manifest(["train-group-0"])
        result = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--checkpoint", str(self.base),
                "--train-data", str(train_path),
                "--validation-data", str(val_path),
                "--pretraining-manifest", str(self.manifest),
                "--pretraining-test-data", str(self.test_data),
                "--steps", "1"
            ],
            capture_output=True, text=True, timeout=30
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pretraining held-out test groups", result.stderr)

    def test_refuses_random_untrained_base(self):
        untrained = self.root / "untrained.pt"
        model = DreamTransformer(self.config)
        torch.save({
            "model_state": model.state_dict(),
            "config": self.config.__dict__,
            "tokenizer": "utf8-byte-v1",
            "training": {"completed_steps": 0}
        }, untrained)
        train_path = self.root / "train.jsonl"
        val_path = self.root / "validation.jsonl"
        self.write_records(train_path, "train")
        self.write_records(val_path, "validation")
        result = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--checkpoint", str(untrained),
                "--train-data", str(train_path),
                "--validation-data", str(val_path),
                "--pretraining-manifest", str(self.manifest),
                "--pretraining-test-data", str(self.test_data)
            ],
            capture_output=True, text=True, timeout=30
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("completed pretraining steps", result.stderr)


if __name__ == "__main__":
    unittest.main()
