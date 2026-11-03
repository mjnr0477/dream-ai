"""Tests for DREAM's assistant-response-only supervised fine-tuning."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from dream_transformer import DreamConfig, DreamTransformer, DreamByteTokenizer
from sft import load_examples

SCRIPT = Path(__file__).with_name("sft.py")


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

    def tearDown(self):
        self.temp.cleanup()

    def write_records(self, path, source_prefix):
        rows = []
        for index in range(3):
            rows.append({
                "id": f"{source_prefix}-{index}",
                "source": f"Reviewed DREAM source {source_prefix}-{index}",
                "source_group": f"{source_prefix}-group-{index}",
                "license": "CC0-1.0",
                "rights_confirmed": True,
                "privacy_reviewed": True,
                "quality_reviewed": True,
                "language": "en",
                "messages": [
                    {"role": "user", "content": f"Question number {index}?"},
                    {"role": "assistant", "content": f"Answer number {index}."}
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
            self.assertIn(b"Answer number", supervised)

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

        resumed = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--checkpoint", str(self.base),
                "--train-data", str(train_path),
                "--validation-data", str(val_path),
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
                "--validation-data", str(val_path)
            ],
            capture_output=True, text=True, timeout=30
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("completed pretraining steps", result.stderr)


if __name__ == "__main__":
    unittest.main()
