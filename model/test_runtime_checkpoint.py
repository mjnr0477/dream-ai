"""Focused tests for fail-closed DREAM checkpoint loading."""
import tempfile
import unittest
from pathlib import Path

import torch

from dream_transformer import DreamConfig, DreamTransformer
from runtime_checkpoint import load_validated_checkpoint


class RuntimeCheckpointValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "checkpoint.pt"
        self.config = DreamConfig(
            vocab_size=256, block_size=32, d_model=16,
            n_heads=4, n_layers=1, dropout=0.0
        )
        self.model = DreamTransformer(self.config)
        self.checkpoint = {
            "model_state": self.model.state_dict(),
            "config": self.config.__dict__.copy(),
            "tokenizer": "utf8-byte-v1",
            "training": {"completed_steps": 2}
        }

    def save(self, payload=None):
        torch.save(self.checkpoint if payload is None else payload, self.path)
        return self.path

    def test_loads_valid_checkpoint_and_returns_metadata(self):
        model, tokenizer, config, metadata = load_validated_checkpoint(self.save())
        self.assertIsInstance(model, DreamTransformer)
        self.assertEqual(tokenizer.decode(tokenizer.encode("Habari 🇰🇪")), "Habari 🇰🇪")
        self.assertEqual(config.__dict__, self.config.__dict__)
        self.assertEqual(metadata["completed_steps"], 2)

    def test_rejects_unexpected_config_fields(self):
        self.checkpoint["config"]["unreviewed_architecture_flag"] = True
        with self.assertRaisesRegex(ValueError, "unexpected or missing fields"):
            load_validated_checkpoint(self.save())

    def test_rejects_missing_config_fields(self):
        del self.checkpoint["config"]["dropout"]
        with self.assertRaisesRegex(ValueError, "unexpected or missing fields"):
            load_validated_checkpoint(self.save())

    def test_rejects_missing_and_unexpected_state_keys(self):
        self.checkpoint["model_state"] = dict(self.checkpoint["model_state"])
        self.checkpoint["model_state"].pop("final_norm.weight")
        with self.assertRaisesRegex(ValueError, "keys do not match"):
            load_validated_checkpoint(self.save())

    def test_rejects_non_tensor_state_values(self):
        self.checkpoint["model_state"] = dict(self.checkpoint["model_state"])
        self.checkpoint["model_state"]["final_norm.weight"] = [1.0] * 16
        with self.assertRaisesRegex(ValueError, "is not a tensor"):
            load_validated_checkpoint(self.save())

    def test_rejects_non_finite_model_weights(self):
        self.checkpoint["model_state"] = dict(self.checkpoint["model_state"])
        bad = self.checkpoint["model_state"]["final_norm.weight"].clone()
        bad[0] = float("nan")
        self.checkpoint["model_state"]["final_norm.weight"] = bad
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            load_validated_checkpoint(self.save())

    def test_rejects_invalid_training_step_metadata(self):
        for value in (0, -1, True, 1.5, "2"):
            with self.subTest(value=value):
                self.checkpoint["training"]["completed_steps"] = value
                with self.assertRaisesRegex(ValueError, "completed DREAM training"):
                    load_validated_checkpoint(self.save())

    def test_rejects_tokenizer_vocab_mismatch(self):
        self.checkpoint["config"]["vocab_size"] = 300
        with self.assertRaisesRegex(ValueError, "Tokenizer vocabulary size"):
            load_validated_checkpoint(self.save())

    def test_rejects_unsafe_or_corrupt_checkpoint_file(self):
        self.path.write_bytes(b"not a torch checkpoint")
        with self.assertRaisesRegex(ValueError, "cannot be safely loaded"):
            load_validated_checkpoint(self.path)


if __name__ == "__main__":
    unittest.main()
