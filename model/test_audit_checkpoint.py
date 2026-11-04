"""Tests for read-only DREAM checkpoint integrity auditing."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from audit_checkpoint import audit_checkpoint
from dream_transformer import DreamConfig, DreamTransformer

SCRIPT = Path(__file__).with_name("audit_checkpoint.py")


class CheckpointAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = DreamConfig(
            block_size=32, d_model=16, n_heads=4, n_layers=1, dropout=0.0
        )
        self.model = DreamTransformer(self.config)
        self.path = self.root / "checkpoint.pt"
        self.valid_checkpoint = {
            "config": self.config.__dict__,
            "model_state": self.model.state_dict(),
            "tokenizer": "utf8-byte-v1",
            "training": {"completed_steps": 12},
        }
        torch.save(self.valid_checkpoint, self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_accepts_valid_checkpoint_and_reports_hash_and_dimensions(self):
        report = audit_checkpoint(self.path)
        self.assertEqual(report["schema"], "dream-checkpoint-audit-v1")
        self.assertEqual(report["completed_steps"], 12)
        self.assertEqual(report["parameter_count"], sum(
            parameter.numel() for parameter in self.model.parameters()
        ))
        self.assertTrue(report["all_floating_tensors_finite"])
        self.assertTrue(report["strict_state_dict_load"])
        self.assertEqual(len(report["checkpoint_sha256"]), 64)

    def test_rejects_missing_or_invalid_training_steps(self):
        for steps in (0, -1, True, 1.5, "12"):
            with self.subTest(steps=steps):
                checkpoint = dict(self.valid_checkpoint)
                checkpoint["training"] = {"completed_steps": steps}
                torch.save(checkpoint, self.path)
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    audit_checkpoint(self.path)

    def test_rejects_non_finite_model_weights(self):
        checkpoint = dict(self.valid_checkpoint)
        state = dict(checkpoint["model_state"])
        key = next(name for name, value in state.items() if value.is_floating_point())
        state[key] = state[key].clone()
        state[key].view(-1)[0] = float("nan")
        checkpoint["model_state"] = state
        torch.save(checkpoint, self.path)
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            audit_checkpoint(self.path)

    def test_rejects_wrong_tensor_shape_and_missing_keys(self):
        checkpoint = dict(self.valid_checkpoint)
        state = dict(checkpoint["model_state"])
        key = next(iter(state))
        state[key] = state[key].reshape(-1)[:1]
        checkpoint["model_state"] = state
        torch.save(checkpoint, self.path)
        with self.assertRaisesRegex(ValueError, "incorrect shape"):
            audit_checkpoint(self.path)

        state = dict(self.valid_checkpoint["model_state"])
        state.pop(next(iter(state)))
        checkpoint["model_state"] = state
        torch.save(checkpoint, self.path)
        with self.assertRaisesRegex(ValueError, "keys do not match"):
            audit_checkpoint(self.path)

    def test_rejects_wrong_tensor_dtype_and_non_string_state_keys(self):
        checkpoint = dict(self.valid_checkpoint)
        state = dict(checkpoint["model_state"])
        key = next(iter(state))
        state[key] = state[key].to(torch.float64)
        checkpoint["model_state"] = state
        torch.save(checkpoint, self.path)
        with self.assertRaisesRegex(ValueError, "incorrect dtype"):
            audit_checkpoint(self.path)

        state = dict(self.valid_checkpoint["model_state"])
        state[123] = torch.tensor(1)
        checkpoint["model_state"] = state
        torch.save(checkpoint, self.path)
        with self.assertRaisesRegex(ValueError, "keys must all be strings"):
            audit_checkpoint(self.path)

    def test_rejects_unsupported_tokenizer_and_extra_config_fields(self):
        checkpoint = dict(self.valid_checkpoint)
        checkpoint["tokenizer"] = "unknown"
        torch.save(checkpoint, self.path)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            audit_checkpoint(self.path)

        checkpoint = dict(self.valid_checkpoint)
        checkpoint["config"] = dict(self.config.__dict__, surprise=True)
        torch.save(checkpoint, self.path)
        with self.assertRaisesRegex(ValueError, "Invalid DREAM model config"):
            audit_checkpoint(self.path)

    def test_cli_writes_json_report(self):
        output = self.root / "reports" / "audit.json"
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--checkpoint", str(self.path),
             "--out", str(output)],
            capture_output=True, text=True, timeout=30
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(report["strict_state_dict_load"])
        self.assertIn("checkpoint_sha256", report)


if __name__ == "__main__":
    unittest.main()
