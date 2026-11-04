"""Tests for DREAM's technical checkpoint/evaluation release gate."""
import json
import tempfile
import unittest
from pathlib import Path

from validate_release import validate


class TechnicalReleaseGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "model.pt"
        self.checkpoint.write_bytes(b"fixture checkpoint bytes")
        import hashlib
        self.checkpoint_hash = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.audit = self.root / "audit.json"
        self.evaluation = self.root / "evaluation.json"
        self.audit.write_text(json.dumps({
            "schema": "dream-checkpoint-audit-v1",
            "checkpoint_sha256": self.checkpoint_hash,
            "strict_state_dict_load": True,
            "all_floating_tensors_finite": True,
            "completed_steps": 120,
            "parameter_count": 5000
        }), encoding="utf-8")
        self.evaluation.write_text(json.dumps({
            "schema": "dream-evaluation-v1",
            "checkpoint_sha256": self.checkpoint_hash,
            "evaluation_corpus_sha256": "a" * 64,
            "parameters": 5000,
            "mean_cross_entropy_nats_per_token": 2.0,
            "mean_cross_entropy_nats_per_byte": 1.0,
            "bits_per_byte": 1.44,
            "perplexity_per_byte": 2.7
        }), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_accepts_matching_technical_artifacts_but_never_approves_production(self):
        report = validate(self.checkpoint, self.audit, self.evaluation)
        self.assertTrue(report["technical_preflight_passed"])
        self.assertFalse(report["production_release_approved"])
        self.assertEqual(report["completed_training_steps"], 120)

    def test_rejects_audit_from_another_checkpoint(self):
        audit = json.loads(self.audit.read_text(encoding="utf-8"))
        audit["checkpoint_sha256"] = "b" * 64
        self.audit.write_text(json.dumps(audit), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "audit hash"):
            validate(self.checkpoint, self.audit, self.evaluation)

    def test_rejects_evaluation_from_another_checkpoint(self):
        report = json.loads(self.evaluation.read_text(encoding="utf-8"))
        report["checkpoint_sha256"] = "c" * 64
        self.evaluation.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Evaluation checkpoint hash"):
            validate(self.checkpoint, self.audit, self.evaluation)

    def test_rejects_non_finite_metric(self):
        report = json.loads(self.evaluation.read_text(encoding="utf-8"))
        report["bits_per_byte"] = float("nan")
        self.evaluation.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate(self.checkpoint, self.audit, self.evaluation)


if __name__ == "__main__":
    unittest.main()
