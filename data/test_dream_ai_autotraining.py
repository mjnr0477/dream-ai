from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "scripts"))

from dream_ai_optin_pipeline import export_optin_rows, prepare_sft_splits
import prepare_dream_ai_autotraining as auto_training
from prepare_dream_ai_autotraining import eligible_rows
from audit_corpus import audit
from verify_dream_ai_sft_audits import verify_audit_reports
from review_candidates import candidate_problem

REVIEW_FLAGS = (
    "rights_confirmed",
    "privacy_reviewed",
    "quality_reviewed",
    "approval_granted",
)
STAMP = "2026-11-07T12:00:00+00:00"


def fake_rows():
    rows = []
    for index in range(32):
        group_index = index // 4
        group_hash = hashlib.sha256(f"source-group-{group_index}".encode()).hexdigest()
        prompt = f"Explain a distinct DREAM AI learning scenario number {index} with a clear step-by-step response."
        response = (
            f"For scenario {index}, first identify the user's goal, then verify the facts, "
            "explain the method, and state any limits or assumptions clearly."
        )
        rows.append({
            "id": str(uuid.uuid4()),
            "mode": "learning",
            "prompt": prompt,
            "response": response,
            "input_text": prompt,
            "target_text": response,
            "consented": True,
            "consent_version": "dream-learning-opt-in-v3",
            "consented_at": STAMP,
            "created_at": STAMP,
            "reviewed_at": STAMP,
            "content_sha256": hashlib.sha256(f"{index}-{prompt}-{response}".encode()).hexdigest(),
            "status": "approved",
            "review_status": "approved",
            "metadata": {
                "source_group_sha256": group_hash,
                "language": "en",
                "human_reviewed_at": STAMP,
                "review": {flag: True for flag in REVIEW_FLAGS},
            },
        })
    return rows


class DreamAutoTrainingTests(unittest.TestCase):
    def test_only_complete_reviewed_v3_rows_are_eligible(self):
        rows = fake_rows()
        valid, skipped = eligible_rows(rows)
        self.assertEqual(len(valid), 32)
        self.assertEqual(len({r["metadata"]["source_group_sha256"] for r in valid}), 8)
        self.assertFalse(any(skipped.values()))

        rows[0]["metadata"]["review"]["quality_reviewed"] = False
        rows[1]["consent_version"] = "dream-learning-opt-in-v2"
        rows[2]["metadata"].pop("source_group_sha256")
        valid, skipped = eligible_rows(rows)
        self.assertEqual(len(valid), 29)
        self.assertEqual(skipped["invalid_review_flags"], 1)
        self.assertEqual(skipped["missing_source_group"], 1)

    def test_export_split_and_audit_produce_a_valid_training_gate(self):
        rows = fake_rows()
        eligible, skipped = eligible_rows(rows)
        self.assertFalse(any(skipped.values()))
        candidates, report = export_optin_rows(eligible)
        self.assertEqual(report["exported"], 32)
        self.assertTrue(all(r["content_sha256"] for r in candidates))

        by_hash = {r["content_sha256"]: r for r in eligible}
        for candidate in candidates:
            source = by_hash[candidate["content_sha256"]]
            for flag in REVIEW_FLAGS:
                candidate[flag] = source["metadata"]["review"][flag]
            candidate["human_reviewed_at"] = source["metadata"]["human_reviewed_at"]
            self.assertIsNone(candidate_problem(candidate))

        train, validation, manifest = prepare_sft_splits(
            candidates,
            min_source_groups=8,
            min_validation_records=4,
        )
        self.assertGreaterEqual(len(train), 20)
        self.assertGreaterEqual(len(validation), 4)

        with tempfile.TemporaryDirectory() as temp:
            train_path = Path(temp) / "train.jsonl"
            validation_path = Path(temp) / "validation.jsonl"
            for target, records in ((train_path, train), (validation_path, validation)):
                target.write_text(
                    "".join(
                        __import__("json").dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                        for record in records
                    ),
                    encoding="utf-8",
                )
            train_audit = audit(train_path)
            validation_audit = audit(validation_path)
            gate = verify_audit_reports(manifest, train_audit, validation_audit)

        self.assertTrue(gate["passed"])
        self.assertEqual(gate["splits"]["train"]["error_count"], 0)
        self.assertEqual(gate["splits"]["validation"]["error_count"], 0)

    def test_next_cycle_uses_latest_validated_candidate_as_parent(self):
        candidate_id = "20261107T120000Z-abcdef123456"
        valid_metrics = {
            "audit_gate_passed": True,
            "baseline_validation_loss": 1.0,
            "candidate_validation_loss": 0.99,
        }
        row = {
            "model_name": candidate_id,
            "artifact_uri": (
                "https://huggingface.co/msjnr0477/dream-ai-adapter/tree/main/"
                "candidates/" + candidate_id
            ),
            "base_model": auto_training.BASE_MODEL,
            "metrics": valid_metrics,
        }
        with patch.object(auto_training, "fetch_all", return_value=[row]):
            self.assertEqual(
                auto_training.latest_candidate_parent("https://example.supabase.co", "key"),
                "candidates/" + candidate_id,
            )

    def test_rejects_candidate_with_validation_regression_as_training_parent(self):
        candidate_id = "20261107T120000Z-abcdef123456"
        row = {
            "model_name": candidate_id,
            "artifact_uri": (
                "https://huggingface.co/msjnr0477/dream-ai-adapter/tree/main/"
                "candidates/" + candidate_id
            ),
            "base_model": auto_training.BASE_MODEL,
            "metrics": {
                "audit_gate_passed": True,
                "baseline_validation_loss": 1.0,
                "candidate_validation_loss": 1.2,
            },
        }
        with unittest.mock.patch.object(auto_training, "fetch_all", return_value=[row]):
            self.assertEqual(
                auto_training.latest_candidate_parent("https://example.supabase.co", "key"),
                "",
            )

    def test_pending_and_unconsented_rows_are_not_exported_for_training(self):
        rows = fake_rows()
        rows[0]["status"] = "pending"
        rows[1]["consented"] = False
        eligible, _ = eligible_rows(rows)
        self.assertEqual(len(eligible), 30)


if __name__ == "__main__":
    unittest.main()
