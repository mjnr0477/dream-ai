"""Tests for the conservative, unapproved OASST1 candidate importer."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "data" / "import_oasst1_candidates.py"
SPEC = importlib.util.spec_from_file_location("oasst1_importer", MODULE_PATH)
oasst1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oasst1)


def good_row():
    return {
        "message_id": "message-1",
        "message_tree_id": "tree-1",
        "user_id": "must-not-be-exported",
        "created_date": "2023-04-12T12:30:00Z",
        "text": "This is a sufficiently long educational conversation message.",
        "lang": "en",
        "deleted": False,
        "synthetic": False,
        "review_result": True,
        "tree_state": "ready_for_export",
        "labels": {
            "pii": {"value": 0.0},
            "spam": {"value": 0.0},
            "not_appropriate": {"value": 0.0},
            "hate_speech": {"value": 0.0},
            "sexual_content": {"value": 0.0},
            "violence": {"value": 0.0},
        },
        "detoxify": {
            "toxicity": 0.0, "severe_toxicity": 0.0, "obscene": 0.0,
            "identity_attack": 0.0, "threat": 0.0, "sexual_explicit": 0.0,
        },
    }


class OASST1CandidateImporterTests(unittest.TestCase):
    def test_candidate_is_never_preapproved(self):
        row = good_row()
        candidate = oasst1.convert_row(row)
        self.assertEqual(candidate["license"], "Apache-2.0")
        self.assertEqual(candidate["source_date"], "2023-04-12")
        self.assertEqual(candidate["language"], "en")
        self.assertFalse(candidate["rights_confirmed"])
        self.assertFalse(candidate["privacy_reviewed"])
        self.assertFalse(candidate["quality_reviewed"])
        self.assertFalse(candidate["approval_granted"])
        self.assertNotIn("user_id", candidate)
        self.assertIn("tree-1", candidate["source_group"])

    def test_requires_positive_human_review_and_non_synthetic_content(self):
        row = good_row()
        row["review_result"] = False
        self.assertEqual(oasst1.candidate_rejection(row), "not_positive_human_review")
        row = good_row()
        row["synthetic"] = True
        self.assertEqual(oasst1.candidate_rejection(row), "synthetic_or_unknown")

    def test_rejects_missing_or_positive_pii_label(self):
        row = good_row()
        del row["labels"]["pii"]
        self.assertEqual(
            oasst1.candidate_rejection(row),
            "pii_label_missing_or_positive",
        )
        row = good_row()
        row["labels"]["pii"]["value"] = 0.01
        self.assertEqual(
            oasst1.candidate_rejection(row),
            "pii_label_missing_or_positive",
        )

    def test_rejects_non_finite_safety_scores(self):
        row = good_row()
        row["labels"]["pii"]["value"] = float("nan")
        self.assertEqual(
            oasst1.candidate_rejection(row),
            "pii_label_missing_or_positive",
        )

    def test_rejects_flagged_content(self):
        row = good_row()
        row["labels"]["spam"]["value"] = 0.01
        self.assertEqual(oasst1.candidate_rejection(row), "flagged_spam")
        row = good_row()
        row["detoxify"]["toxicity"] = 0.2
        self.assertEqual(oasst1.candidate_rejection(row), "detoxify_toxicity")

    def test_requires_ready_tree_and_valid_metadata(self):
        row = good_row()
        row["tree_state"] = "aborted_low_grade"
        self.assertEqual(oasst1.candidate_rejection(row), "tree_not_ready_for_export")
        row = good_row()
        row["created_date"] = "not-a-date"
        self.assertEqual(oasst1.candidate_rejection(row), "invalid_source_date")

    def test_import_writes_candidates_and_audit_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.jsonl"
            output = root / "candidates.jsonl"
            report = root / "report.json"
            source.write_text(
                json.dumps(good_row()) + "\n"
                + json.dumps({**good_row(), "message_id": "bad", "synthetic": True}) + "\n",
                encoding="utf-8",
            )
            import subprocess
            import sys
            subprocess.run([
                sys.executable, str(MODULE_PATH),
                "--input", str(source),
                "--output", str(output),
                "--report", str(report),
            ], check=True, capture_output=True, text=True)
            candidate = json.loads(output.read_text(encoding="utf-8"))
            audit = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(audit["candidate_records"], 1)
            self.assertFalse(audit["training_approved"])
            self.assertTrue(audit["human_review_required"])
            self.assertFalse(candidate["approval_granted"])


if __name__ == "__main__":
    unittest.main()
