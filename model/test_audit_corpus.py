import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("audit_corpus.py")


class CorpusAuditTests(unittest.TestCase):
    def record(self, record_id="one", text="This is a sufficiently long sample record for corpus auditing."):
        return {
            "id": record_id,
            "source": "Human-authored test source",
            "source_group": f"group-{record_id}",
            "source_url": "https://example.org/source",
            "license": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "rights_basis": "open-license",
            "rights_confirmed": True,
            "privacy_reviewed": True,
            "quality_reviewed": True,
            "language": "en",
            "text": text
        }

    def run_audit(self, root, rows, extra=()):
        source = root / "candidate.jsonl"
        output = root / "audit.json"
        source.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8"
        )
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--input", str(source), "--out", str(output), *extra],
            capture_output=True, text=True
        )
        return result, json.loads(output.read_text(encoding="utf-8"))

    def test_clean_corpus_is_still_not_auto_approved(self):
        with tempfile.TemporaryDirectory() as directory:
            result, report = self.run_audit(Path(directory), [self.record()])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(report["status"], "review_required")
            self.assertFalse(report["approval_granted"])
            self.assertEqual(report["records"], 1)
            self.assertEqual(report["error_count"], 0)

    def test_flags_possible_personal_data_without_copying_it_to_report(self):
        private_text = "Contact learner@example.org or call 0712345678 for help with this lesson."
        with tempfile.TemporaryDirectory() as directory:
            result, report = self.run_audit(Path(directory), [self.record(text=private_text)])
            self.assertEqual(result.returncode, 0, result.stderr)
            codes = {item["code"] for item in report["findings"]}
            self.assertIn("possible_email", codes)
            self.assertIn("possible_kenyan_phone", codes)
            serialized = json.dumps(report)
            self.assertNotIn("learner@example.org", serialized)
            self.assertNotIn("0712345678", serialized)

    def test_duplicate_content_and_incomplete_reviews_are_reported(self):
        first = self.record("first")
        second = self.record("second")
        second["text"] = "  THIS IS A SUFFICIENTLY LONG SAMPLE RECORD FOR CORPUS AUDITING. "
        second["privacy_reviewed"] = False
        with tempfile.TemporaryDirectory() as directory:
            result, report = self.run_audit(Path(directory), [first, second])
            self.assertEqual(result.returncode, 0, result.stderr)
            codes = {item["code"] for item in report["findings"]}
            self.assertIn("duplicate_normalized_content", codes)
            self.assertIn("human_review_incomplete", codes)
            self.assertEqual(report["unreviewed_records"], 1)

    def test_invalid_metadata_returns_nonzero_and_writes_report(self):
        row = self.record()
        row["license_url"] = "http://example.org/license"
        with tempfile.TemporaryDirectory() as directory:
            result, report = self.run_audit(Path(directory), [row])
            self.assertEqual(result.returncode, 1)
            self.assertGreater(report["error_count"], 0)
            self.assertFalse(report["approval_granted"])

    def test_oversized_record_is_rejected(self):
        row = self.record(text="x" * 200)
        with tempfile.TemporaryDirectory() as directory:
            result, report = self.run_audit(
                Path(directory), [row], ("--max-record-bytes", "50")
            )
            self.assertEqual(result.returncode, 1)
            self.assertEqual(report["errors"][0]["code"], "record_too_large")


if __name__ == "__main__":
    unittest.main()
