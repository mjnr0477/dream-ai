"""Tests for FineWeb2 candidate conversion and approval safeguards."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("import_fineweb2_candidates.py")


class FineWeb2CandidateImporterTests(unittest.TestCase):
    def run_import(self, root, *extra):
        source = root / "fineweb2.jsonl"
        output = root / "candidates.jsonl"
        summary = root / "summary.json"
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--input", str(source),
             "--output", str(output), "--summary", str(summary), *extra],
            capture_output=True, text=True, timeout=30
        )
        return result, output, summary

    def test_candidates_are_never_auto_approved_and_keep_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {"id": "doc-1", "text": "A sufficiently long sample document for DREAM data review. " * 2,
                 "url": "https://example.org/page/1", "date": "2024-02-03T10:00:00Z",
                 "language": "swa", "language_script": "Latn"},
                {"id": "doc-2", "text": "Another sufficiently long document for an independent source review. " * 2,
                 "url": "https://news.example.net/story/2", "date": "2023-11-12",
                 "language": "eng", "language_script": "Latn"},
            ]
            (root / "fineweb2.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            result, output, summary = self.run_import(root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            candidates = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(candidates), 2)
            self.assertEqual(candidates[0]["source_group"], "example.org")
            self.assertEqual(candidates[0]["source_date"], "2024-02-03")
            self.assertEqual(candidates[0]["language"], "swa")
            for item in candidates:
                self.assertFalse(item["rights_confirmed"])
                self.assertFalse(item["privacy_reviewed"])
                self.assertFalse(item["quality_reviewed"])
                self.assertFalse(item["approval_granted"])
                self.assertIn("unverified", item["rights_basis"])
            report = json.loads(summary.read_text())
            self.assertEqual(report["review_status"], "UNAPPROVED_CANDIDATES_ONLY")
            self.assertTrue(report["all_review_flags_false"])

    def test_language_filter_and_per_language_cap_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(4):
                rows.append({
                    "id": f"swa-{index}",
                    "text": f"Swahili candidate document number {index} with enough text for review. " * 2,
                    "url": f"https://swahili.example.org/{index}",
                    "date": "2024-01-01", "language": "swa"
                })
            rows.append({
                "id": "amh-1", "text": "Amharic candidate document with enough text for review. " * 2,
                "url": "https://amh.example.org/1", "date": "2024-01-01", "language": "amh"
            })
            (root / "fineweb2.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            first, output, _ = self.run_import(
                root, "--languages", "swa", "--max-per-language", "2"
            )
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            first_content = output.read_text()
            second, _, _ = self.run_import(
                root, "--languages", "swa", "--max-per-language", "2"
            )
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertEqual(output.read_text(), first_content)
            self.assertEqual(len(first_content.splitlines()), 2)
            self.assertTrue(all(json.loads(line)["language"] == "swa"
                                for line in first_content.splitlines()))

    def test_rejects_invalid_or_missing_source_url(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {
                "id": "bad", "text": "A sufficiently long document that should fail due to URL. " * 2,
                "url": "not-a-url", "date": "2024-01-01", "language": "eng"
            }
            (root / "fineweb2.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            result, _, _ = self.run_import(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("absolute HTTP(S) URL", result.stderr)

    def test_rejects_invalid_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fineweb2.jsonl").write_text("{invalid}\n", encoding="utf-8")
            result, _, _ = self.run_import(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Invalid FineWeb2 record", result.stderr)


if __name__ == "__main__":
    unittest.main()
