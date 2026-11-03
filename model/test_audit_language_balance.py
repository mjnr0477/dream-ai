"""Tests for privacy-conscious DREAM language balance reporting."""
import json
import tempfile
import unittest
from pathlib import Path

from audit_language_balance import audit, script_of


class LanguageBalanceTests(unittest.TestCase):
    def write_rows(self, root, rows):
        path = root / "corpus.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8"
        )
        return path

    def row(self, record_id, language, text, group):
        return {
            "id": record_id, "language": language, "text": text,
            "source_group": group
        }

    def test_reports_per_language_counts_scripts_and_warnings_without_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_rows(root, [
                self.row("en-1", "en", "Hello world!", "en-source-1"),
                self.row("en-2", "en", "Another sentence.", "en-source-2"),
                self.row("sw-1", "sw", "Habari za asubuhi.", "sw-source-1"),
                self.row("ar-1", "ar", "مرحبا بالعالم", "ar-source-1")
            ])
            report = audit(path, min_records=2, max_share=0.6)
            self.assertEqual(report["language_count"], 3)
            self.assertEqual(report["languages"]["en"]["records"], 2)
            self.assertGreater(report["languages"]["ar"]["scripts"]["Arabic"], 0)
            self.assertGreater(report["languages"]["sw"]["scripts"]["Latin"], 0)
            self.assertFalse(report["approval_granted"])
            self.assertEqual(report["status"], "review_required")
            self.assertNotIn("Hello world", json.dumps(report))
            self.assertTrue(any(x["code"] == "FEW_SOURCE_GROUPS" for x in report["warnings"]))

    def test_flags_language_dominance_and_low_resource_language(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_rows(root, [
                self.row("a", "en", "English sentence one.", "a"),
                self.row("b", "en", "English sentence two.", "b"),
                self.row("c", "en", "English sentence three.", "c"),
                self.row("d", "sw", "Sentensi ya Kiswahili.", "d")
            ])
            report = audit(path, min_records=2, max_share=0.7)
            codes = {(item["code"], item["language"]) for item in report["warnings"]}
            self.assertIn(("LANGUAGE_DOMINANCE", "en"), codes)
            self.assertIn(("LOW_RECORD_COUNT", "sw"), codes)

    def test_recognizes_major_writing_systems(self):
        self.assertEqual(script_of("क"), "Devanagari")
        self.assertEqual(script_of("中"), "Han")
        self.assertEqual(script_of("あ"), "Hiragana")
        self.assertEqual(script_of("한"), "Hangul")
        self.assertEqual(script_of("ሀ"), "Ethiopic")

    def test_rejects_invalid_language_tag_and_missing_source_group(self):
        for row in (
            self.row("bad", "not a language", "Some sufficiently long text.", "g"),
            self.row("bad-group", "en", "Some sufficiently long text.", "")
        ):
            with self.subTest(row=row), tempfile.TemporaryDirectory() as directory:
                path = self.write_rows(Path(directory), [row])
                with self.assertRaises(ValueError):
                    audit(path, min_records=1)

    def test_rejects_empty_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_rows(Path(directory), [])
            with self.assertRaisesRegex(ValueError, "no records"):
                audit(path)


if __name__ == "__main__":
    unittest.main()
