import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("prepare_corpus.py")


class CorpusPreparationTests(unittest.TestCase):
    def make_records(self, count=6, rights=True):
        rows = []
        for index in range(count):
            rows.append({
                "id": f"record-{index}",
                "source": f"DREAM authored source {index}",
                "source_group": f"independent-group-{index}",
                "license": "CC0-1.0",
                "rights_confirmed": rights,
                "privacy_reviewed": True,
                "quality_reviewed": True,
                "language": "en",
                "text": (
                    f"Record {index} is an independently authored training example. "
                    "It contains enough meaningful text to pass the corpus preparation gate."
                )
            })
        return rows

    def run_prepare(self, root, rows):
        source = root / "records.jsonl"
        source.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8"
        )
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--input", str(source),
             "--output", str(root / "prepared")],
            capture_output=True, text=True
        )

    def test_creates_disjoint_source_splits_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_prepare(root, self.make_records())
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((root / "prepared" / "manifest.json").read_text())
            self.assertEqual(manifest["records"], 6)
            groups = {}
            for split in ("train", "validation", "test"):
                text = (root / "prepared" / f"{split}.txt").read_text()
                groups[split] = {
                    f"independent-group-{i}" for i in range(6)
                    if f"Record {i}" in text
                }
                self.assertTrue(text.strip())
            self.assertFalse(groups["train"] & groups["validation"])
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["validation"] & groups["test"])

    def test_rejects_duplicate_id_even_after_duplicate_text_was_skipped(self):
        rows = self.make_records()
        rows[1]["text"] = rows[0]["text"]
        rows[2]["id"] = rows[1]["id"]
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_prepare(Path(directory), rows)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Duplicate record id", result.stderr)

    def test_rejects_unconfirmed_rights(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_prepare(Path(directory), self.make_records(rights=False))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Rights not confirmed", result.stderr)

    def test_rejects_unreviewed_privacy_and_quality(self):
        for field, message in (
            ("privacy_reviewed", "Privacy review not confirmed"),
            ("quality_reviewed", "Quality review not confirmed")
        ):
            rows = self.make_records()
            rows[0][field] = False
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                result = self.run_prepare(Path(directory), rows)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_rejects_too_few_independent_sources(self):
        rows = self.make_records(count=3)
        for row in rows:
            row["source_group"] = "same-source"
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_prepare(Path(directory), rows)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("three independent source groups", result.stderr)


if __name__ == "__main__":
    unittest.main()
