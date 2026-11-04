import hashlib
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
                "source_url": f"https://example.org/source/{index}",
                "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "rights_basis": "original-authorship",
                "source_date": "2026-01-15",
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
                split_manifest = manifest["splits"][split]
                self.assertEqual(set(split_manifest["source_group_ids"]), groups[split])
                self.assertEqual(split_manifest["records"], len(groups[split]))
                self.assertEqual(split_manifest["source_groups"], len(groups[split]))
                self.assertEqual(split_manifest["utf8_bytes"], len(text.encode("utf-8")))
                self.assertEqual(split_manifest["record_ids"], sorted(
                    f"record-{i}" for i in range(6) if f"Record {i}" in text
                ))
                self.assertEqual(split_manifest["language_counts"], {"en": len(groups[split])})
                self.assertEqual(split_manifest["license_counts"], {"CC0-1.0": len(groups[split])})
                self.assertEqual(
                    split_manifest["sha256"],
                    hashlib.sha256(text.encode("utf-8")).hexdigest()
                )
            self.assertFalse(groups["train"] & groups["validation"])
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["validation"] & groups["test"])

    def test_exports_hash_verified_per_language_files_without_split_leakage(self):
        rows = self.make_records(count=6)
        multilingual_rows = []
        for row in rows:
            multilingual_rows.append(row)
            translated = dict(row)
            translated["id"] = row["id"] + "-sw"
            translated["language"] = "sw"
            translated["text"] = (
                f"Maandishi ya mfano {row['id']} yanaeleza maarifa kwa Kiswahili. "
                "Kila rekodi inatoka kwa kundi huru lililokaguliwa."
            )
            multilingual_rows.append(translated)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_prepare(root, multilingual_rows)
            self.assertEqual(result.returncode, 0, result.stderr)
            prepared = root / "prepared"
            manifest = json.loads((prepared / "manifest.json").read_text())
            all_groups = {}
            for split in ("train", "validation", "test"):
                split_meta = manifest["splits"][split]
                language_files = split_meta["language_files"]
                self.assertEqual(set(language_files), {"en", "sw"})
                all_groups[split] = set(split_meta["source_group_ids"])
                for language, metadata in language_files.items():
                    language_path = prepared / metadata["path"]
                    content = language_path.read_text(encoding="utf-8")
                    self.assertTrue(content.strip())
                    self.assertEqual(metadata["records"], len(metadata["record_ids"]))
                    self.assertEqual(
                        metadata["sha256"],
                        hashlib.sha256(language_path.read_bytes()).hexdigest()
                    )
                    self.assertEqual(metadata["utf8_bytes"], len(language_path.read_bytes()))
                    self.assertEqual(
                        set(metadata["source_group_ids"]),
                        all_groups[split]
                    )
                    self.assertIn("Record", content) if language == "en" else self.assertIn(
                        "Maandishi", content
                    )
            self.assertFalse(all_groups["train"] & all_groups["validation"])
            self.assertFalse(all_groups["train"] & all_groups["test"])
            self.assertFalse(all_groups["validation"] & all_groups["test"])

    def test_manifest_counts_removed_exact_duplicate_text(self):
        rows = self.make_records()
        rows[1]["text"] = rows[0]["text"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_prepare(root, rows)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((root / "prepared" / "manifest.json").read_text())
            self.assertEqual(manifest["records"], 5)
            self.assertEqual(manifest["duplicate_text_records_removed"], 1)

    def test_rejects_duplicate_id_even_after_duplicate_text_was_skipped(self):
        rows = self.make_records()
        rows[1]["text"] = rows[0]["text"]
        rows[2]["id"] = rows[1]["id"]
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_prepare(Path(directory), rows)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Duplicate record id", result.stderr)

    def test_manifest_reports_license_language_and_date_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_prepare(root, self.make_records())
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((root / "prepared" / "manifest.json").read_text())
            self.assertEqual(manifest["language_counts"], {"en": 6})
            self.assertEqual(manifest["license_counts"], {"CC0-1.0": 6})
            self.assertEqual(
                manifest["source_date_range"],
                {"earliest": "2026-01-15", "latest": "2026-01-15"}
            )

    def test_rejects_missing_or_invalid_source_date(self):
        for value in (None, "15-01-2026", "2026-02-30"):
            rows = self.make_records()
            if value is None:
                del rows[0]["source_date"]
            else:
                rows[0]["source_date"] = value
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                result = self.run_prepare(Path(directory), rows)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(
                    "Missing required metadata" in result.stderr
                    or "Invalid source_date" in result.stderr
                )

    def test_manifest_reports_rights_basis_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_prepare(root, self.make_records())
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((root / "prepared" / "manifest.json").read_text())
            self.assertEqual(manifest["rights_basis_counts"], {"original-authorship": 6})

    def test_rejects_missing_or_invalid_source_and_license_provenance(self):
        cases = (
            ("source_url", None, "Missing required metadata"),
            ("source_url", "http://example.org/source", "absolute HTTPS URL"),
            ("license_url", "http://example.org/license", "absolute HTTPS URL"),
            ("rights_basis", "unverified", "Invalid rights_basis"),
        )
        for field, value, expected in cases:
            rows = self.make_records()
            if value is None:
                del rows[0][field]
            else:
                rows[0][field] = value
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                result = self.run_prepare(Path(directory), rows)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)

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
