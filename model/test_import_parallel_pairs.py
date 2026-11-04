"""Safety and determinism tests for the parallel-pair candidate importer."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("import_parallel_pairs.py")


class ParallelPairImporterTests(unittest.TestCase):
    def make_input(self, root, count=20, with_groups=True):
        path = root / "parallel.jsonl"
        rows = []
        for index in range(count):
            row = {
                "translation": {
                    "dav": f"Source sentence {index} has enough distinct words.",
                    "swa": f"Sentensi ya Kiswahili {index} ina maneno tofauti."
                }
            }
            if with_groups:
                row["source_group"] = f"book-chapter-{index // 2}"
            rows.append(row)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return path

    def run_import(self, source, output):
        return subprocess.run([
            sys.executable, str(SCRIPT),
            "--input", str(source),
            "--source-lang", "dav",
            "--target-lang", "swa",
            "--source-name", "Local parallel export",
            "--source-url", "https://example.org/dataset",
            "--license", "CC-BY-4.0",
            "--license-url", "https://creativecommons.org/licenses/by/4.0/",
            "--source-date", "2025-01-15",
            "--output-dir", str(output),
            "--validation-percent", "20",
        ], capture_output=True, text=True, timeout=30)

    def test_creates_deterministic_unapproved_source_isolated_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_input(root)
            output = root / "out"
            first = self.run_import(source, output)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            summary = json.loads((output / "import-summary.json").read_text())
            train = [json.loads(x) for x in (output / "sft-train-candidates.jsonl").read_text().splitlines()]
            validation = [json.loads(x) for x in (output / "sft-validation-candidates.jsonl").read_text().splitlines()]
            self.assertTrue(train and validation)
            self.assertFalse({x["source_group"] for x in train} & {x["source_group"] for x in validation})
            self.assertEqual(summary["review_status"], "human_review_required")
            self.assertFalse(summary["approval_granted"])
            for row in train + validation:
                self.assertEqual(row["rights_basis"], "unverified")
                self.assertFalse(row["rights_confirmed"])
                self.assertFalse(row["privacy_reviewed"])
                self.assertFalse(row["quality_reviewed"])
                self.assertTrue(row["id"].startswith("parallel-"))
            before = (output / "sft-train-candidates.jsonl").read_bytes()
            second = self.run_import(source, output)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(before, (output / "sft-train-candidates.jsonl").read_bytes())

    def test_rejects_missing_grouping_instead_of_fake_independence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_input(root, with_groups=False)
            result = self.run_import(source, root / "out")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("independent splits require grouping metadata", result.stderr)

    def test_hashes_external_group_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_input(root)
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            rows[0]["source_group"] = "person@example.com"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            output = root / "out"
            result = self.run_import(source, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            combined = "".join(p.read_text() for p in output.iterdir() if p.is_file())
            self.assertNotIn("person@example.com", combined)

    def test_rejects_conflicting_translation_and_duplicate_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_input(root)
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            rows[1]["translation"]["dav"] = rows[0]["translation"]["dav"]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = self.run_import(source, root / "conflict")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Conflicting translations", result.stderr)

            rows = [json.loads(line) for line in source.read_text().splitlines()]
            rows[1]["translation"] = dict(rows[0]["translation"])
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = self.run_import(source, root / "duplicate")
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((root / "duplicate" / "import-summary.json").read_text())
            self.assertEqual(summary["duplicate_pairs_removed"], 1)

    def test_rejects_invalid_url_and_language_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_input(root)
            result = self.run_import(source, root / "bad-url")
            self.assertEqual(result.returncode, 0, result.stderr)
            # A URL containing embedded credentials must be rejected.
            command = [
                sys.executable, str(SCRIPT), "--input", str(source),
                "--source-lang", "dav", "--target-lang", "swa",
                "--source-name", "source", "--source-url", "https://user:pass@example.org/data",
                "--license", "unknown", "--license-url", "https://example.org/license",
                "--source-date", "2025-01-15", "--output-dir", str(root / "invalid")
            ]
            bad = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("absolute HTTPS URLs", bad.stderr)

    def test_story_id_groups_and_sft_loader_after_human_review(self):
        from dream_transformer import DreamByteTokenizer
        from sft import load_examples

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_input(root, count=8)
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            for row in rows:
                row.pop("source_group", None)
                row["story_id"] = "private-story-" + row["translation"]["dav"].split()[-4]
                row["id"] = "private-id@example.test"
            # Distinct story IDs and record IDs must not be copied into generated files.
            for index, row in enumerate(rows):
                row["story_id"] = f"private-story-{index // 2}"
                row["id"] = f"private-{index}@example.test"
            source.write_text("".join(json.dumps(row) + "\\n" for row in rows), encoding="utf-8")
            output = root / "out"
            result = self.run_import(source, output)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            combined = "".join(path.read_text() for path in output.iterdir())
            self.assertNotIn("private-story-", combined)
            self.assertNotIn("private-", combined)
            train_path = output / "sft-train-candidates.jsonl"
            train = [json.loads(line) for line in train_path.read_text().splitlines()]
            for row in train:
                row["rights_confirmed"] = True
                row["privacy_reviewed"] = True
                row["quality_reviewed"] = True
            reviewed = root / "reviewed.jsonl"
            reviewed.write_text("".join(json.dumps(row) + "\\n" for row in train), encoding="utf-8")
            examples = load_examples(reviewed, DreamByteTokenizer(), block_size=256)
            self.assertEqual(len(examples), len(train))

    def test_rejects_same_language_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_input(root)
            command = [
                sys.executable, str(SCRIPT), "--input", str(source),
                "--source-lang", "dav", "--target-lang", "dav",
                "--source-name", "Local parallel export",
                "--source-url", "https://example.org/dataset",
                "--license", "CC-BY-4.0",
                "--license-url", "https://example.org/license",
                "--source-date", "2025-01-15",
                "--output-dir", str(root / "out"),
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be different", result.stderr)


if __name__ == "__main__":
    unittest.main()
