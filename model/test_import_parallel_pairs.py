"""Tests for importing licensed parallel-pair exports into SFT review candidates."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dream_transformer import DreamByteTokenizer
from sft import load_examples

SCRIPT = Path(__file__).with_name("import_parallel_pairs.py")


class ParallelPairImportTests(unittest.TestCase):
    def make_input(self, root, count=40):
        path = root / "parallel.jsonl"
        rows = [
            {
                "id": f"pair-{index}",
                "translation": {
                    "dav": f"Source language sentence number {index} with enough text.",
                    "swa": f"Sentensi ya Kiswahili nambari {index} yenye maneno ya kutosha."
                }
            }
            for index in range(count)
        ]
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8"
        )
        return path

    def run_import(self, source, output, extra=()):
        return subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--input", str(source),
                "--source-lang", "dav",
                "--target-lang", "swa",
                "--source-name", "Kenyan Low-Resource Language Parallel Corpora",
                "--source-url", "https://huggingface.co/datasets/thinkKenya/kenyan-low-resource-language-data",
                "--license", "CC-BY-4.0",
                "--license-url", "https://creativecommons.org/licenses/by/4.0/",
                "--source-date", "2026-11-03",
                "--output-dir", str(output),
                "--validation-percent", "20",
                *extra
            ],
            capture_output=True, text=True, timeout=30
        )

    def test_creates_deterministic_unreviewed_train_and_validation_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_input(root)
            output = root / "candidates"
            result = self.run_import(source, output)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            train_path = output / "sft-train-candidates.jsonl"
            validation_path = output / "sft-validation-candidates.jsonl"
            self.assertTrue(train_path.is_file())
            self.assertTrue(validation_path.is_file())
            train = [json.loads(line) for line in train_path.read_text().splitlines()]
            validation = [json.loads(line) for line in validation_path.read_text().splitlines()]
            self.assertTrue(train)
            self.assertTrue(validation)
            self.assertFalse(
                {row["source_group"] for row in train}
                & {row["source_group"] for row in validation}
            )
            for row in train + validation:
                self.assertEqual(row["rights_basis"], "open-license")
                self.assertFalse(row["rights_confirmed"])
                self.assertFalse(row["privacy_reviewed"])
                self.assertFalse(row["quality_reviewed"])
                self.assertEqual(row["messages"][0]["role"], "user")
                self.assertEqual(row["messages"][1]["role"], "assistant")
                self.assertIn("Translate", row["messages"][0]["content"])
            summary = json.loads((output / "import-summary.json").read_text())
            self.assertEqual(summary["input_pairs"], 40)
            self.assertEqual(summary["duplicate_pairs_removed"], 0)
            self.assertEqual(summary["review_status"], "human_review_required")

            # The importer output becomes valid SFT input only after a human
            # has explicitly completed the three review attestations.
            reviewed_path = root / "reviewed-sft-train.jsonl"
            for row in train:
                row["rights_confirmed"] = True
                row["privacy_reviewed"] = True
                row["quality_reviewed"] = True
            reviewed_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + chr(10) for row in train),
                encoding="utf-8"
            )
            examples = load_examples(reviewed_path, DreamByteTokenizer(), block_size=128)
            self.assertEqual(len(examples), len(train))

    def test_pair_fingerprint_separator_cannot_collapse_distinct_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_input(root, count=40)
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            rows[0]["translation"] = {"dav": "a", "swa": "bc"}
            rows[1]["translation"] = {"dav": "ab", "swa": "c"}
            source.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8"
            )
            output = root / "fingerprints"
            result = self.run_import(source, output)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads((output / "import-summary.json").read_text())
            self.assertEqual(summary["unique_pairs"], 40)
            self.assertEqual(summary["duplicate_pairs_removed"], 0)

    def test_rejects_conflicting_translations_for_same_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_input(root, count=40)
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            rows[1]["translation"]["dav"] = rows[0]["translation"]["dav"]
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8"
            )
            result = self.run_import(source, root / "conflicting-source")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Conflicting translations for the same normalized source sentence", result.stderr)

    def test_deduplicates_identical_parallel_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_input(root)
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            rows[1]["translation"] = dict(rows[0]["translation"])
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8"
            )
            output = root / "candidates"
            result = self.run_import(source, output)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads((output / "import-summary.json").read_text())
            self.assertEqual(summary["duplicate_pairs_removed"], 1)
            self.assertEqual(summary["unique_pairs"], 39)

    def test_keeps_story_groups_together_across_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_input(root, count=40)
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            for index, row in enumerate(rows):
                row["story_id"] = f"story-{index // 2}"
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8"
            )
            output = root / "story-groups"
            result = self.run_import(source, output)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            train = [json.loads(line) for line in (output / "sft-train-candidates.jsonl").read_text().splitlines()]
            validation = [json.loads(line) for line in (output / "sft-validation-candidates.jsonl").read_text().splitlines()]
            train_groups = {row["source_group"] for row in train}
            validation_groups = {row["source_group"] for row in validation}
            self.assertFalse(train_groups & validation_groups)
            self.assertTrue(all(group.startswith("story-") for group in train_groups | validation_groups))
            summary = json.loads((output / "import-summary.json").read_text())
            self.assertEqual(summary["source_group_basis_counts"]["story_id"], 40)

    def test_rejects_duplicate_record_ids_for_different_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_input(root, count=40)
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            rows[1]["id"] = rows[0]["id"]
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8"
            )
            result = self.run_import(source, root / "duplicate-ids")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Duplicate parallel record id", result.stderr)

    def test_rejects_invalid_language_fields_and_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_input(root, count=4)
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            rows[0]["translation"].pop("dav")
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8"
            )
            result = self.run_import(source, root / "bad")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing source language", result.stderr)

            source = self.make_input(root, count=40)
            result = self.run_import(source, root / "bad-date", extra=("--source-date", "03-11-2026"))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("YYYY-MM-DD", result.stderr)


if __name__ == "__main__":
    unittest.main()
