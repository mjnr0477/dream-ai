"""Tests for deterministic source-group-disjoint DREAM SFT splitting."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.split_sft import split_sft


def sft_record(record_id, group):
    return {
        "id": record_id,
        "source": "OpenAssistant OASST1",
        "source_group": group,
        "source_url": "https://huggingface.co/datasets/OpenAssistant/oasst1",
        "license": "Apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "rights_basis": "open-license",
        "source_date": "2023-04-12",
        "rights_confirmed": True,
        "privacy_reviewed": True,
        "quality_reviewed": True,
        "approval_granted": True,
        "language": "en",
        "messages": [
            {"role": "user", "content": f"Explain the topic for record {record_id} in simple terms."},
            {"role": "assistant", "content": f"This is a sufficiently detailed assistant answer for {record_id}, with a clear explanation."}
        ]
    }


def write_records(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8"
    )


class SplitSFTTests(unittest.TestCase):
    def test_split_is_source_group_disjoint_and_writes_valid_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "approved.jsonl"
            train = root / "sft-train.jsonl"
            validation = root / "sft-validation.jsonl"
            manifest_path = root / "sft-split-manifest.json"
            write_records(source, [sft_record(f"id-{i}", f"group-{i}") for i in range(4)])

            summary = split_sft(source, train, validation, manifest_path, validation_fraction=0.25, seed=7)
            train_rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
            validation_rows = [json.loads(line) for line in validation.read_text(encoding="utf-8").splitlines()]
            train_groups = {row["source_group"] for row in train_rows}
            validation_groups = {row["source_group"] for row in validation_rows}
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(summary["records"], 4)
            self.assertEqual(summary["train_records"], 3)
            self.assertEqual(summary["validation_records"], 1)
            self.assertTrue(train_groups.isdisjoint(validation_groups))
            self.assertTrue(manifest["source_groups_disjoint"])
            self.assertEqual(manifest["record_count"], 4)
            self.assertEqual(manifest["train_records"], 3)
            self.assertEqual(manifest["validation_records"], 1)
            self.assertNotIn("group-0", json.dumps(manifest))

    def test_same_seed_produces_identical_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "approved.jsonl"
            write_records(source, [sft_record(f"id-{i}", f"group-{i}") for i in range(8)])
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            split_sft(source, first / "train.jsonl", first / "validation.jsonl", first / "manifest.json", seed=99)
            split_sft(source, second / "train.jsonl", second / "validation.jsonl", second / "manifest.json", seed=99)
            self.assertEqual((first / "train.jsonl").read_bytes(), (second / "train.jsonl").read_bytes())
            self.assertEqual((first / "validation.jsonl").read_bytes(), (second / "validation.jsonl").read_bytes())

    def test_unapproved_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "approved.jsonl"
            item = sft_record("id-1", "group-1")
            item["approval_granted"] = False
            write_records(source, [item, sft_record("id-2", "group-2")])
            with self.assertRaisesRegex(ValueError, "approval_granted"):
                split_sft(source, root / "train.jsonl", root / "validation.jsonl", root / "manifest.json")

    def test_duplicate_conversation_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "approved.jsonl"
            first = sft_record("id-1", "group-1")
            second = sft_record("id-2", "group-2")
            second["messages"] = first["messages"]
            write_records(source, [first, second])
            with self.assertRaisesRegex(ValueError, "Duplicate conversation content"):
                split_sft(source, root / "train.jsonl", root / "validation.jsonl", root / "manifest.json")

    def test_at_least_two_independent_source_groups_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "approved.jsonl"
            write_records(source, [sft_record("id-1", "same-group"), sft_record("id-2", "same-group")])
            with self.assertRaisesRegex(ValueError, "At least two independent source groups"):
                split_sft(source, root / "train.jsonl", root / "validation.jsonl", root / "manifest.json")

    def test_existing_outputs_are_not_overwritten_without_explicit_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "approved.jsonl"
            train = root / "train.jsonl"
            train.write_text("keep me", encoding="utf-8")
            write_records(source, [sft_record("id-1", "group-1"), sft_record("id-2", "group-2")])
            with self.assertRaises(FileExistsError):
                split_sft(source, train, root / "validation.jsonl", root / "manifest.json")
            self.assertEqual(train.read_text(encoding="utf-8"), "keep me")

    def test_input_and_output_paths_must_be_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "approved.jsonl"
            write_records(source, [sft_record("id-1", "group-1"), sft_record("id-2", "group-2")])
            with self.assertRaisesRegex(ValueError, "paths must be different"):
                split_sft(source, source, root / "validation.jsonl", root / "manifest.json")


if __name__ == "__main__":
    unittest.main()
