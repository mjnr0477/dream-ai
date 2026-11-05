"""Tests for manifest-backed multilingual evaluation coverage."""
import unittest

from evaluate_multilingual import validate_language_coverage


def manifest_with_coverage(groups_by_split):
    language_files = {
        split: (
            {"sw": {"source_group_ids": groups}}
            if groups else {}
        )
        for split, groups in groups_by_split.items()
    }
    counts = {split: len(groups) for split, groups in groups_by_split.items()}
    all_groups = {group for groups in groups_by_split.values() for group in groups}
    return {
        "schema": "dream-corpus-v1",
        "language_split_coverage": {
            "sw": {
                "independent_source_groups": len(all_groups),
                "eligible_for_three_way_evaluation": len(all_groups) >= 3,
                "source_groups_by_split": counts
            }
        },
        "splits": {
            split: {"language_files": language_files[split]}
            for split in ("train", "validation", "test")
        }
    }


class MultilingualCoverageTests(unittest.TestCase):
    def test_accepts_three_way_language_coverage(self):
        manifest = manifest_with_coverage({
            "train": ["train-source"],
            "validation": ["validation-source"],
            "test": ["test-source"]
        })
        coverage = validate_language_coverage(manifest, "sw")
        self.assertTrue(coverage["eligible_for_three_way_evaluation"])
        self.assertEqual(coverage["independent_source_groups"], 3)
        self.assertEqual(coverage["source_groups_by_split"]["test"], 1)

    def test_reports_rare_language_as_not_eligible(self):
        manifest = manifest_with_coverage({
            "train": [],
            "validation": [],
            "test": ["single-source"]
        })
        coverage = validate_language_coverage(manifest, "sw")
        self.assertFalse(coverage["eligible_for_three_way_evaluation"])
        self.assertEqual(coverage["independent_source_groups"], 1)

    def test_rejects_inaccurate_declared_counts(self):
        manifest = manifest_with_coverage({
            "train": ["train-source"],
            "validation": ["validation-source"],
            "test": ["test-source"]
        })
        manifest["language_split_coverage"]["sw"]["source_groups_by_split"]["test"] = 2
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_language_coverage(manifest, "sw")

    def test_rejects_cross_split_source_group_leakage(self):
        manifest = manifest_with_coverage({
            "train": ["shared-source"],
            "validation": ["validation-source"],
            "test": ["shared-source"]
        })
        with self.assertRaisesRegex(ValueError, "leak across"):
            validate_language_coverage(manifest, "sw")

    def test_rejects_eligible_language_missing_a_split(self):
        manifest = manifest_with_coverage({
            "train": ["train-source", "second-train-source"],
            "validation": ["validation-source"],
            "test": []
        })
        with self.assertRaisesRegex(ValueError, "eligible language"):
            validate_language_coverage(manifest, "sw")

    def test_accepts_legacy_manifest_without_coverage_metadata(self):
        self.assertIsNone(validate_language_coverage({
            "schema": "dream-corpus-v1",
            "splits": {}
        }, "sw"))


if __name__ == "__main__":
    unittest.main()
