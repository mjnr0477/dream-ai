"""Tests for aggregation of human-scored DREAM benchmark reports."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from score_task_benchmark import aggregate, write_json_atomic


def sample_report():
    return {
        "schema": "dream-native-task-benchmark-v1",
        "task_count": 3,
        "checkpoint_sha256": "a" * 64,
        "benchmark_sha256": "b" * 64,
        "tokenizer": "dream-bpe-v1",
        "parameters": 1234,
        "completed_training_steps": 10,
        "data_provenance_verified": True,
        "results": [
            {"id": "math-1", "category": "math", "rubric": ["a", "b"], "rubric_scores": [1, 0]},
            {"id": "math-2", "category": "math", "rubric": ["a", "b"], "rubric_scores": [1, None]},
            {"id": "language-1", "category": "language", "rubric": ["a"], "rubric_scores": [None]}
        ]
    }


class HumanTaskScoreTests(unittest.TestCase):
    def test_aggregates_only_human_scored_rubric_items(self):
        summary = aggregate(sample_report())
        self.assertEqual(summary["task_count"], 3)
        self.assertEqual(summary["complete_tasks"], 1)
        self.assertEqual(summary["scored_rubric_items"], 3)
        self.assertEqual(summary["total_rubric_items"], 5)
        self.assertEqual(summary["earned_points"], 2)
        self.assertEqual(summary["score_percent"], 66.67)
        self.assertEqual(summary["scoring_coverage_percent"], 60.0)
        self.assertEqual(summary["categories"]["math"]["score_percent"], 66.67)
        self.assertEqual(summary["categories"]["language"]["score_percent"], None)
        self.assertEqual(summary["data_provenance_verified"], True)

    def test_aggregates_multilingual_scores_per_language(self):
        report = sample_report()
        report["results"][1].update({
            "language_code": "swa",
            "language_name": "Kiswahili"
        })
        report["results"][2].update({
            "language_code": "ara",
            "language_name": "Arabic"
        })
        summary = aggregate(report)
        self.assertEqual(summary["language_count"], 2)
        self.assertEqual(summary["languages"]["swa"]["language_name"], "Kiswahili")
        self.assertEqual(summary["languages"]["swa"]["score_percent"], 100.0)
        self.assertIsNone(summary["languages"]["ara"]["score_percent"])
        self.assertEqual(summary["tasks"][1]["language_code"], "swa")

    def test_rejects_incomplete_language_score_metadata(self):
        report = sample_report()
        report["results"][0]["language_code"] = "swa"
        with self.assertRaisesRegex(ValueError, "both language_code and language_name"):
            aggregate(report)

    def test_no_scores_yet_returns_null_accuracy_not_zero(self):
        report = sample_report()
        for item in report["results"]:
            item["rubric_scores"] = [None] * len(item["rubric"])
        summary = aggregate(report)
        self.assertIsNone(summary["score_percent"])
        self.assertEqual(summary["scored_rubric_items"], 0)
        self.assertEqual(summary["scoring_coverage_percent"], 0.0)

    def test_rejects_invalid_scores_and_shape(self):
        invalid = []
        bad_score = sample_report()
        bad_score["results"][0]["rubric_scores"][0] = True
        invalid.append(bad_score)
        bad_value = sample_report()
        bad_value["results"][0]["rubric_scores"][0] = 2
        invalid.append(bad_value)
        bad_length = sample_report()
        bad_length["results"][0]["rubric_scores"].pop()
        invalid.append(bad_length)
        bad_count = sample_report()
        bad_count["task_count"] = 99
        invalid.append(bad_count)
        for report in invalid:
            with self.subTest(report=report):
                with self.assertRaises(ValueError):
                    aggregate(report)

    def test_rejects_duplicate_ids(self):
        report = sample_report()
        report["results"][1]["id"] = report["results"][0]["id"]
        with self.assertRaisesRegex(ValueError, "Duplicate task id"):
            aggregate(report)

    def test_writes_json_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "summary.json"
            payload = aggregate(sample_report())
            write_json_atomic(output, payload)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), payload)
            self.assertEqual(list(output.parent.iterdir()), [output])


if __name__ == "__main__":
    unittest.main()
