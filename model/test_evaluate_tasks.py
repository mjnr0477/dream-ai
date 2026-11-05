"""Tests for DREAM's deterministic core task benchmark."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).parent / "evaluate_tasks.py"
SPEC = importlib.util.spec_from_file_location("dream_task_eval", MODULE)
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


class TaskEvaluationTests(unittest.TestCase):
    def test_normalizes_punctuation_and_unicode_without_ascii_loss(self):
        self.assertEqual(evaluator.normalize_answer("  Habari, ZA asubuhi! "), "habari za asubuhi")
        self.assertEqual(evaluator.normalize_answer("Nairobi."), "nairobi")
        self.assertEqual(evaluator.normalize_answer("42"), "42")

    def test_loads_valid_unique_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.jsonl"
            path.write_text(
                json.dumps({
                    "id": "one", "category": "math", "language": "en",
                    "prompt": "What is 2 + 2?", "reference": "4"
                }) + "\n",
                encoding="utf-8",
            )
            tasks = evaluator.load_tasks(path)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["reference"], "4")

    def test_rejects_duplicate_ids_and_multiline_prompts(self):
        task = {
            "id": "one", "category": "math", "language": "en",
            "prompt": "What is 2 + 2?", "reference": "4"
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.jsonl"
            path.write_text(json.dumps(task) + "\n" + json.dumps(task) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate benchmark id"):
                evaluator.load_tasks(path)
            task["prompt"] = "line one\nline two"
            path.write_text(json.dumps(task) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one line"):
                evaluator.load_tasks(path)

    def test_rejects_empty_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.jsonl"
            path.write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no tasks"):
                evaluator.load_tasks(path)

    def test_reports_per_category_and_language(self):
        results = [
            {"category": "math", "language": "en", "correct": True},
            {"category": "math", "language": "en", "correct": False},
            {"category": "translation", "language": "sw", "correct": True},
        ]
        self.assertEqual(
            evaluator.summarize(results, "category")["math"],
            {"correct": 1, "total": 2, "exact_match_rate": 0.5},
        )
        self.assertEqual(
            evaluator.summarize(results, "language")["sw"],
            {"correct": 1, "total": 1, "exact_match_rate": 1.0},
        )


if __name__ == "__main__":
    unittest.main()
