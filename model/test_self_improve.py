"""Tests for DREAM's conservative self-improvement promotion gate."""
import math
import unittest

from self_improve import decide_promotion


class SelfImprovementGateTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {"bits_per_byte": 1.25}
        self.candidate = {"bits_per_byte": 1.20}
        self.baseline_tasks = {"exact_match_rate": 0.30}
        self.candidate_tasks = {"exact_match_rate": 0.35}

    def test_promotes_only_when_held_out_loss_and_task_score_do_not_regress(self):
        result = decide_promotion(
            self.baseline, self.candidate,
            self.baseline_tasks, self.candidate_tasks,
        )
        self.assertTrue(result["promote"])
        self.assertTrue(result["checks"]["held_out_language_model_loss_not_regressed"])
        self.assertTrue(result["checks"]["task_benchmark_not_regressed"])

    def test_rejects_held_out_loss_regression(self):
        candidate = {"bits_per_byte": 1.251}
        result = decide_promotion(
            self.baseline, candidate,
            self.baseline_tasks, self.candidate_tasks,
        )
        self.assertFalse(result["promote"])
        self.assertFalse(result["checks"]["held_out_language_model_loss_not_regressed"])

    def test_rejects_task_benchmark_regression(self):
        candidate_tasks = {"exact_match_rate": 0.29}
        result = decide_promotion(
            self.baseline, self.candidate,
            self.baseline_tasks, candidate_tasks,
        )
        self.assertFalse(result["promote"])
        self.assertFalse(result["checks"]["task_benchmark_not_regressed"])

    def test_allows_first_instruction_checkpoint_without_task_baseline(self):
        result = decide_promotion(self.baseline, self.candidate, None, self.candidate_tasks)
        self.assertTrue(result["promote"])
        self.assertFalse(result["task_baseline_available"])

    def test_rejects_missing_or_non_finite_metrics(self):
        for value in (None, True, float("nan"), float("inf"), -1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    decide_promotion(
                        self.baseline, {"bits_per_byte": value},
                        self.baseline_tasks, self.candidate_tasks,
                    )

    def test_rejects_invalid_tolerance(self):
        for tolerance in (-0.1, float("nan"), math.inf, True):
            with self.subTest(tolerance=tolerance):
                with self.assertRaises(ValueError):
                    decide_promotion(
                        self.baseline, self.candidate,
                        self.baseline_tasks, self.candidate_tasks,
                        tolerance,
                    )

    def test_tolerance_is_explicit_and_applied_only_to_language_loss(self):
        result = decide_promotion(
            self.baseline, {"bits_per_byte": 1.26},
            self.baseline_tasks, self.candidate_tasks,
            max_regression_bits_per_byte=0.02,
        )
        self.assertTrue(result["promote"])


if __name__ == "__main__":
    unittest.main()
