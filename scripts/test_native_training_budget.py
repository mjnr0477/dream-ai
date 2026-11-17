"""Tests for the shared five-minute native DREAM training budget."""
import unittest

from native_training_budget import (
    MAX_SESSION_SECONDS,
    create_deadline,
    seconds_remaining,
    validate_session_seconds,
)


class NativeTrainingBudgetTests(unittest.TestCase):
    def test_default_maximum_is_five_minutes(self):
        self.assertEqual(MAX_SESSION_SECONDS, 300)
        self.assertEqual(validate_session_seconds("300"), 300)

    def test_accepts_shorter_integer_budgets(self):
        for value in ("1", "60", "299"):
            with self.subTest(value=value):
                self.assertEqual(validate_session_seconds(value), int(value))

    def test_rejects_unbounded_fractional_or_invalid_budgets(self):
        for value in ("0", "-1", "301", "3600", "3.5", "nan", "inf", ""):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_session_seconds(value)

    def test_deadline_is_anchored_to_monotonic_start(self):
        self.assertEqual(create_deadline("300", now=1000.25), 1300.25)

    def test_remaining_time_never_rounds_up(self):
        deadline = 1300.0
        self.assertEqual(seconds_remaining(deadline, now=1000.0), 300)
        self.assertEqual(seconds_remaining(deadline, now=1000.1), 299)
        self.assertEqual(seconds_remaining(deadline, now=1299.1), 0)
        self.assertEqual(seconds_remaining(deadline, now=1300.0), 0)
        self.assertEqual(seconds_remaining(deadline, now=1301.0), 0)

    def test_rejects_non_finite_times(self):
        for deadline, now in ((float("nan"), 1), (2, float("inf")), (-1, 0)):
            with self.subTest(deadline=deadline, now=now):
                with self.assertRaises(ValueError):
                    seconds_remaining(deadline, now=now)


if __name__ == "__main__":
    unittest.main()
