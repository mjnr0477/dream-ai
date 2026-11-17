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



class NativeTrainingOrchestrationContractTests(unittest.TestCase):
    """Guard the shell controller against accidentally restarting the clock."""

    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        cls.script = Path(__file__).with_name(
            "train-native-dream-gpu-container.sh"
        ).read_text(encoding="utf-8")

    def test_one_deadline_is_created_only_after_both_audits(self):
        audit = self.script.index("python model/audit_corpus.py")
        split = self.script.index("python model/prepare_corpus.py")
        deadline_line = 'SESSION_DEADLINE="$(python "$BUDGET_HELPER" deadline --seconds "$SESSION_BUDGET_SECONDS")"'
        deadline = self.script.index(deadline_line)
        self.assertEqual(self.script.count(deadline_line), 1)
        self.assertLess(audit, split)
        self.assertLess(split, deadline)

    def test_pretraining_and_sft_receive_the_identical_deadline(self):
        pretrain_start = self.script.index("python model/train.py \\")
        pretrain_end = self.script.index('\\n[[ -s "$PRETRAIN" ]]', pretrain_start)
        sft_start = self.script.index("python model/sft.py \\")
        sft_end = self.script.index('\\n\\n[[ -s "$SFT" ]]', sft_start)
        pretrain = self.script[pretrain_start:pretrain_end]
        sft = self.script[sft_start:sft_end]
        argument = '--deadline-monotonic "$SESSION_DEADLINE"'
        self.assertIn(argument, pretrain)
        self.assertIn(argument, sft)
        self.assertEqual(self.script.count(argument), 2)

    def test_sft_is_deferred_when_pretraining_consumes_the_budget(self):
        remaining_check = self.script.index(
            "if (( REMAINING_SESSION_SECONDS <= 0 )); then"
        )
        sft_start = self.script.index("python model/sft.py \\")
        defer_block = self.script[remaining_check:sft_start]
        self.assertLess(remaining_check, sft_start)
        self.assertIn("SFT is safely deferred", defer_block)
        self.assertIn("exit 0", defer_block)

    def test_independent_per_stage_time_budgets_are_not_reintroduced(self):
        self.assertNotIn("DREAM_NATIVE_PRETRAIN_MAX_SECONDS", self.script)
        self.assertNotIn("DREAM_NATIVE_SFT_MAX_SECONDS", self.script)
        self.assertNotIn("--max-seconds", self.script)

if __name__ == "__main__":
    unittest.main()
