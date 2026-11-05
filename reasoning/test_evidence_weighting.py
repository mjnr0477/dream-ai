"""Regression tests for reliability-weighted contradiction evidence."""
import unittest

from reasoning import Evidence, Planner, Task, TaskStatus


class EvidenceWeightingTests(unittest.TestCase):
    def _confidence(self, contradiction_reliability):
        plan = Planner.build([Task("t", "Task")])
        plan.mark_in_progress("t")
        return plan.submit_outcome("t", "Observed result", [
            Evidence("Primary check passed", "automated test", True, 0.9),
            Evidence("Conflicting observation", "review", False, contradiction_reliability),
        ]).confidence

    def test_low_reliability_conflict_has_smaller_penalty(self):
        self.assertAlmostEqual(self._confidence(0.1), 0.875)
        self.assertAlmostEqual(self._confidence(0.9), 0.675)
        self.assertGreater(self._confidence(0.1), self._confidence(0.9))

    def test_even_weak_contradiction_still_blocks_completion(self):
        plan = Planner.build([Task("t", "Task")])
        plan.mark_in_progress("t")
        report = plan.submit_outcome("t", "Observed result", [
            Evidence("Check passed", "automated test", True, 1.0),
            Evidence("Weak conflicting signal", "unverified note", False, 0.01),
        ])
        self.assertFalse(report.verified)
        self.assertEqual(plan.status("t"), TaskStatus.NEEDS_REVIEW)


if __name__ == "__main__":
    unittest.main()
