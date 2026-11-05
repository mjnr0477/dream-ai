"""Unit tests for DREAM's deterministic reasoning planner."""
import unittest
from reasoning import Evidence, PlanError, Planner, Task, TaskStatus


class PlannerTests(unittest.TestCase):
    def test_dependency_gates_downstream_tasks(self):
        plan = Planner.build([Task("inspect", "Inspect"), Task("build", "Build", depends_on=("inspect",))])
        self.assertEqual(plan.status("build"), TaskStatus.BLOCKED)
        plan.mark_in_progress("inspect")
        report = plan.submit_outcome("inspect", "Inspected source",
            [Evidence("Inventory exists", "repository review", True, 0.95)])
        self.assertTrue(report.verified)
        self.assertEqual(plan.status("build"), TaskStatus.READY)

    def test_missing_evidence_requires_review(self):
        plan = Planner.build([Task("t", "Task")])
        plan.mark_in_progress("t")
        report = plan.submit_outcome("t", "Probably done")
        self.assertFalse(report.verified)
        self.assertEqual(plan.status("t"), TaskStatus.NEEDS_REVIEW)

    def test_contradiction_and_uncertainty_require_review(self):
        plan = Planner.build([Task("t", "Task")])
        plan.mark_in_progress("t")
        report = plan.submit_outcome("t", "Result", [
            Evidence("Test passed", "test run", True, 1.0),
            Evidence("Manual check failed", "review", False, 0.9)],
            ["Production behavior not checked"])
        self.assertFalse(report.verified)
        self.assertEqual(report.contradicting_evidence, 1)
        self.assertTrue(report.unresolved)

    def test_cycles_and_missing_dependencies_are_rejected(self):
        with self.assertRaises(PlanError):
            Planner.build([Task("a", "A", depends_on=("b",)), Task("b", "B", depends_on=("a",))])
        with self.assertRaises(PlanError):
            Planner.build([Task("a", "A", depends_on=("missing",))])

    def test_retry_resets_previous_attempt(self):
        plan = Planner.build([Task("t", "Task")])
        plan.mark_in_progress("t")
        plan.submit_outcome("t", "Unverified")
        plan.retry("t", "Need a reproducible test")
        self.assertEqual(plan.status("t"), TaskStatus.READY)
        self.assertEqual(plan.records["t"].outcome, "")
        self.assertEqual(plan.records["t"].attempts, 1)

    def test_invalid_reliability_is_rejected(self):
        with self.assertRaises(PlanError):
            Evidence("claim", "source", True, 1.1)


if __name__ == "__main__":
    unittest.main()
