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

    def test_each_success_criterion_needs_matching_evidence(self):
        plan = Planner.build([Task("t", "Task", success_criteria=("unit tests pass", "API contract holds"))])
        plan.mark_in_progress("t")
        report = plan.submit_outcome("t", "Tests passed", [
            Evidence("Unit suite passed", "CI", True, 0.99, criterion="unit tests pass")
        ])
        self.assertFalse(report.verified)
        self.assertTrue(any("API contract holds" in reason for reason in report.reasons))

    def test_retry_note_is_preserved_as_audit_history(self):
        plan = Planner.build([Task("t", "Task")])
        plan.mark_in_progress("t")
        plan.submit_outcome("t", "Unverified")
        plan.retry("t", "Add an integration test")
        record = plan.records["t"]
        self.assertEqual(record.retry_notes, ["Add an integration test"])
        self.assertEqual(record.uncertainty, [])

    def test_invalid_reliability_is_rejected(self):
        for value in (1.1, float("nan"), float("inf"), True, "0.5", 10 ** 10000):
            with self.subTest(value=value), self.assertRaises(PlanError):
                Evidence("claim", "source", True, value)

    def test_ready_tasks_excludes_work_already_in_progress(self):
        plan = Planner.build([Task("ready", "Ready"), Task("active", "Active")])
        plan.mark_in_progress("active")
        self.assertEqual([task.id for task in plan.ready_tasks()], ["ready"])

    def test_task_and_evidence_reject_malformed_types_with_plan_error(self):
        invalid_tasks = [
            lambda: Task(None, "Title"),
            lambda: Task("id", 12),
            lambda: Task("id", "Title", description=None),
            lambda: Task("id", "Title", depends_on=("ok", 7)),
            lambda: Task("id", "Title", success_criteria=("ok", None)),
            lambda: Task("id", "Title", requires_evidence=1),
        ]
        invalid_evidence = [
            lambda: Evidence(None, "source", True, 0.5),
            lambda: Evidence("claim", [], True, 0.5),
            lambda: Evidence("claim", "source", 1, 0.5),
            lambda: Evidence("claim", "source", True, 0.5, criterion=4),
        ]
        for factory in invalid_tasks + invalid_evidence:
            with self.subTest(factory=factory), self.assertRaises(PlanError):
                factory()

    def test_submit_outcome_validates_all_inputs_before_mutation(self):
        plan = Planner.build([Task("t", "Task")])
        plan.mark_in_progress("t")
        with self.assertRaises(PlanError):
            plan.submit_outcome("t", "Done", [Evidence("ok", "test", True, 1.0)], [None])
        record = plan.records["t"]
        self.assertEqual(record.outcome, "")
        self.assertEqual(record.evidence, [])
        self.assertEqual(record.uncertainty, [])

        with self.assertRaises(PlanError):
            plan.submit_outcome("t", 123)
        with self.assertRaises(PlanError):
            plan.submit_outcome("t", "Done", [object()])
        with self.assertRaises(PlanError):
            plan.submit_outcome("t", "Done", uncertainty="not-an-array")
        with self.assertRaises(PlanError):
            plan.status([])


if __name__ == "__main__":
    unittest.main()
