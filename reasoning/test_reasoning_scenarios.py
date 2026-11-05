"""Adversarial scenario tests for multi-step reasoning orchestration.

These tests verify planner guarantees, not the language model's ability to
invent a correct plan or solve the task. Outcomes and evidence are explicit.
"""
import unittest

from reasoning import Evidence, PlanError, Planner, Task, TaskStatus


class ReasoningScenarioTests(unittest.TestCase):
    @staticmethod
    def _complete(plan, task_id, claim="Observed check passed"):
        plan.mark_in_progress(task_id)
        return plan.submit_outcome(task_id, claim, [
            Evidence(claim, "automated regression test", True, 1.0)
        ])

    def test_three_stage_chain_unlocks_only_after_verified_predecessor(self):
        plan = Planner.build([
            Task("report", "Report result", depends_on=("analyze",)),
            Task("analyze", "Analyze data", depends_on=("collect",)),
            Task("collect", "Collect data"),
        ])
        self.assertEqual(plan.status("collect"), TaskStatus.READY)
        self.assertEqual(plan.status("analyze"), TaskStatus.BLOCKED)
        self.assertEqual(plan.status("report"), TaskStatus.BLOCKED)
        self._complete(plan, "collect")
        self.assertEqual(plan.status("analyze"), TaskStatus.READY)
        self.assertEqual(plan.status("report"), TaskStatus.BLOCKED)
        self._complete(plan, "analyze")
        self.assertEqual(plan.status("report"), TaskStatus.READY)
        self._complete(plan, "report")
        self.assertEqual(plan.status("report"), TaskStatus.COMPLETED)

    def test_fan_in_waits_for_every_independent_prerequisite(self):
        plan = Planner.build([
            Task("final", "Combine results", depends_on=("math", "research")),
            Task("math", "Check calculation"),
            Task("research", "Verify source"),
        ])
        self._complete(plan, "math")
        self.assertEqual(plan.status("final"), TaskStatus.BLOCKED)
        self._complete(plan, "research")
        self.assertEqual(plan.status("final"), TaskStatus.READY)

    def test_low_reliability_contradiction_still_blocks_downstream_work(self):
        plan = Planner.build([
            Task("publish", "Publish answer", depends_on=("verify",)),
            Task("verify", "Verify answer"),
        ])
        plan.mark_in_progress("verify")
        report = plan.submit_outcome("verify", "Calculation appears correct", [
            Evidence("Arithmetic check passed", "calculator", True, 0.99),
            Evidence("Possible input mismatch", "unverified note", False, 0.01),
        ])
        self.assertFalse(report.verified)
        self.assertEqual(plan.status("verify"), TaskStatus.NEEDS_REVIEW)
        self.assertEqual(plan.status("publish"), TaskStatus.BLOCKED)

    def test_ambiguous_input_remains_unresolved_instead_of_being_assumed(self):
        plan = Planner.build([Task("answer", "Answer price question")])
        plan.mark_in_progress("answer")
        report = plan.submit_outcome(
            "answer",
            "The exact price cannot be established from the supplied context",
            [Evidence("No live price source was supplied", "input audit", True, 1.0)],
            uncertainty=["Product size and market location are unspecified",
                         "No current market quote is available"],
        )
        self.assertFalse(report.verified)
        self.assertEqual(len(report.unresolved), 2)
        self.assertEqual(plan.status("answer"), TaskStatus.NEEDS_REVIEW)

    def test_multistep_math_requires_evidence_for_each_calculation_stage(self):
        plan = Planner.build([Task(
            "solve", "Solve sequential inventory calculation",
            success_criteria=("subtract morning sales", "subtract afternoon sales",
                              "add purchased stock"),
        )])
        plan.mark_in_progress("solve")
        partial = plan.submit_outcome("solve", "18 - 6 - 4 + 36 = 44", [
            Evidence("Morning sale: 18 / 3 = 6", "calculation", True, 1.0,
                     criterion="subtract morning sales"),
            Evidence("Afternoon sale: subtract 4", "calculation", True, 1.0,
                     criterion="subtract afternoon sales"),
        ])
        self.assertFalse(partial.verified)
        self.assertTrue(any("add purchased stock" in reason for reason in partial.reasons))
        self.assertEqual(plan.status("solve"), TaskStatus.NEEDS_REVIEW)

    def test_duplicate_task_ids_are_rejected(self):
        with self.assertRaises(PlanError):
            Planner.build([Task("same", "First"), Task("same", "Second")])


if __name__ == "__main__":
    unittest.main()
