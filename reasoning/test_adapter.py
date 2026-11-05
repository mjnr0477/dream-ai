"""Tests for the strict model-output adapter."""
import unittest

from reasoning import PlanError, TaskStatus
from reasoning.adapter import plan_from_payload, plan_snapshot


class AdapterTests(unittest.TestCase):
    def test_valid_payload_builds_dependency_order(self):
        plan = plan_from_payload({"tasks": [
            {"id": "research", "title": "Research", "success_criteria": ["sources checked"]},
            {"id": "design", "title": "Design", "depends_on": ["research"]},
        ]})
        self.assertEqual(plan.status("research"), TaskStatus.READY)
        self.assertEqual(plan.status("design"), TaskStatus.BLOCKED)
        self.assertEqual(plan_snapshot(plan)["tasks"][1]["depends_on"], ["research"])

    def test_unknown_fields_are_rejected(self):
        with self.assertRaises(PlanError):
            plan_from_payload({"tasks": [{"id": "x", "title": "X", "tool": "shell"}]})

    def test_invalid_types_are_rejected(self):
        with self.assertRaises(PlanError):
            plan_from_payload({"tasks": [{"id": "x", "title": "X", "requires_evidence": "false"}]})

    def test_empty_or_oversized_plan_is_rejected(self):
        with self.assertRaises(PlanError):
            plan_from_payload({"tasks": []})
        with self.assertRaises(PlanError):
            plan_from_payload({"tasks": [{"id": str(i), "title": "T"} for i in range(101)]})


if __name__ == "__main__":
    unittest.main()
