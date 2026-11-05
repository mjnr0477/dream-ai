"""Strict JSON boundary for model-proposed DREAM task plans.

This adapter validates untrusted structured output before it reaches the planner.
It does not execute tasks, call tools, or grant external-action permissions.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .planner import Plan, PlanError, Planner, Task

_MAX_TASKS = 100
_MAX_TEXT = 4000


def _text(value: Any, field: str, *, optional: bool = False) -> str:
    if optional and value is None:
        return ""
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"{field} must be a non-empty string")
    if len(value) > _MAX_TEXT:
        raise PlanError(f"{field} exceeds {_MAX_TEXT} characters")
    return value.strip()


def plan_from_payload(payload: Mapping[str, Any]) -> Plan:
    """Validate a JSON-decoded mapping and build a dependency-safe plan.

    Expected shape: {"tasks": [{"id", "title", "description"?,
    "depends_on"?, "success_criteria"?, "requires_evidence"?}]}.
    Unknown keys are rejected to catch schema drift instead of silently
    accepting fields the planner does not enforce.
    """
    if not isinstance(payload, Mapping):
        raise PlanError("Plan payload must be an object")
    if set(payload) != {"tasks"}:
        raise PlanError("Plan payload must contain only the 'tasks' field")
    raw_tasks = payload["tasks"]
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise PlanError("tasks must be a non-empty array")
    if len(raw_tasks) > _MAX_TASKS:
        raise PlanError(f"Plan cannot contain more than {_MAX_TASKS} tasks")

    tasks: list[Task] = []
    allowed = {"id", "title", "description", "depends_on",
               "success_criteria", "requires_evidence"}
    for index, raw in enumerate(raw_tasks):
        label = f"tasks[{index}]"
        if not isinstance(raw, Mapping):
            raise PlanError(f"{label} must be an object")
        unknown = set(raw) - allowed
        if unknown:
            raise PlanError(f"{label} has unsupported field(s): {', '.join(sorted(unknown))}")
        task_id = _text(raw.get("id"), f"{label}.id")
        title = _text(raw.get("title"), f"{label}.title")
        description = _text(raw.get("description", ""), f"{label}.description", optional=True)
        dependencies = raw.get("depends_on", [])
        criteria = raw.get("success_criteria", [])
        if not isinstance(dependencies, list) or not all(isinstance(x, str) and x.strip() for x in dependencies):
            raise PlanError(f"{label}.depends_on must be an array of non-empty strings")
        if not isinstance(criteria, list) or not all(isinstance(x, str) and x.strip() for x in criteria):
            raise PlanError(f"{label}.success_criteria must be an array of non-empty strings")
        requires_evidence = raw.get("requires_evidence", True)
        if not isinstance(requires_evidence, bool):
            raise PlanError(f"{label}.requires_evidence must be a boolean")
        tasks.append(Task(
            id=task_id, title=title, description=description,
            depends_on=tuple(x.strip() for x in dependencies),
            success_criteria=tuple(x.strip() for x in criteria),
            requires_evidence=requires_evidence,
        ))
    return Planner.build(tasks)


def plan_snapshot(plan: Plan) -> dict[str, Any]:
    """Return a JSON-serializable summary of plan state."""
    return {
        "tasks": [
            {
                "id": record.task.id,
                "title": record.task.title,
                "status": record.status.value,
                "depends_on": list(record.task.depends_on),
                "attempts": record.attempts,
                "retry_notes": list(record.retry_notes),
            }
            for record in plan.records.values()
        ]
    }
