"""Deterministic task planning with explicit evidence and uncertainty.

This module orchestrates work; it is not a language model and does not generate
reasoning by itself. A model or user supplies tasks, outcomes and evidence.
The planner validates dependencies and records whether completion is justified.
It has no network, shell, filesystem or external-action capabilities.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional, Sequence


class PlanError(ValueError):
    """Raised when a plan or task transition is invalid."""


class TaskStatus(str, Enum):
    BLOCKED = "blocked"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    NEEDS_REVIEW = "needs_review"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class Evidence:
    """A traceable observation supporting or contradicting a task outcome."""
    claim: str
    source: str
    supports: bool
    reliability: float
    observed_at: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.claim.strip() or not self.source.strip():
            raise PlanError("Evidence requires a non-empty claim and source")
        if not 0.0 <= self.reliability <= 1.0:
            raise PlanError("Evidence reliability must be between 0 and 1")


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    description: str = ""
    depends_on: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()
    requires_evidence: bool = True

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.title.strip():
            raise PlanError("Every task needs a non-empty id and title")
        if self.id in self.depends_on:
            raise PlanError(f"Task {self.id!r} cannot depend on itself")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise PlanError(f"Task {self.id!r} has duplicate dependencies")


@dataclass
class TaskRecord:
    task: Task
    status: TaskStatus = TaskStatus.BLOCKED
    outcome: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    attempts: int = 0
    uncertainty: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VerificationReport:
    task_id: str
    verified: bool
    confidence: float
    supporting_evidence: int
    contradicting_evidence: int
    unresolved: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass
class Plan:
    records: dict[str, TaskRecord]

    def status(self, task_id: str) -> TaskStatus:
        return self._record(task_id).status

    def _record(self, task_id: str) -> TaskRecord:
        try:
            return self.records[task_id]
        except KeyError as exc:
            raise PlanError(f"Unknown task: {task_id}") from exc

    def ready_tasks(self) -> list[Task]:
        return [r.task for r in self.records.values()
                if r.status in (TaskStatus.READY, TaskStatus.IN_PROGRESS)]

    def mark_in_progress(self, task_id: str) -> None:
        record = self._record(task_id)
        if record.status != TaskStatus.READY:
            raise PlanError(f"Task {task_id!r} is not ready")
        record.status = TaskStatus.IN_PROGRESS
        record.attempts += 1

    def submit_outcome(self, task_id: str, outcome: str,
                       evidence: Iterable[Evidence] = (),
                       uncertainty: Iterable[str] = ()) -> VerificationReport:
        record = self._record(task_id)
        if record.status != TaskStatus.IN_PROGRESS:
            raise PlanError(f"Task {task_id!r} is not in progress")
        if not outcome.strip():
            raise PlanError("An outcome is required")
        record.outcome = outcome.strip()
        record.evidence.extend(evidence)
        record.uncertainty = [item.strip() for item in uncertainty if item.strip()]
        report = self.verify(task_id)
        record.status = TaskStatus.COMPLETED if report.verified else TaskStatus.NEEDS_REVIEW
        self._refresh_readiness()
        return report

    def verify(self, task_id: str) -> VerificationReport:
        record = self._record(task_id)
        support = [e for e in record.evidence if e.supports]
        contradictions = [e for e in record.evidence if not e.supports]
        unresolved = list(record.uncertainty)
        reasons: list[str] = []
        if not record.outcome:
            reasons.append("No outcome has been submitted")
        if record.task.requires_evidence and not support:
            reasons.append("No supporting evidence was supplied")
        if contradictions:
            reasons.append("Contradictory evidence requires review")
        if unresolved:
            reasons.append("Unresolved uncertainty remains")
        if record.task.success_criteria and not support:
            reasons.append("Success criteria have not been evidenced")
        mean_support = (sum(e.reliability for e in support) / len(support)) if support else 0.0
        penalty = min(0.75, 0.25 * len(contradictions))
        confidence = max(0.0, min(1.0, mean_support - penalty))
        return VerificationReport(
            task_id, bool(record.outcome) and not reasons, round(confidence, 3),
            len(support), len(contradictions), tuple(unresolved), tuple(reasons)
        )

    def retry(self, task_id: str, note: str = "") -> None:
        record = self._record(task_id)
        if record.status not in (TaskStatus.NEEDS_REVIEW, TaskStatus.FAILED):
            raise PlanError("Only a task needing review or marked failed can be retried")
        if note.strip():
            record.uncertainty.append(note.strip())
        record.status = TaskStatus.READY
        record.outcome = ""
        record.evidence.clear()
        record.uncertainty.clear()

    def _refresh_readiness(self) -> None:
        for record in self.records.values():
            if record.status in (TaskStatus.COMPLETED, TaskStatus.FAILED,
                                 TaskStatus.IN_PROGRESS, TaskStatus.NEEDS_REVIEW):
                continue
            deps = [self.records[key].status for key in record.task.depends_on]
            record.status = (TaskStatus.READY if all(s == TaskStatus.COMPLETED for s in deps)
                             else TaskStatus.BLOCKED)


class Planner:
    """Build dependency-safe plans and reject missing references or cycles."""
    @staticmethod
    def build(tasks: Sequence[Task]) -> Plan:
        records: dict[str, TaskRecord] = {}
        for task in tasks:
            if task.id in records:
                raise PlanError(f"Duplicate task id: {task.id}")
            records[task.id] = TaskRecord(task)
        for task in tasks:
            missing = set(task.depends_on) - records.keys()
            if missing:
                raise PlanError(f"Task {task.id!r} depends on unknown task(s): {', '.join(sorted(missing))}")
        indegree = {key: len(record.task.depends_on) for key, record in records.items()}
        children = {key: [] for key in records}
        for key, record in records.items():
            for parent in record.task.depends_on:
                children[parent].append(key)
        queue = [key for key, degree in indegree.items() if degree == 0]
        visited = 0
        while queue:
            current = queue.pop(0)
            visited += 1
            for child in children[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if visited != len(records):
            raise PlanError("Plan contains a dependency cycle")
        plan = Plan(records)
        plan._refresh_readiness()
        return plan
