"""Deterministic task planning with explicit evidence and uncertainty.

This module orchestrates work; it is not a language model and does not generate
reasoning by itself. A model or user supplies tasks, outcomes and evidence.
The planner validates dependencies and records whether completion is justified.
It has no network, shell, filesystem or external-action capabilities.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import math
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
    criterion: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.claim, str) or not self.claim.strip():
            raise PlanError("Evidence claim must be a non-empty string")
        if not isinstance(self.source, str) or not self.source.strip():
            raise PlanError("Evidence source must be a non-empty string")
        if not isinstance(self.supports, bool):
            raise PlanError("Evidence supports must be a boolean")
        if isinstance(self.reliability, bool) or not isinstance(self.reliability, (int, float)):
            raise PlanError("Evidence reliability must be a finite number between 0 and 1")
        try:
            reliability = float(self.reliability)
        except (OverflowError, ValueError) as exc:
            raise PlanError("Evidence reliability must be a finite number between 0 and 1") from exc
        if not math.isfinite(reliability) or not 0.0 <= reliability <= 1.0:
            raise PlanError("Evidence reliability must be a finite number between 0 and 1")
        if self.observed_at is not None and (not isinstance(self.observed_at, str) or not self.observed_at.strip()):
            raise PlanError("Evidence observed_at must be a non-empty string when supplied")
        if self.criterion is not None and (not isinstance(self.criterion, str) or not self.criterion.strip()):
            raise PlanError("Evidence criterion must be a non-empty string when supplied")


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    description: str = ""
    depends_on: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()
    requires_evidence: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise PlanError("Every task needs a non-empty string id")
        if not isinstance(self.title, str) or not self.title.strip():
            raise PlanError("Every task needs a non-empty string title")
        if not isinstance(self.description, str):
            raise PlanError("Task description must be a string")
        if not isinstance(self.depends_on, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.depends_on
        ):
            raise PlanError("Task dependencies must be a tuple of non-empty strings")
        if not isinstance(self.success_criteria, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.success_criteria
        ):
            raise PlanError("Task success criteria must be a tuple of non-empty strings")
        if not isinstance(self.requires_evidence, bool):
            raise PlanError("Task requires_evidence must be a boolean")
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
    retry_notes: list[str] = field(default_factory=list)


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
        if not isinstance(task_id, str) or not task_id:
            raise PlanError("Task id must be a non-empty string")
        try:
            return self.records[task_id]
        except KeyError as exc:
            raise PlanError(f"Unknown task: {task_id}") from exc

    def ready_tasks(self) -> list[Task]:
        """Return only tasks that are eligible to be started."""
        return [record.task for record in self.records.values()
                if record.status == TaskStatus.READY]

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
        if not isinstance(outcome, str) or not outcome.strip():
            raise PlanError("An outcome must be a non-empty string")
        try:
            evidence_items = tuple(evidence)
        except TypeError as exc:
            raise PlanError("Evidence must be an iterable of Evidence records") from exc
        if any(not isinstance(item, Evidence) for item in evidence_items):
            raise PlanError("Evidence must contain only Evidence records")
        if isinstance(uncertainty, (str, bytes)):
            raise PlanError("Uncertainty must be an iterable of non-empty strings")
        try:
            uncertainty_items = tuple(uncertainty)
        except TypeError as exc:
            raise PlanError("Uncertainty must be an iterable of non-empty strings") from exc
        if any(not isinstance(item, str) or not item.strip() for item in uncertainty_items):
            raise PlanError("Uncertainty must contain only non-empty strings")

        # Validate every supplied value before mutating the task record.
        record.outcome = outcome.strip()
        record.evidence.extend(evidence_items)
        record.uncertainty = [item.strip() for item in uncertainty_items]
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
        if record.task.success_criteria:
            evidenced = {e.criterion for e in support if e.criterion}
            missing = [criterion for criterion in record.task.success_criteria
                       if criterion not in evidenced]
            if missing:
                reasons.append("Success criteria lack matching evidence: " + ", ".join(missing))
        mean_support = (sum(e.reliability for e in support) / len(support)) if support else 0.0
        # Contradictions reduce the evidence-strength indicator in proportion
        # to their caller-supplied reliability. Counting each contradiction
        # equally made a weak, low-reliability objection as influential as a
        # directly observed failure. This remains a heuristic, not a probability.
        contradiction_penalty = sum(0.25 * e.reliability for e in contradictions)
        penalty = min(0.75, contradiction_penalty)
        confidence = max(0.0, min(1.0, mean_support - penalty))
        return VerificationReport(
            task_id, bool(record.outcome) and not reasons, round(confidence, 3),
            len(support), len(contradictions), tuple(unresolved), tuple(reasons)
        )

    def retry(self, task_id: str, note: str = "") -> None:
        record = self._record(task_id)
        if not isinstance(note, str):
            raise PlanError("Retry note must be a string")
        if record.status not in (TaskStatus.NEEDS_REVIEW, TaskStatus.FAILED):
            raise PlanError("Only a task needing review or marked failed can be retried")
        if note.strip():
            record.retry_notes.append(note.strip())
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
        if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Sequence):
            raise PlanError("Tasks must be a sequence of Task records")
        records: dict[str, TaskRecord] = {}
        for task in tasks:
            if not isinstance(task, Task):
                raise PlanError("Tasks must contain only Task records")
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
