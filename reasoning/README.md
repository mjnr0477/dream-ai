# DREAM reasoning and planning primitives

This package adds a deterministic orchestration layer for multi-step tasks. It
does not replace the native transformer, generate model reasoning, execute
commands, browse the network or perform external actions.

## Design

- Explicit task IDs, descriptions, success criteria and dependencies.
- Dependency validation with cycle detection before execution.
- Strict lifecycle: blocked → ready → in progress → completed or needs review.
- Evidence records include a claim, source, support/contradiction flag and
  caller-assigned reliability.
- Contradictions and unresolved uncertainty prevent automatic completion.
- Verification failures can be retried without retaining stale outcomes.
- No tools, network access, shell execution or implicit permissions.

The confidence field is an evidence-strength indicator, **not** a calibrated
probability that a conclusion is true. Evidence reliability is caller-supplied
and must never be invented by the planner.

## Example

    from reasoning import Evidence, Planner, Task

    plan = Planner.build([
        Task("inspect", "Inspect repository"),
        Task("change", "Implement change", depends_on=("inspect",)),
    ])
    plan.mark_in_progress("inspect")
    report = plan.submit_outcome(
        "inspect", "Reviewed target files",
        [Evidence("Files inspected", "repository review", True, 0.9)],
    )
    assert report.verified

## Run tests

    python -m unittest reasoning.test_planner -v

## Integration boundary

A model-facing adapter may translate model-proposed plans into Task objects,
request approval before external or destructive actions, and feed actual
tool/test results back as Evidence. Model-generated text alone is never proof
that a task succeeded. The native model remains responsible for language
understanding and proposing candidate plans; this package validates plan
structure and records execution evidence.
