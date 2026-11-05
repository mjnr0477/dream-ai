# DREAM reasoning and planning primitives

This package provides a deterministic orchestration layer for multi-step tasks.
It does not replace the native transformer, generate model reasoning, execute
commands, browse the network or perform external actions.

## Design

- Explicit task IDs, descriptions, success criteria and dependencies.
- Dependency validation with cycle detection before execution.
- Strict lifecycle: blocked → ready → in progress → completed or needs review.
- Evidence records include a claim, source, support/contradiction flag,
  caller-assigned reliability and optional success-criterion identifier.
- Contradictions, missing criterion evidence and unresolved uncertainty prevent
  automatic completion.
- Retry history is retained while stale outcome/evidence is cleared.
- No tools, network access, shell execution or implicit permissions.

The confidence field is an evidence-strength indicator, **not** a calibrated
probability that a conclusion is true. Evidence reliability is caller-supplied
and must never be invented by the planner.

## Model-output adapter

Use `plan_from_payload` to validate a model-proposed JSON object before the
planner accepts it. Unknown fields are rejected; task count and text lengths
are bounded; dependency references and cycles are checked by `Planner.build`.

    from reasoning import plan_from_payload, plan_snapshot

    plan = plan_from_payload({
        "tasks": [
            {
                "id": "inspect",
                "title": "Inspect repository",
                "success_criteria": ["target files reviewed"],
            },
            {
                "id": "change",
                "title": "Implement change",
                "depends_on": ["inspect"],
            },
        ]
    })
    print(plan_snapshot(plan))

This adapter validates structure only. It does not authorize or execute a tool.
External, destructive or financial actions must go through the separate
permission-controlled tool-execution layer and require its approval flow.

## Evidence and completion

A task with success criteria can be marked complete only when each criterion
has at least one supporting evidence record tagged with that exact criterion.
Contradictory evidence or unresolved uncertainty keeps the task in review.
Model-generated prose by itself is not evidence of execution or success.

## Run tests

    python -m unittest discover -s reasoning -p 'test_*.py' -v

## Runtime integration boundary

The Node application and native model runtime are separate from this Python
package. A future integration should call this adapter from a controlled
Python runtime boundary or port the same validated contract to the runtime
language; it must not spawn arbitrary shell commands or trust model JSON
without validation. The native model proposes candidate plans, while the
planner records transitions and checks actual test/tool observations supplied
as evidence.
