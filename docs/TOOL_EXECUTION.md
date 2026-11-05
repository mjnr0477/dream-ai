# DREAM AI tool execution boundary

This module is an application-side execution boundary, not an autonomous computer agent or a sandbox.

## Contract

- Only tools explicitly registered by trusted server code can execute.
- Every call requires an authenticated principal object supplied by a trusted server-side authentication layer and the tool's named permission.
- Tool inputs are validated against a deliberately small JSON-schema subset. Unknown object properties are rejected by default.
- Read tools need authorization. Write and destructive tools additionally require a server-side approval verifier. The verifier receives the opaque approval token, principal ID, tool ID and SHA-256 digest of the exact canonical input. The default verifier denies all sensitive actions.
- Execution has per-tool timeouts, cancellation signals, input/output size limits and a global concurrency limit. Inputs are byte-limited before recursive schema validation. A handler that resolves during abort still returns a timeout result, and timed-out handlers retain their concurrency slot until they settle.
- Audit events record identifiers, outcome, duration and reason codes; raw input/output values are never logged.
- Handlers receive only the principal ID, cloned input, an AbortSignal and an execution ID.

## Critical integration requirements

Do not expose this registry directly to an unauthenticated HTTP route or trust a model-generated `confirmed` flag. The approval verifier must validate an opaque, server-issued approval token against a real user confirmation and bind it to the authenticated principal, tool ID and exact input digest. Do not use a client-provided boolean or a plain digest as proof of approval. The authenticated principal must come from verified session middleware. The model may propose a tool call; it cannot grant itself permissions or approvals.

Register handlers in trusted application startup code only. Do not load arbitrary handler code, accept user-supplied URLs as privileged destinations, or pass credentials to a handler unless its scope explicitly requires them. Network access, filesystem access and process execution are not sandboxed by this module: high-risk handlers must use separately isolated infrastructure with network-deny-by-default controls.

This first increment intentionally does not wire tools into chat or expose an execution endpoint. It establishes and tests the server-side boundary before any tool can be invoked by a model.

## Bounded multi-step agent orchestration

`lib/agent-orchestrator.js` adds a structured executor on top of the trusted registry. Plans contain at most 20 uniquely identified tasks, registered tool IDs, JSON inputs, explicit dependencies and bounded retry settings. Dependency cycles and missing references are rejected before any tool runs.

The orchestrator records per-task status, attempts, execution IDs, results and errors. It enforces a global step cap, checks interruption between tool calls, blocks dependants when prerequisites fail, and supports trusted input resolution from completed task results. Read-only calls may retry at most three times. Write and destructive calls are never automatically retried because a timeout or transport failure can leave side effects uncertain. They require an injected server-side approval provider; the provider must issue or validate a real user approval bound to the principal, tool and exact inputs. Tasks marked for verification require a trusted verifier to return `{ verified: true }` before they are marked complete.

This module is deliberately not connected to the public chat route. The current server does not yet provide verified user sessions, durable task storage, an approval-token issuer, or isolated execution workers. Do not create a public route around this module until those boundaries exist. Task state is returned to the caller and is not durable across process restarts. Tool handlers still need infrastructure-level isolation for untrusted code, network access, shell execution or privileged filesystem operations.
