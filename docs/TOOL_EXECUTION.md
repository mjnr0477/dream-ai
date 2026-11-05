# DREAM AI tool execution boundary

This module is an application-side execution boundary, not an autonomous computer agent or a sandbox.

## Contract

- Only tools explicitly registered by trusted server code can execute.
- Every call requires an authenticated principal object supplied by a trusted server-side authentication layer and the tool's named permission.
- Tool inputs are validated against a deliberately small JSON-schema subset. Nested schemas are recursively checked, nesting is bounded, unknown object properties are rejected by default, and serialized input/output sizes are capped.
- Read tools need authorization. Write and destructive tools additionally require a one-time, expiring approval token issued by trusted server code after a real user confirmation. The token is bound to the exact tool ID, principal and input.
- Execution has per-tool timeouts, cooperative cancellation signals, input/output size limits and a global concurrency limit. A timed-out handler that ignores cancellation continues to occupy its concurrency slot until it actually settles; the registry does not pretend it has killed in-process code.
- Audit events record identifiers, outcome, duration and reason codes; raw input/output values and approval tokens are never logged.
- Handlers receive only the principal ID, cloned input, an AbortSignal and an execution ID.

## Critical integration requirements

Do not expose this registry directly to an unauthenticated HTTP route or trust a model-generated approval flag. The approval token must be issued by a server-side approval flow only after a real user confirmation, must never be sent to the model or exposed in chat, and the authenticated principal must come from verified session middleware. The model may propose a tool call; it cannot grant itself permissions or approvals.

Register handlers in trusted application startup code only. Do not load arbitrary handler code, accept user-supplied URLs as privileged destinations, or pass credentials to a handler unless its scope explicitly requires them. Network access, filesystem access and process execution are not sandboxed by this module: high-risk handlers must use separately isolated infrastructure with network-deny-by-default controls and a hard process/container kill boundary.

This first increment intentionally does not wire tools into chat or expose an execution endpoint. It establishes and tests the server-side boundary before any tool can be invoked by a model.
