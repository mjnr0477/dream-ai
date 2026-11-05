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
