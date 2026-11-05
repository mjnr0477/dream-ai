# DREAM AI Security Review — Session 7

Date: 2026-11-05
Repository: mjnr0477/dream-ai
Review basis: current main at branch creation (main ref; see PR base SHA for exact commit), plus direct source inspection of server.js, lib/dream-engine.js, lib/research.js, package.json, existing security tests and the Session 8 security review.

## Executive summary

The application has meaningful request-boundary protections, but it is not yet an authenticated multi-user service. Session 7 independently found that the per-client concurrency helper introduced by PR #167 was not wired into the public HTTP routes. PR #174 corrects this integration gap and adds a live HTTP regression test.

## Existing controls confirmed

- JSON-only request validation, request body limits, message/history limits and role/mode allowlists.
- Origin and Fetch Metadata checks, baseline security headers and a restrictive static-page CSP.
- A bounded process-local per-IP token bucket shared by chat and research.
- Global concurrency caps and minimized public health/capability diagnostics.
- HTTPS-only remote model transport, redirect rejection, embedded-credential rejection and bounded model request timeouts.
- Native runtime bearer-token support and constant-time token comparison are documented in the existing security review.
- Tool execution and external actions remain disabled by default; no general-purpose shell/code-execution route or file-upload route was found in the inspected public Node API.

## Findings

| ID | Severity | Finding | Status |
|---|---|---|---|
| S7-01 | High | PR #167's ClientConcurrencyLimiter helper was unit-tested but not imported or invoked by server.js; real chat/research requests were not receiving the advertised per-client concurrency control. | Fixed in PR #174; awaiting CI and review |
| S7-07 | Medium | The chat route returned unexpected runtime exception messages directly to clients, potentially disclosing deployment/configuration details. | Fixed in PR #174 with generic public error and regression test |
| S7-02 | High | No authenticated user/account system or server-side conversation store was found in the inspected application. Per-IP limits do not establish user identity or data ownership. | Open architectural requirement |
| S7-03 | Medium | The per-IP rate limiter is process-local and is not a shared quota across multiple instances. | Documented limitation |
| S7-04 | Medium | Native runtime authentication depends on deployment configuration; remote runtime deployment must set a strong token and restrict network exposure. | Deployment verification required |
| S7-05 | Medium | Prompt/system instructions are not an authorization boundary. Any future tool integration must enforce permissions, argument validation and confirmations outside the model. | Keep tool execution disabled until enforced |
| S7-06 | Informational | Conversation history is browser-local; users sharing a browser profile can access that local history. | Privacy disclosure / future account design |

## PR #174 changes

- Invoke the existing client concurrency limiter for both POST /api/chat and POST /api/research.
- Acquire the slot before the route handler consumes rate-limit tokens or performs downstream work.
- Return HTTP 429 with Retry-After when the same known client already has an active operation.
- Release the slot in finally for normal completion, validation errors, rate-limit exits and handler failures.
- Preserve global caps, per-IP rate limits, origin checks and loopback development behavior.
- Add a live HTTP test using a local mock native runtime. It checks same-client chat/research exclusion, independent-client progress and slot recovery.
- Replace unexpected chat runtime exception details with a generic public 502 response; add a route-level regression test using an intentionally invalid runtime URL.
- Add API input regression tests for invalid message roles/shapes and request bodies exceeding the 32 KiB limit.
- Ensure oversized request bodies are drained without buffering further data, so the server can return HTTP 413 instead of destroying the request stream.

## Verification still required

- Full GitHub Actions CI: Node/application tests, reasoning tests and native-model tests.
- Confirm the integration test passes on CI and inspect any failure logs.
- Re-check the final PR head and main base before merge.
- Do not merge if required checks are missing, pending or failing.

## Scope limitations

This is a source-based security review, not a live penetration test. It does not certify account isolation, production secret configuration, upload security, or prompt-injection immunity. The public API currently has no authenticated account-owned resources to test for cross-user IDOR. No production service or secret was changed.
