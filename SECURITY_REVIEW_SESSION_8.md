# DREAM AI Security Review — Session 8

Date: 2026-11-05
Scope: read-only review of the `main` branch application and native model HTTP server.
Branch: `session-8/security-review-2026-11-05`

## Executive summary

The repository already has useful defensive controls: bounded JSON body sizes, message/history limits, a global concurrency cap in the Node app, request validation in the native runtime, HTTPS enforcement for remote model URLs, redirect rejection, bearer-token authentication for the native inference endpoint when a token is configured, and a fail-closed policy for future external actions.

The highest-priority remaining risk is the public Node chat endpoint. This branch adds a bounded in-memory per-IP token bucket (12 requests/minute shared by chat and research) using the first Render X-Forwarded-For address, but there is still no user authentication or per-user quota. The process-wide two-request concurrency limit can still be saturated by multiple IPs. The endpoint also requires JSON and checks browser request provenance, including the exact origin scheme and host, closing the identified cross-site request-triggering path. These are partial mitigations, not a complete abuse boundary.

No evidence was found in the inspected code that user chat is persisted server-side or that code execution/tools are enabled. Conversation history is browser-local storage, so this is not currently a multi-account data-isolation implementation. Do not claim account isolation until accounts and server-side ownership checks exist and are tested.

## Findings

| ID | Severity | Area | Finding | Recommended action |
|---|---|---|---|---|
| S8-01 | High | `POST /api/chat` | No user authentication or per-user quota. This branch adds a process-local 12/minute per-IP token bucket shared by chat and research, but multiple IPs can still exhaust the global two-chat limit. | Keep the per-IP limiter; before broad public launch add authenticated per-user quotas or a trusted edge challenge, monitoring, and a shared limiter if scaling beyond one instance. On Render, use only the first X-Forwarded-For address as documented by Render; do not trust arbitrary later entries. |
| S8-02 | High | `POST /api/chat` | The handler now requires JSON, rejects cross-site Fetch Metadata and compares the full Origin (scheme, host and empty path) against the configured public origin or an HTTPS-by-default host-derived origin. | Implemented in this branch with regression tests for missing/non-JSON content type, hostile origins and same-host HTTP-to-HTTPS scheme mismatch. Keep DREAM_PUBLIC_ORIGIN configured if the public origin differs from the service host. |
| S8-03 | Medium | Node HTTP responses | Security headers are inconsistent. JSON responses set `nosniff`; static responses add `nosniff` and `Referrer-Policy`; neither path consistently sets frame protection, a restrictive permissions policy, or a content security policy. | Add a centralized header policy. Introduce CSP only after verifying the current HTML/CSS/JS does not depend on inline scripts/styles or unsafe eval. |
| S8-04 | Medium | `GET /api/health`, `GET /api/capabilities`, native `GET /health` | Public status endpoints disclose runtime readiness and model metadata (including architecture, device, parameter count, training steps and checkpoint hash). This is not a credential leak, but it provides reconnaissance information. | Keep only operationally necessary fields public; consider a minimal public health response and protect detailed diagnostics behind operator authentication. |
| S8-05 | Medium | Native runtime | Native inference is authenticated only when a token is configured. The runtime correctly refuses non-loopback binding with a token shorter than 24 characters, but a loopback-bound process can intentionally run without one. | Preserve loopback-only unauthenticated development. Require a strong token and HTTPS/network restriction for every remote deployment; add a startup/deployment test for remote bind configurations. |
| S8-06 | Medium | `lib/dream-engine.js` | The system safety policy is useful guidance but is not a security boundary against prompt injection. There is no tool execution currently, so this is not presently an arbitrary-code-execution path. | Keep tools disabled by default. When tools are introduced, enforce permissions in application code, isolate untrusted retrieved content, validate tool arguments, and require confirmation for sensitive actions. Never rely on model text alone for authorization. |
| S8-07 | Informational | Frontend storage | Conversation history is stored in browser `localStorage`, not in authenticated server-side accounts. Anyone with access to the same browser profile can inspect it; clearing browser storage removes it. | Make this explicit in the privacy UI. If server-side accounts are added, enforce owner checks on every conversation read/write/delete and test cross-account IDOR. |

## Attack-surface checks performed

- Chat request validation: mode allowlist, non-empty message array, last-message role, role allowlist, per-message character limit, history truncation and 32 KiB request-body cap are present.
- Application load shedding: a process-wide two-active-chat limit returns 503 rather than queuing unlimited work. This does not provide fairness between callers.
- Native inference: request body, message count, message size, generation settings and model name are validated; a non-blocking model lock rejects excess inference.
- Native authentication: bearer token is compared using constant-time comparison when configured; remote binding requires a token of at least 24 characters.
- Model transport: remote model URLs require HTTPS; embedded URL credentials are rejected; redirects are rejected; request timeout is bounded.
- Output rendering: frontend assistant/user text is inserted with text nodes or `textContent`; code examples are not executed as HTML.
- Tool permissions: repository documentation and capability contract state tool execution is disabled by default and external actions require approval. No general-purpose shell or arbitrary code execution route was identified in the inspected application files.
- Data isolation: no server-side account or conversation storage was identified in the inspected code. Cross-account isolation therefore cannot yet be meaningfully certified.

## Required regression tests before public launch

1. Reject `text/plain`, missing content type and malformed content type on `POST /api/chat`.
2. Reject cross-site `Sec-Fetch-Site: cross-site` and hostile `Origin`; allow same-origin requests and documented non-browser clients.
3. Verify rate limits are per trusted client identity and cannot be bypassed by supplying a forged forwarding header.
4. Verify one caller cannot monopolize all inference slots indefinitely; include timeout and recovery tests.
5. Verify every response path has the intended security headers, including 404, 405, 413 and unexpected errors.
6. Verify runtime startup rejects remote bind without a sufficiently strong token.
7. When accounts exist, test horizontal and vertical authorization: user A cannot read, modify, export or delete user B's resources by changing IDs.
8. When tools exist, test prompt injection in user messages, retrieved pages and tool outputs; confirm authorization is enforced outside the model and unapproved actions never execute.

## Release recommendation

Do not describe DREAM AI as having completed account isolation or robust public API abuse protection yet. Keep arbitrary tool execution disabled. Address S8-01 and S8-02 before exposing an expensive or long-running model endpoint to unrestricted public traffic. This review is based on source inspection; it is not a live penetration test, and no production service was attacked or modified.


## Additional controls added in the rate-limit branch

- Chat and public research share a per-IP token bucket with a 12-request burst and 12-request/minute refill; rejected requests receive HTTP 429 and a Retry-After header.
- Render's first X-Forwarded-For address is used as the client IP; invalid/missing values fall back to the socket address. The in-memory map is capped at 5,000 buckets and prunes idle entries when full.
- Origin checks compare scheme as well as host. Non-loopback hosts default to HTTPS because Render terminates TLS before forwarding HTTP to the app; local loopback development defaults to HTTP. DREAM_PUBLIC_ORIGIN can explicitly override the expected public origin.
- This limiter is process-local. It is appropriate only for the current single-instance free service; use a shared store and authenticated user quotas if the service scales.
