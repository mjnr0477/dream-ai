# DREAM AI Security Review — Session 8

Date: 2026-11-05
Reviewed base: current `main` at `c5b2779b0db798b1609ced1653d740d4edf179ba`
Scope: Node application request boundary, rate limiting, public diagnostics, browser response policy, native runtime configuration, prompt injection and future tool permissions.

## Controls confirmed on current main
- Both chat and research require JSON and validate browser request provenance.
- Origin checks compare the complete origin (scheme and host); non-loopback hosts default to HTTPS and `DREAM_PUBLIC_ORIGIN` can set an explicit public origin.
- Shared bounded in-memory token bucket limits chat and research by the first valid `X-Forwarded-For` address, with a configurable capacity capped at 120 requests/minute and a bounded bucket map.
- Common security headers and restrictive CSP are applied; inline style attributes are allowed for the existing textarea auto-resize behavior, while scripts remain same-origin.
- Public health/capabilities responses omit detailed checkpoint and hardware diagnostics.
- Native model transport enforces HTTPS for remote URLs, rejects redirects and embedded credentials, and supports bearer-token authentication.
- Tool execution and external actions remain disabled/not configured.

## Session 8 regression coverage added in this PR
- Reject missing/non-JSON chat and research requests.
- Reject hostile Origin and cross-site Fetch Metadata.
- Reject an origin that matches the host but uses the wrong scheme.
- Verify chat and research share the per-IP token bucket and that the first forwarded address is the rate-limit identity.
- Verify API security headers and static-page CSP.

## Residual risks
1. The in-memory rate limiter is per process, not a distributed quota. It is suitable only for a single instance and does not provide authenticated per-user quotas. The first forwarded address is trusted under the Render deployment assumption; this must be revisited if the proxy topology changes.
2. The global concurrency caps can still cause contention across distinct client IPs.
3. Authentication and server-side account isolation are not implemented. Conversation history remains browser-local.
4. System prompts are not a security boundary. Any future tool must enforce authorization in application code, validate arguments, isolate untrusted content, use least privilege, log actions, and require confirmation for consequential operations.
5. CSP and the live browser experience still require a real browser smoke test. This source review is not a penetration test.

## Verification
The PR triggers the repository CI workflow, which runs `npm test` and native-model checks. No test pass is claimed until CI completes successfully. No production deployment is included.

## Release recommendation
Do not merge until all required CI jobs pass and an independent reviewer checks the change. Do not claim account isolation, authenticated per-user quotas, or unrestricted tool execution.
