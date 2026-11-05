# DREAM AI Security Review — Session 8

Date: 2026-11-05
Target: current main, including chat, public research and tool registry boundaries.

## Summary

This is a source-level review and remediation record, not a penetration test. The current service has no user-account authentication, no account isolation, and no unrestricted tool execution. The controls below reduce browser-origin abuse, anonymous request bursts and information exposure; they do not establish per-user identity or frontier-grade security.

## Findings and disposition

| ID | Severity | Finding | Disposition |
|---|---|---|---|
| S8-01 | High | Anonymous chat could be flooded and one caller could occupy global inference slots. | Partially mitigated: 20/minute per trusted client identity, one active request per identity, bounded in-memory storage and Retry-After. Still requires authenticated per-user quotas and shared edge controls before broad launch. |
| S8-02 | High | Chat/research accepted POST requests without strict JSON and browser-origin validation. | Addressed: JSON-only POST, same-origin Origin validation, and cross-site Fetch Metadata rejection. |
| S8-03 | Medium | Static pages and error responses had inconsistent security headers and no CSP. | Addressed: CSP, frame protection, MIME sniffing protection, referrer policy and permissions policy on static and API responses. Browser compatibility still needs real-device verification. |
| S8-04 | Medium | Public health/capability responses exposed checkpoint hashes, training counts and hardware details. | Addressed: public health exposes only configured/ready booleans; capability model metadata omits checkpoint and hardware diagnostics. |
| S8-05 | Medium | Remote native inference depends on deployment configuration. | Open deployment gate: require HTTPS, strong token, network restrictions and verified checkpoint identity. |
| S8-06 | Medium | System prompts cannot enforce tool authorization against prompt injection. | Tool registry is server-side and default-deny for write/destructive actions, but no authenticated approval service or tool endpoint is connected. |
| S8-07 | Informational | Conversation history is browser-local, not account-backed server storage. | No account isolation is claimed. Add ownership checks if server-side accounts are introduced. |

## API protections

- Chat and research POST requests require application/json.
- A supplied Origin must match the configured public origin or request origin; cross-site Fetch Metadata is rejected.
- Render client identity uses CF-Connecting-IP only with a valid Cloudflare CF-Ray; arbitrary X-Forwarded-For is ignored.
- The process-local quota is 20 requests/minute per client identity, with one active request per identity and bounded bucket storage.
- The quota is appropriate only for the current single-instance free service. Multi-instance deployment requires a shared store.
- Static responses use a restrictive CSP. Same-origin scripts are allowed; Google Fonts are explicitly allowed; inline style attributes remain allowed for textarea auto-resize; inline scripts are not enabled.
- Public health and capabilities no longer expose checkpoint hashes, training counts or hardware details.

## Remaining launch gates

- Add authenticated user sessions and per-user quotas.
- Add a shared edge/rate-limit layer before scaling beyond one instance.
- Verify the CSP in real browsers and on Android.
- Keep tool execution disabled until the app has authenticated principals and a real server-issued, short-lived, one-time approval service.
- Verify native runtime HTTPS, token rotation, checkpoint identity and network isolation.

No production deployment is included in this review. No claim of complete security or unrestricted access is made.
