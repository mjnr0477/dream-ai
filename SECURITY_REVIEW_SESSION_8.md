# DREAM AI Security Review — Session 8

Date: 2026-11-05
Target reviewed: current `main` application, including chat and public research routes.
Implementation branch: `session-8-security-final-2026-11-05`

## Executive summary

The application has bounded JSON bodies, chat message/history limits, a process-wide concurrency cap, native-runtime identity checks, HTTPS enforcement for remote model URLs, redirect rejection, and bearer-token support for native inference when configured. Tool execution remains disabled by default.

This session adds request-provenance checks to both expensive public POST routes, consistent response security headers, a restrictive static-content CSP, and reduced public model diagnostics. These controls reduce browser-origin abuse and information exposure. They do not replace authentication, per-user quotas, edge rate limiting, or authorization.

## Findings and disposition

| ID | Severity | Finding | Disposition |
|---|---|---|---|
| S8-01 | High | Public chat lacks authentication and per-user/client quota. The global concurrency limit can be monopolized. | Open launch blocker. Configure trusted edge rate limits and application quotas after the production proxy/client identity model is established. Never trust arbitrary forwarded-IP headers. |
| S8-02 | High | Chat and research accepted POST requests without requiring JSON or checking browser request provenance. | Addressed in this branch: require JSON; reject cross-site Fetch Metadata; validate supplied Origin against request Host. Regression tests added. |
| S8-03 | Medium | Response security headers were inconsistent and static content lacked CSP. | Addressed in this branch: common security headers and CSP. Existing Google Fonts, same-origin scripts and textarea auto-resize are accounted for. Browser verification remains required. |
| S8-04 | Medium | Public status endpoints exposed model runtime details including training step count, checkpoint hash and hardware details. | Addressed in this branch: public health is coarse; capabilities omit training step and checkpoint hash. |
| S8-05 | Medium | Native inference authentication depends on deployment configuration. | Open deployment gate: remote runtime must use HTTPS, a strong token and network restrictions. |
| S8-06 | Medium | System prompts alone cannot enforce authorization against prompt injection. | Tools remain disabled. Any future tool must enforce permissions server-side, validate arguments, isolate untrusted content and require approval for consequential actions. |
| S8-07 | Informational | Conversation history is browser-local, not account-backed server storage. | No account isolation is claimed. Add ownership checks and cross-account authorization tests if server-side accounts are introduced. |

## Implementation details

- `POST /api/chat` and `POST /api/research` require `application/json`.
- Cross-site `Sec-Fetch-Site` values are rejected; supplied Origin must be a valid HTTP(S) origin matching the request Host.
- JSON, static, error and overload responses receive consistent security headers.
- Static responses receive a CSP restricting scripts to same-origin resources. Inline style attributes are allowed narrowly to preserve the existing textarea auto-resize behavior; no inline scripts are enabled.
- Public health no longer exposes runtime status strings, model name, training steps, checkpoint hash, device, parameters or tokenizer.
- Capabilities retain the public capability contract but omit training steps and checkpoint hash.
- Regression tests cover chat and research request validation, same-origin fallback behavior, diagnostic minimization, and security headers.

## Verification status

- Source review performed against current `main`, not just the older session branch.
- Regression tests are committed but must pass in GitHub Actions before merge.
- A real-browser CSP compatibility check and independent security review are still required.
- This is not a live penetration test. No production service was attacked or modified.

## Release gate

Do not merge/deploy until the CI test job passes and an independent reviewer checks the resulting diff. Keep tools disabled. Do not claim per-user rate limiting, account isolation, or unrestricted secure tool execution until those controls exist and are tested.
