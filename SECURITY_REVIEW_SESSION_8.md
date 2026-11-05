# DREAM AI Security Review — Session 8

Date: 2026-11-05
Base reviewed: current `main` at integration-branch creation
Scope: Node application request boundary, public diagnostics, browser response policy, native runtime configuration and future tool permissions.

## Changes implemented on this branch
- Validate `application/json` and browser request provenance for both `POST /api/chat` and `POST /api/research`.
- Reject cross-site Fetch Metadata and invalid/mismatched Origin values when supplied.
- Apply common security headers to JSON, static and error responses.
- Add a restrictive same-origin Content Security Policy. Inline style attributes alone are permitted because the existing composer adjusts textarea height through `element.style.height`; inline scripts and eval remain disallowed.
- Reduce public health and capability diagnostics by removing runtime status, device, parameter, training-step and checkpoint-hash details.
- Add regression tests for chat and research request validation, same-origin behavior, public diagnostic minimization and response headers.

## Important residual risks
1. **Public API abuse protection remains incomplete.** The chat and research routes have process-wide concurrency caps, not trusted per-user/per-IP rate limits or quotas. A caller may monopolize capacity. Do not treat these controls as sufficient for unrestricted public launch.
2. **Authentication and account isolation are not implemented.** Conversation history remains browser-local. There is no server-side account ownership model to certify.
3. **Prompt injection is not solved by system prompts.** No arbitrary code-execution route was identified in the reviewed app. Keep external tools disabled unless every action is independently authorized server-side, least-privilege scoped, logged and confirmed when consequential.
4. **Native runtime deployment must retain its existing constraints.** Remote runtime access must use HTTPS and a strong bearer token; do not expose an unauthenticated native listener.
5. **CSP requires browser verification.** Source inspection shows external same-origin scripts and Google Fonts; composer inline style attributes are intentionally permitted. A real browser smoke test remains necessary.

## Verification status
The repository CI workflow runs `npm test` on pull requests to `main`. This branch is based on current `main` so the PR can run the full existing Node and native-model CI pipeline. No test pass is claimed until the workflow completes successfully. This is a source-level security review, not a live penetration test.

## Release recommendation
Keep this work unmerged until the PR's full CI is green and an independent reviewer checks the diff. Even after CI, do not describe the service as having per-user rate limits, account isolation, or unrestricted tool execution. Production deployment remains a separate release decision.
