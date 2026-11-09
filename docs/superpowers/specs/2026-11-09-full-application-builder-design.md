# DREAM AI Full Application Builder — Design

## Intent
Turn DREAM AI's existing validated multi-file code responses into a clear project-generation workflow. A user should be able to ask for a complete application, receive a coherent project bundle, edit individual files, and download the whole project. The scope spans web, mobile, desktop, backend/API, command-line, games, browser extensions, bots and embedded/IoT software.

## Goals
- Detect application-building requests even when the user selected General chat rather than Coding mode.
- Classify the requested platform and give the model specific, practical build instructions.
- Produce complete, consistent project files with manifests, setup instructions, configuration examples and tests.
- Validate generated paths and size limits using the existing code-artifact parser.
- Offer a one-click ZIP download of the validated file bundle.
- Report build/test/deployment status honestly.

## Non-goals and boundaries
- Do not execute generated code on the DREAM web server.
- Do not claim a project compiles, passes tests, or is deployed unless a suitable isolated tool actually ran it.
- Do not automatically access private repositories, accounts, cloud resources, keys or paid services.
- Do not add DREAM user registration, profiles, subscriptions or billing.
- Do not bypass platform review, signing, licensing, API scopes or developer program rules.

## Approaches considered
1. Prompt-only improvement: minimal changes, but requests in General mode would still not reliably return project bundles and users would only have individual file downloads.
2. Dedicated app-request guidance plus validated file bundles and browser ZIP export: reuses DREAM's current chat/artifact infrastructure, supports multiple app families, and avoids unsafe server-side execution. Recommended.
3. Full autonomous build/test/deploy sandbox: the strongest eventual experience, but requires isolated ephemeral compute, strict CPU/memory/network limits, artifact storage, job lifecycle, consent and provider credentials. Those services are not currently available in this app, so this is a later phase.

## Product behavior
- Keep the user's requested framework and repository conventions when supplied.
- When stack is unspecified, choose a mainstream stack appropriate to the requested target and briefly state the choice.
- Ask only for missing details that materially alter the app. Otherwise proceed with sensible defaults.
- Generate a multi-file bundle with a README, manifest/build configuration, .env.example when relevant, tests and run/build instructions.
- Use DREAM's existing DREAM FILE artifact format and validation limits.
- Expose a ZIP download for bundles. Revalidate paths, duplicates and size limits in the browser before creating the archive.
- Distinguish generated files from real build/test results. State platform prerequisites such as Xcode/macOS for iOS builds when relevant.

## Architecture calls
- Add a pure request-classification and prompt-guidance module in lib/app-builder.js.
- Add dynamic application-building guidance to buildSystemPrompt, independent of chat mode.
- Parse artifacts for explicit app-building requests in server.js, while preserving Coding mode behavior.
- Add a small dependency-free ZIP writer in public/project-archive.js; the browser downloads a ZIP, without executing files.
- Extend capability metadata to disclose project-generation scope and the current missing sandbox boundary.
- Add tests for classification, prompt constraints, archive format, and malicious/invalid paths.
