# Full Application Builder Implementation Plan

> **For agentic workers:** Use the native implementation workflow in this session. Steps use checkbox syntax for tracking.

**Goal:** Let DREAM AI recognize app-building requests across chat modes, generate validated multi-file project bundles for major app families, and let users download bundles as ZIP files.

**Architecture:** Reuse the existing buildSystemPrompt, parseCodeArtifactsIfPresent, and browser artifact renderer. Add a pure app-request classifier, dynamic system guidance, and a dependency-free ZIP helper; do not add server-side code execution.

**Tech Stack:** Node.js 20+, JavaScript ES modules, built-in node:test, browser Blob, TextEncoder, DataView.

**Spec:** docs/superpowers/specs/2026-11-09-full-application-builder-design.md

## Global Constraints
- Preserve the existing guest-first experience and Plugins/Gmail implementation.
- Preserve the existing multi-file artifact delimiters and parser limits (50 files, 64 KiB/file, 256 KiB total).
- Do not execute model-generated code in the Node web process.
- Do not claim build/test/deployment success without real execution evidence.
- Do not add API keys, secrets, new paid dependencies, or user account features.

## Review Focus
- Ordinary requests asking what app development means must not be misclassified as a request to generate a project.
- Explicit requests in General mode must still produce validated artifacts.
- ZIP packaging must reject traversal paths, absolute paths, duplicate names and oversized bundles.
- Unicode filenames and file contents must survive the archive unchanged.
- ZIP output must remain within classic ZIP limits and never use Zip64 for this bounded format.

---

### Task 1: App request recognition and model guidance

**Files:**
- Create: lib/app-builder.js
- Create: lib/app-builder.test.js
- Modify: lib/dream-engine.js
- Modify: lib/dream-engine.test.js

**Interfaces:**
- isAppBuildRequest(query: string) -> boolean
- classifyAppTarget(query: string) -> string
- buildAppBuilderGuidance(query: string) -> string

- [ ] Write tests for app build detection, non-build questions, platform classification, and guidance requirements.
- [ ] Run node --test lib/app-builder.test.js and confirm the missing-module failure.
- [ ] Implement classifier and target-specific guidance.
- [ ] Inject guidance dynamically for app requests in any chat mode.
- [ ] Run app-builder and dream-engine tests.

### Task 2: Server-side artifact handoff and honest capability metadata

**Files:**
- Modify: server.js
- Modify: lib/capabilities.js
- Modify: lib/capabilities.test.js

**Interfaces:**
- App requests identified by isAppBuildRequest are eligible for parseCodeArtifactsIfPresent even if mode is General or Learning.
- Capability metadata declares project generation and archive download while keeping execution sandbox status unavailable.

- [ ] Add a regression test for the request-classification and capability contract.
- [ ] Keep existing Coding mode artifact extraction unchanged.
- [ ] Report generated project files separately from execution capability.

### Task 3: Safe browser ZIP export

**Files:**
- Create: public/project-archive.js
- Create: public/project-archive.test.js
- Modify: public/app.js

**Interfaces:**
- createProjectZip(artifacts, limits?) -> Blob

- [ ] Test ZIP signatures, CRC-32, UTF-8 names, file payloads, and rejection of malformed/unsafe bundles.
- [ ] Confirm tests fail before implementation.
- [ ] Implement bounded, uncompressed classic ZIP output with UTF-8 flags.
- [ ] Add a Download project ZIP action above the existing per-file edit/copy/download controls.

### Task 4: Documentation and verification

**Files:**
- Modify: README.md
- Modify: package.json

- [ ] Document supported target families, archive workflow, current execution limitations and platform prerequisites.
- [ ] Add syntax checks for new modules.
- [ ] Run the complete npm test suite and inspect the complete diff.
- [ ] Keep this feature in its own PR; do not deploy automatically.
