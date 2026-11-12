# DREAM NOTEBOOK implementation plan

> **For implementation:** Execute each task in order. Write tests first, observe the expected failure, implement the smallest change, and run the complete relevant test suite.

**Goal:** Add a first-party, owner-authenticated notebook UI to DREAM HOST, with browser-side notebook persistence and asynchronous Python cell execution through a separate trusted worker.

**Architecture:** Extend the existing DREAM HOST Node control plane with a separate notebook job queue and token-protected routes. Serve static notebook assets from HOST. The notebook worker is a separate Python process and does not load the chat model. Browser IndexedDB stores notebook documents; the HOST's in-memory state stores only active jobs and workers.

**Constraints:** Preserve all existing chat behavior. Notebook API must fail closed when `DREAM_HOST_NOTEBOOK_TOKEN` is not configured. The worker is for a single trusted operator, not public multi-tenant execution. Worker files and kernels are temporary. GPU training depends on the hardware attached to the worker.

## Task 1 — Queue behavior tests

**Files:** `dream-host/notebook.test.js`, `dream-host/notebook-core.js`

- Test that jobs are validated and queued with notebook ID, cell ID and code.
- Test that only a registered notebook worker can claim a job.
- Test that a worker can complete only its own job.
- Test that stale workers release/requeue or fail their active job within bounded attempts.
- Test queue capacity, status snapshots and output-size validation.
- Verify the new tests fail before implementation.

## Task 2 — Notebook queue implementation

**File:** `dream-host/notebook-core.js`

- Implement an in-memory queue isolated from chat jobs/workers.
- Implement worker register, heartbeat, claim, complete/fail, job lookup and status.
- Validate notebook/cell/job/worker IDs and limit code/output sizes.
- Use bounded job lifetime, retry count and terminal-job cleanup.

## Task 3 — HOST HTTP integration and static serving

**File:** `dream-host/server.js`

- Preserve current health, readiness, OpenAI-compatible chat and chat worker routes.
- Serve `/notebook`, `/notebook.js` and `/notebook.css` with correct MIME and no-cache headers.
- Add notebook client routes guarded by `DREAM_HOST_NOTEBOOK_TOKEN`.
- Add notebook worker routes guarded by the existing worker token.
- When notebook token is missing, return a clear 503 configuration response instead of starting an unauthenticated executor.

## Task 4 — Operator-owned Python kernel worker

**File:** `dream-host/notebook_worker.py`

- Register as a notebook worker, heartbeat, claim cell jobs and report results.
- Maintain an isolated per-notebook Python interpreter subprocess with separate namespaces and working directories.
- Capture standard output, standard error, last-expression display and tracebacks.
- Enforce execution and output bounds; strip HOST worker credentials from the child process environment.
- Require an explicit trusted-worker setting and document that this is not a multi-tenant OS sandbox.
- Detect CUDA/CPU availability without reserving GPU memory for the chat model.

## Task 5 — Notebook frontend

**Files:** `dream-host/notebook.html`, `dream-host/notebook.js`, `dream-host/notebook.css`

- Mobile-first code/Markdown cell editor, run/stop state, run-all, insert/delete, worker status and output panels.
- Persist notebook content in IndexedDB.
- Import/export valid Jupyter `.ipynb`.
- Store the access token only in session storage and never print it or embed it in generated HTML.
- Use textContent for execution output; do not inject user code or worker output as HTML.

## Task 6 — Documentation and CI

**Files:** `dream-host/README.md`, `.github/workflows/dream-host-tests.yml`

- Document the one-time notebook token, worker startup, compatible dependencies, CPU/GPU limits, trusted-code boundary, local persistence and export workflow.
- Add Python syntax validation for the notebook worker and keep the existing HOST integration test active.

## Task 7 — Verification and handoff

- Run Node syntax checks, Node tests, Python compilation and the full relevant CI suite.
- Inspect the final diff and confirm the original chat routes and worker code remain unchanged except for routing compatibility.
- Open a PR to `dream-host-runtime`; do not claim the live Render deployment changed until a deployment is independently verified.
