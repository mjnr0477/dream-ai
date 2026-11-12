# DREAM NOTEBOOK platform design

Date: 2026-11-12

## Intent

Build a first-party notebook experience served by DREAM HOST so DREAM AI development and training workflows no longer depend on a Kaggle or Hugging Face notebook UI. The notebook is an operator tool for editing and executing Python cells against a connected DREAM worker.

## Goals

- Serve a mobile-friendly notebook at `/notebook` from the existing DREAM HOST service.
- Support Python and Markdown cells, sequential cell execution, captured stdout/stderr/result, run-all, cell insertion/deletion, and live worker status.
- Import and export standard Jupyter `.ipynb` files.
- Save notebook documents in browser IndexedDB so Render's ephemeral filesystem is not misrepresented as durable storage.
- Dispatch cell jobs through a bounded, authenticated DREAM HOST queue to a separate notebook worker.
- Keep notebook execution separate from the existing chat inference worker and chat request queue.
- Keep DREAM AI's existing data-rights, privacy, human-review, held-out evaluation and candidate-promotion safeguards.

## Non-goals

- Public, multi-tenant code execution.
- New DREAM user accounts, sign-in, billing or subscriptions.
- Claiming that DREAM HOST or DREAM COMPUTE provides free GPU hardware.
- Automatically training on conversations or unreviewed public data.
- Automatically replacing the active adapter or production checkpoint.
- Permanent cloud notebook-file or checkpoint storage in this first increment.

## Approaches considered

1. **Browser-only Python (Pyodide).** Lowest server risk and simplest hosting, but runs in the browser, cannot use the DREAM GPU worker, and does not provide the intended hosted training path. Not selected.
2. **Jupyter Server directly inside the Render web service.** Familiar notebook kernel, but it mixes arbitrary code execution with the public app process, consumes the web service's limited memory/CPU, and Render's free filesystem is not durable. Not selected.
3. **DREAM HOST notebook UI plus a separately authenticated worker.** HOST serves the editor and API; an operator-owned worker executes cells in per-notebook Python kernels and reports results. Browser IndexedDB keeps notebook documents; `.ipynb` import/export provides portability. This reuses the current host deployment and keeps execution off the web process. **Selected** because it matches the existing HOST/worker architecture and preserves the chat path.

## Architecture and behavior

- `GET /notebook` serves the UI; the existing health endpoint behavior remains unchanged.
- The browser stores notebook content in IndexedDB and keeps the notebook access token only for the active browser session.
- The operator configures `DREAM_HOST_NOTEBOOK_TOKEN` and `DREAM_HOST_NOTEBOOK_WORKER_TOKEN` (each at least 32 characters and distinct from each other and the chat tokens). Notebook client API calls require the former; notebook worker routes require the latter. If either is absent or reused, notebook API calls fail closed with a setup error and the rest of DREAM HOST still starts normally.
- The independent notebook worker authenticates with the dedicated `DREAM_HOST_NOTEBOOK_WORKER_TOKEN` and registers a notebook capability. It does not load the chat model into GPU memory. Notebook code can import the packages installed on that worker.
- Notebook jobs are asynchronous: submit, claim, execute, complete/fail, and poll status. Queue size, cell size, output size, execution duration and worker heartbeat are bounded.
- The worker is an operator-owned trusted-code executor, not a hardened multi-tenant sandbox. It must only be run on a machine dedicated to the operator's own notebooks. Do not expose it as a public code-execution service.
- HOST job/worker state is in memory and is lost on service restart. Notebook documents are stored in the user's browser and can be exported. Kernel variables and files on a worker are temporary and are lost when that worker/kernel stops.
- A connected CPU worker can execute general Python. GPU training requires a worker machine with a compatible GPU and training libraries; HOST itself does not create GPU capacity.

## DREAM AI training boundary

The notebook can run DREAM's existing native training and adapter-candidate workflows when their reviewed datasets, project files and compatible dependencies are available on the worker. Data must pass the existing consent, source-rights, privacy, quality, provenance and held-out split checks. A completed training run produces a candidate and evaluation evidence only; it does not silently promote or replace the live model.

## Compatibility and change control

- Preserve existing `/v1/chat/completions`, `/v1/models`, worker registration, health and readiness semantics.
- Do not require notebook environment variables for the current chat service to start.
- Keep notebook worker identity and queue separate from chat workers/jobs.
- Do not change Dream Education, Dream Wi-Fi, VYBORA, the DREAM AI inference provider, or the DREAM COMPUTE packed deployment bundle in this increment.
