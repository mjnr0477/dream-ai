# DREAM HOST

DREAM HOST is the first-party control plane for DREAM AI. It hosts the AI inference API and the DREAM NOTEBOOK interface while compute runs on separately connected, operator-controlled workers.

## Current release (0.3.0, backward-compatible)

- All existing 0.2.0 inference endpoints and chat worker behavior remain available.
- First-party notebook interface at `/notebook`.
- Python and Markdown cells, run-one/run-all, execution output, worker status, and standard Jupyter `.ipynb` import/export.
- Browser IndexedDB notebook storage; the free Render service's temporary filesystem is not used as permanent notebook storage.
- Remembered browser sign-in uses a 30-day HttpOnly, Secure, SameSite=Strict cookie; the raw owner key is not saved in JavaScript storage or notebook exports. Use “Forget access” on shared devices.
- Dataset tools download HTTPS files to the worker's temporary notebook workspace, scan /kaggle/input and the notebook dataset folder, and remove downloaded files.
- One-click import of DREAM's existing reviewed, resumable adapter-training workflow; training creates a candidate adapter and never overwrites the active adapter.
- Separate authenticated notebook job queue and a standalone Python notebook worker. It does not load the chat model into GPU memory.
- Notebook code and output sizes, pending queue, queued-job wait time, worker heartbeats, retries and execution windows are bounded.
- Notebook execution is owner-only and fails closed unless both distinct notebook owner and notebook worker tokens are configured.

DREAM HOST does not create GPU capacity. Notebook cells execute on an operator-owned worker connected to HOST. A CPU worker can run ordinary Python; GPU training requires compatible hardware and packages on that worker. The existing chat inference worker remains a separate compatibility path.

- OpenAI-compatible POST /v1/chat/completions and GET /v1/models.
- Authenticated worker registration, heartbeat, job claiming, result and failure reporting.
- Strict worker readiness: the connected worker must report the exact configured model and adapter.
- Separate liveness (`/health/live`) and readiness (`/health/ready`) endpoints.
- Queue depth, uptime, connected/ready/idle/busy worker counts, and an explicit in-memory persistence indicator.
- An integration test covering worker registration, mismatch rejection, job claim, result delivery and client completion.
- Bounded in-memory queue, request sizes and timeouts.
- No inbound tunnel, public Kaggle port, or Cloudflare Quick Tunnel required.

## Deploy on Render

Create a Node web service from the dream-host-runtime branch:

- Name: dream-host-live
- Build command: npm install --ignore-scripts
- Start command: node dream-host/server.js
- Region: Singapore
- Plan: Free

Set these environment variables on Render (never commit tokens):

- DREAM_HOST_WORKER_TOKEN: random secret, at least 32 characters.
- DREAM_HOST_CLIENT_TOKEN: a different random secret, at least 32 characters.
- DREAM_HOST_MODEL_NAME: Qwen/Qwen3-1.7B

The process binds to 0.0.0.0:$PORT and exposes GET /health.

## Connect the Kaggle GPU worker

In Kaggle Notebook -> Add-ons -> Secrets, add:

- HF_TOKEN: your Hugging Face read token (existing secret).
- DREAM_HOST_WORKER_TOKEN: the worker token from Render.

Set the following notebook environment values before running worker.py:

- DREAM_HOST_URL: the Render base URL, e.g. https://dream-host-live.onrender.com (no /v1 suffix).
- DREAM_HOST_WORKER_TOKEN: read it from the Kaggle secret named above.
- HF_TOKEN: read it from the existing Kaggle secret.

The worker loads Qwen/Qwen3-1.7B and msjnr0477/dream-ai-adapter onto the available CUDA GPU, registers itself, sends heartbeats every 15 seconds, polls for jobs, and reconnects after network errors.

From a Kaggle notebook cell, after the Render service is deployed:

```python
from kaggle_secrets import UserSecretsClient
import os

secrets_client = UserSecretsClient()
os.environ["HF_TOKEN"] = secrets_client.get_secret("HF_TOKEN")
os.environ["DREAM_HOST_WORKER_TOKEN"] = secrets_client.get_secret("DREAM_HOST_WORKER_TOKEN")
os.environ["DREAM_HOST_URL"] = "https://dream-host-live.onrender.com"

!curl -fsSL https://raw.githubusercontent.com/mjnr0477/dream-ai/dream-host-runtime/dream-host/worker.py -o /kaggle/working/dream_host_worker.py
!python /kaggle/working/dream_host_worker.py
```

The earlier DREAM AI Kaggle notebook already used torch, transformers and peft. Avoid reinstalling or upgrading torch unnecessarily; install only a package that is actually missing. If the previous tunnel-based inference server is still running, restart the Kaggle session first because this worker loads its own model copy. Keep the notebook session running while the worker is needed.

## DREAM NOTEBOOK

Open `https://dream-host-live.onrender.com/notebook` after the 0.3.0 host release is deployed.

### One-time HOST configuration

Add both of these private environment variables to the DREAM HOST Render service:

- `DREAM_HOST_NOTEBOOK_TOKEN`: a new random secret used by the notebook owner in the browser.
- `DREAM_HOST_NOTEBOOK_WORKER_TOKEN`: a separate random secret used only by the notebook Python worker.

Each value must be at least 32 characters and distinct from the other and from the existing chat `DREAM_HOST_WORKER_TOKEN` and `DREAM_HOST_CLIENT_TOKEN`. Do not place any token in the frontend source, a notebook cell, a public repository, or a notebook export.

### Start an operator-owned notebook worker

On a machine dedicated to the notebook owner, install the notebook worker requirements and a compatible Python environment. Install PyTorch separately when GPU/accelerator training is needed, using a build compatible with the machine's driver and accelerator.

Set these environment variables privately:

- `DREAM_HOST_URL=https://dream-host-live.onrender.com`
- `DREAM_HOST_NOTEBOOK_WORKER_TOKEN=<the dedicated notebook worker token>`
- `DREAM_HOST_NOTEBOOK_TRUSTED_MODE=1`
- Optional for DREAM adapter training: `DREAM_NOTEBOOK_HF_TOKEN=<your Hugging Face token>`

To populate the optional training token from a Kaggle Secret named `HF_TOKEN`, run this before starting the worker:

```python
from kaggle_secrets import UserSecretsClient
import os

os.environ["DREAM_NOTEBOOK_HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

The worker passes only this explicitly named training token to its trusted notebook child processes. It does not pass the DREAM HOST worker token or other host credentials. Notebook code can access this optional token, so run only notebooks and cells you trust.

Then install and run:

```bash
python -m pip install -r dream-host/requirements-notebook-worker.txt
python dream-host/notebook_worker.py
```

The worker registers as a notebook executor, maintains a separate Python kernel for each notebook, heartbeats while cells run, and returns stdout, stderr, expression results and tracebacks. It allows up to four active kernels by default and cleans kernels that have been idle for 30 minutes. The worker's Python environment is the environment available to notebook cells; packages installed there are available to subsequent cells.

### Persistent artifact vault (Cloudflare R2)

The production artifact vault stores candidate checkpoints and training reports in a private Cloudflare R2 bucket. It uses the R2 S3-compatible API; local filesystem storage is only a development fallback and is not durable on a free Render service.

Configure these variables on the `dream-host-live` Render service:

- `DREAM_R2_ACCOUNT_ID`: the Cloudflare Account ID.
- `DREAM_R2_ACCESS_KEY_ID`: the R2 S3 Access Key ID.
- `DREAM_R2_SECRET_ACCESS_KEY`: the matching R2 Secret Access Key.
- `DREAM_R2_BUCKET`: `dream-ai-model-vault`.
- `DREAM_HOST_ARTIFACT_TOKEN`: a new random private token of at least 32 characters, distinct from the chat and notebook tokens.
- Optional `DREAM_HOST_ARTIFACT_MAX_BYTES`: maximum artifact size in bytes; defaults to 536870912 (512 MiB).

Create the R2 API token with **Object Read & Write** permission and restrict it to the `dream-ai-model-vault` bucket. Keep that bucket private. The Access Key ID and Secret Access Key belong only in Render's encrypted environment settings; never put them in GitHub, a notebook cell, the Kaggle worker, frontend code, or chat.

For the optional Kaggle publish cell, create a Kaggle Secret named `DREAM_NOTEBOOK_ARTIFACT_TOKEN` with the same dedicated artifact token value as `DREAM_HOST_ARTIFACT_TOKEN`. The publisher talks only to the authenticated DREAM HOST API. R2 credentials never leave Render. The API stores each artifact under its SHA-256 digest, verifies integrity on download, and lists only validated metadata.

When configured, `/health` reports `artifactStorage: "r2"`. A missing or invalid R2 configuration does not silently claim persistent storage; the artifact API stays disabled unless R2 or an explicitly configured local development directory is available.

### Storage and security boundaries

- Notebook documents are saved in the current browser's IndexedDB. Export important work as `.ipynb`; imported Jupyter Notebook version 4 files are supported.
- The notebook owner key is exchanged once for a browser-managed HttpOnly session cookie. The default “Remember this device” duration is 30 days; “Forget access” clears it. A shared browser should not be remembered.
- HTTPS dataset downloads are saved under `datasets/` in the connected worker's per-notebook temporary workspace. The scanner also inspects private Kaggle attachments under `/kaggle/input`. Worker kernels and files are temporary and disappear when the worker restarts or its idle-kernel cleanup runs.
- HOST worker registrations, queues and active jobs are in memory and are cleared by a HOST restart. The browser keeps the notebook document, but kernel variables and worker-local files are temporary.
- Notebook cell execution runs trusted Python code. The worker is **not** a hardened multi-tenant sandbox. Start it only on a machine you control, do not expose its worker token, and do not use it for untrusted public notebooks.
- The notebook API requires both `DREAM_HOST_NOTEBOOK_TOKEN` and `DREAM_HOST_NOTEBOOK_WORKER_TOKEN`; if either is absent or reused, the notebook UI is still served but execution/status endpoints return 503. Existing chat service startup does not depend on these variables.
- A connected CPU can execute ordinary code. Long or large model-training runs require enough RAM/VRAM, disk, and a compatible accelerator. HOST itself does not supply those resources.

### DREAM AI training

The notebook is an execution environment, not automatic permission to train. Use the existing DREAM AI training scripts only with datasets that pass DREAM's consent, rights, provenance, privacy, quality and split-isolation reviews. Training should create a candidate checkpoint/adapter and evaluation report; it must not overwrite the active production model automatically.

## API

- GET /health — public liveness and aggregate status; does not expose secrets.
- GET /api/v1/status — client-token protected aggregate status.
- GET /v1/models — client-token protected model list.
- POST /v1/chat/completions — client-token protected OpenAI-compatible chat request; non-streaming requests only.
- POST /api/v1/workers/register — worker-token protected registration.
- POST /api/v1/workers/heartbeat — worker-token protected heartbeat.
- POST /api/v1/workers/jobs/claim — worker-token protected FIFO job claim.
- POST /api/v1/workers/jobs/:id/result — worker-token protected result delivery.
- POST /api/v1/workers/jobs/:id/fail — worker-token protected failure delivery.

All protected routes require Authorization: Bearer <token>.

## Important reliability limits

This is a control plane, not a new source of GPU compute. The free Render instance can sleep and restart, while Kaggle notebook sessions have runtime and quota limits. Version 0.2.0 still keeps its job queue and worker registry in process memory; a Render restart clears queued jobs and active leases. The worker reconnects when its notebook process is running again, but it cannot wake a stopped Kaggle session. A durable queue requires a dedicated persistent database or queue service; uninterrupted GPU availability requires a compute worker that does not depend on notebook session limits.

DREAM AI is currently configured to use the fixed `https://dream-host-live.onrender.com/v1` endpoint. The application can start while the worker is absent, but model inference will return an unavailable response until `/health/ready` reports `modelReady: true`. Do not describe the model as connected until an authenticated end-to-end completion has succeeded.


## Tests

Run the complete repository checks, including the DREAM HOST integration test, with `npm test`. The host test starts an isolated local server and verifies authentication, model/adapter readiness, queue claim, result delivery and the OpenAI-compatible response.
