# DREAM HOST

DREAM HOST is a lightweight Render-hosted control plane for GPU workers. It does not run a GPU or host Kaggle itself. A Kaggle notebook runs the worker agent, connects outbound to this stable HTTPS API, pulls inference jobs, runs Qwen3-1.7B plus the DREAM adapter on its GPU, and posts results back.

## Current release (0.2.0)

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
