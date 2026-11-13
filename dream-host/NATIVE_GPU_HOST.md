# DREAM AI — self-hosted native GPU host

This is the independent GPU path for DREAM's own transformer. It does not use Kaggle, Colab, Hugging Face inference, or a temporary notebook. The worker runs on an NVIDIA computer that we control and connects outbound to the existing DREAM HOST service.

## What becomes persistent

- Docker uses `restart: unless-stopped`, so the worker is restarted after a process/container failure and when Docker starts after a host reboot.
- The named Docker volume `dream-native-model-cache` stores the verified checkpoint downloaded from the private DREAM R2 artifact vault.
- The worker verifies the exact checkpoint SHA-256 before loading it; an altered or unapproved artifact is rejected.
- The worker registers with DREAM HOST, heartbeats, polls its job queue, and retries connections when HOST is unavailable.
- No public inbound port is exposed by the GPU worker. It only makes outbound HTTPS requests to DREAM HOST.

This gives persistent model-cache and process-restart behavior on the physical machine. It does **not** guarantee power, internet, hardware uptime, or free GPU access. The physical host must remain powered and connected. The existing free Render HOST service can sleep/restart; its chat queue and worker registry are currently in memory. The worker can re-register, but jobs already held only in that in-memory queue are not guaranteed to survive a HOST restart. This setup does not claim that queue is durable.

## Hardware and software required

- A 64-bit Linux computer with an NVIDIA CUDA-capable GPU and a driver compatible with the pinned CUDA 12.6 container.
- NVIDIA Container Toolkit configured for Docker.
- Docker Engine and Docker Compose v2 with support for the Compose `gpus: all` setting.
- Reliable power, internet, and persistent disk space for model/checkpoint and training artifacts.

A phone or Render's free Node service cannot substitute for a physical NVIDIA GPU. We have not obtained a physical host simply by adding these files.

## Prepare private credentials

On the GPU computer, clone the DREAM AI repository and use the deployed `dream-host-runtime` code. Copy `dream-host/native-gpu.env.example` to `dream-host/native-gpu.env`. Fill in the two different 32+ character credentials from the encrypted environment settings of `dream-host-live`:

- `DREAM_HOST_WORKER_TOKEN`: the dedicated worker token.
- `DREAM_HOST_ARTIFACT_TOKEN`: the separate artifact-vault token, not the raw Cloudflare R2 access key/secret.

Also set `DREAM_NATIVE_CHECKPOINT_SHA256` to the approved SHA-256 identity of the native instruction checkpoint already uploaded to R2. Do not put credentials in GitHub, paste them into chat, or commit `native-gpu.env`.

## Start and inspect

From the repository root:

```sh
bash scripts/start-native-gpu-host.sh
```

The launcher refuses to start if the machine has no visible NVIDIA GPU, Docker cannot pass the GPU into the container, either token is missing/reused, or the checkpoint hash is not pinned. To inspect after startup:

```sh
docker compose --project-directory dream-host -f dream-host/compose.native-gpu.yml ps
docker compose --project-directory dream-host -f dream-host/compose.native-gpu.yml logs -f dream-native-gpu-worker
```

To stop the worker without deleting its persistent checkpoint cache:

```sh
docker compose --project-directory dream-host -f dream-host/compose.native-gpu.yml down
```

Do not add `-v` to `down` unless you intentionally want to delete the named model-cache volume.

## Native checkpoint gate — required before the GPU worker can serve chat

The existing native worker deliberately refuses to load a random file, a pretraining-only checkpoint, or an unpinned "latest" artifact. It requires an audited native DREAM **instruction-tuned** checkpoint with positive `sft_training.completed_steps`, saved in the private R2 artifact vault as:

`dream-native-sft-checkpoint.pt`

It then downloads that exact artifact, verifies its SHA-256, and loads it with `model/runtime_checkpoint.py`. The current repository contains the training scripts and worker, but the current inspection did not find a prepared, human-approved native training corpus or a released native SFT checkpoint in the repository. A 97 MB Qwen LoRA adapter is not that checkpoint.

DREAM's existing training pipeline must be supplied with an approved corpus and separately reviewed SFT examples. Training must remain a candidate process: keep the same corpus/manifest for resume, preserve held-out test data, pass the audit and evaluation checks, and only then publish a checkpoint to R2 and configure its hash. Do not automatically learn from unreviewed model-generated text or private users' conversations.

This worker image is the durable execution side; it does not invent training data or create model weights when no approved corpus/checkpoint exists. Until both the host hardware and checkpoint are available, the current Cloudflare Workers AI route remains the independent app's managed fallback.
