# DREAM AI — persistent GPU worker

This is the provider-independent replacement for the Kaggle notebook connector. It keeps the existing DREAM HOST, DREAM NOTEBOOK and Cloudflare R2 artifact vault. It does not create a new cloud store or require Railway, Kaggle, Hugging Face, HF tokens or a borrowed foundation model for native DREAM training.

## What runs where

- **DREAM HOST / NOTEBOOK:** remains the control plane and notebook UI on the existing Render service.
- **Persistent GPU worker:** runs on a Linux machine with an NVIDIA GPU and CUDA-enabled PyTorch. The systemd service reconnects automatically after a reboot or network interruption.
- **Cloudflare R2:** remains the durable source of prepared-corpus archives and native DREAM checkpoints. Every restore and upload is SHA-256/size verified by the notebook.
- **Native model:** `model/train.py` and `model/sft.py` train DREAM from random initialization. They do not load Qwen, an adapter, Transformers model weights or Hugging Face repositories.

**Important hardware boundary:** this repository and R2 cannot manufacture GPU compute. Render's current Free service is a CPU web service. The installer below must run on a persistent Linux host that already has an NVIDIA GPU and a working CUDA-enabled PyTorch installation. It is not a Kaggle notebook and it is not intended to run in Android Termux or on Render Free.

## Install on the persistent NVIDIA host

1. Install the NVIDIA driver and a CUDA-enabled PyTorch build that is compatible with that driver's version. Confirm `nvidia-smi` and `python3 -c 'import torch; print(torch.cuda.is_available())'` both work.
2. Check out the merged DREAM branch containing this setup. The installer uses `/opt/dream-ai` by default and refuses to overwrite an unrelated directory or a Git checkout with uncommitted changes.
3. Run:

```sh
sudo bash scripts/install-dream-notebook-worker.sh
```

The installer prompts without echoing for two distinct existing secrets from the `dream-host-live` Render environment:

- `DREAM_HOST_NOTEBOOK_WORKER_TOKEN`
- `DREAM_HOST_ARTIFACT_TOKEN` (entered into the worker as `DREAM_NOTEBOOK_ARTIFACT_TOKEN`)

It writes them to the root-only file `/etc/dream-ai/notebook-worker.env`, installs the systemd service and enables it at boot. It does not print the secrets. It does not change any Render environment variables.

Check the connection:

```sh
sudo systemctl status dream-notebook-worker
sudo journalctl -u dream-notebook-worker -f
```

The existing DREAM NOTEBOOK worker status should show the host as a connected CUDA device. If the worker is disconnected, inspect this service's journal before changing any Render configuration.

## Persistent data paths

The service creates these directories, owned by the dedicated `dreamai` account:

- `/var/lib/dream-ai/data/approved-records.jsonl` — source corpus only after real rights, privacy and quality review.
- `/var/lib/dream-ai/data/sft-train.jsonl` and `sft-validation.jsonl` — optional instruction-tuning splits after their review/audit gates.
- `/var/lib/dream-ai/checkpoints/` — local recovery copies.
- `/var/lib/dream-ai/logs/` — bounded training logs.

Copy approved source files to those paths through a secure file transfer. Do not put credentials or private user conversations into the training corpus. The notebook reads these paths from environment variables; it does not search Kaggle inputs or mount Kaggle datasets.

## Use the new notebook

Import `training/dream_native_persistent_notebook.ipynb` into the existing DREAM NOTEBOOK UI. It reuses the existing R2 artifact endpoints and artifact token. It restores the latest hash-verified prepared corpus and native checkpoints before starting. It backs up the exact prepared corpus and uploads checkpoints during and after training.

The training time budgets are configurable through the worker environment. The defaults are 1,500 seconds per training cell, below the 1,800-second notebook execution ceiling. A later run resumes from the last valid R2 checkpoint; it does not silently restart from random weights when a compatible checkpoint exists.

Training data must still pass the existing provenance, rights, privacy, quality, language-balance and human-review checks. R2 is storage, not a substitute for an approved corpus or a training GPU.

## Inference migration is gated separately

The native inference server already exists at `model/serve.py` and loads only a validated DREAM checkpoint. However, the current production chat path still identifies the legacy Qwen3 LoRA model. The previously inspected R2 adapter archive is not a native DREAM checkpoint and cannot be loaded by `model/serve.py`.

Do not change the live model settings yet. First complete native pretraining and instruction tuning, verify the native checkpoint, run the held-out and task-level evaluations, and explicitly approve the release. Only then should the production inference worker be switched to the native checkpoint and the legacy Qwen/Hugging Face path retired. This migration branch preserves the currently deployed service while that gate is outstanding.

## Operational honesty

A persistent worker means a machine that stays powered on and has continuous GPU access. A systemd restart policy makes the connector recover after process or machine restarts; it does not keep a rented GPU instance alive after its provider shuts it down or its credits expire. No free, always-on GPU capacity is included with Render Free or with the R2 bucket.
