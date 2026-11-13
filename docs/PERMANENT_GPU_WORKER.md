# DREAM AI — permanent training and native inference compute

This migration keeps the existing DREAM HOST, DREAM NOTEBOOK and Cloudflare R2 artifact vault. It does not create a new cloud store. Native DREAM training and inference do not use Kaggle, Hugging Face model downloads, Transformers or PEFT.

## What runs where

- **DREAM HOST / NOTEBOOK:** remains the control plane and notebook UI on the existing Render service.
- **Persistent NVIDIA machine:** runs the existing notebook worker as a systemd service, so it reconnects after process or machine restarts. When a release-ready native checkpoint exists, a second systemd service runs the native inference worker.
- **Cloudflare R2:** remains the durable store for prepared-corpus archives and native DREAM checkpoints. Every restore and upload is SHA-256/size verified; inference requires an exact, operator-approved checkpoint hash.
- **Native model:** `model/train.py` and `model/sft.py` train DREAM's own transformer from random initialization. `dream-host/native_inference_worker.py` loads only a validated native checkpoint from R2.

**Hardware boundary:** R2 is storage, not GPU compute. The current Render Free services are CPU web services. The installer must run on a persistent Linux machine that already has an NVIDIA GPU, a compatible driver and CUDA-enabled PyTorch. It is not designed for Android Termux or Render Free.

## Install the persistent notebook/GPU connection

1. On the Linux NVIDIA host, verify that `nvidia-smi` works and that the Python environment you will use reports `torch.cuda.is_available() == True`.
2. Check out the merged `dream-host-runtime` branch. The installer uses `/opt/dream-ai` by default and refuses to overwrite an unrelated directory or a Git checkout with uncommitted changes.
3. Run:

```sh
sudo bash scripts/install-dream-notebook-worker.sh
```

The installer prompts without echoing for three distinct existing secrets from the `dream-host-live` Render environment:

- `DREAM_HOST_NOTEBOOK_WORKER_TOKEN`
- `DREAM_HOST_WORKER_TOKEN` (reserved for native inference when it is approved)
- `DREAM_HOST_ARTIFACT_TOKEN` (also stored in the worker environment as `DREAM_NOTEBOOK_ARTIFACT_TOKEN`)

It writes them to the root-only file `/etc/dream-ai/notebook-worker.env`, installs both systemd units, and enables only the notebook worker. It does not print the secrets or change any Render environment variables. The native inference service remains disabled.

Check the training/notebook worker:

```sh
sudo systemctl status dream-notebook-worker
sudo journalctl -u dream-notebook-worker -f
```

The existing DREAM NOTEBOOK status should show the persistent host connected as a CUDA device. Notebook connections are training/execution only by default; they no longer trigger the legacy Qwen/Hugging Face model warm-up.

## Persistent data paths

The service creates these directories, owned by the dedicated `dreamai` account:

- `/var/lib/dream-ai/data/approved-records.jsonl` — source corpus only after real rights, privacy and quality review.
- `/var/lib/dream-ai/data/sft-train.jsonl` and `sft-validation.jsonl` — optional instruction-tuning splits after their review/audit gates.
- `/var/lib/dream-ai/data/prepared/` — prepared, audited corpus restored from or backed up to R2.
- `/var/lib/dream-ai/checkpoints/` — local checkpoint recovery copies.
- `/var/lib/dream-ai/logs/` — training and audit reports.

Copy reviewed source files to these paths through a secure file transfer. Do not put credentials or private user conversations into the training corpus.

## Use the existing DREAM NOTEBOOK UI

The existing `/notebook/training-template.ipynb` download now serves `training/dream_native_persistent_notebook.ipynb`. It uses the existing R2 artifact endpoints and artifact token; it restores the latest hash-verified prepared corpus and native checkpoints and uploads checkpoints during and after training.

Training uses provider-independent host paths, not hosted notebook input folders. The default training window is 1,500 seconds per cell, below the existing 1,800-second notebook execution ceiling. Later runs resume from the latest compatible R2 checkpoint. Existing source, rights, privacy, quality, language-balance and human-review checks are retained.

## Activate native inference only after evaluation

The native inference worker is present but deliberately disabled. It requires an instruction-tuned DREAM checkpoint, not a pretraining-only checkpoint or the legacy Qwen adapter.

Before activation, complete native pretraining and SFT, run the held-out evaluation and task-level evaluation, inspect the generated outputs, and approve a specific release. Record the SHA-256 of the exact `dream-native-sft-checkpoint.pt` artifact in R2. Do not use an unreviewed checkpoint hash.

Then:

1. On `dream-host-live` in Render, set `DREAM_HOST_MODEL_NAME=dream-native-v1` and `DREAM_HOST_ADAPTER_NAME=none`. Keep `DREAM_HOST_NOTEBOOK_CHAT_WARMUP` unset/false. Wait for the host deployment to finish.
2. On `dream-ai-live`, change its configured model identifier to `dream-native-v1` so app requests use the same model name.
3. On the persistent NVIDIA host, run:

```sh
sudo bash scripts/enable-dream-native-inference-worker.sh
```

The script reads the existing root-only worker/artifact secrets, asks for the approved checkpoint SHA-256, and confirms that the exact artifact is in the existing R2 vault and that DREAM HOST reports the matching native model/adapter settings. It then enables the native worker.

Verify its process and logs:

```sh
sudo systemctl status dream-native-inference-worker
sudo journalctl -u dream-native-inference-worker -f
```

DREAM HOST health should then report a ready GPU worker. Test a real chat request through `dream-ai-live` before retiring the legacy worker. The existing Qwen/adapter artifact is not overwritten or repurposed.

## Operational honesty

A persistent service manager restarts a worker after process or machine restarts; it cannot keep a rented GPU powered on after its provider suspends the machine or its credits expire. Render Free and R2 do not include always-on GPU capacity. This code prepares the persistent connection and model-loading path; it does not provision a GPU machine or claim that native DREAM is already trained or production-ready.
