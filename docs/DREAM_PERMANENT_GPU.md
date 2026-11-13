# DREAM AI — persistent Lightning GPU connection

This branch adds a provider-independent worker setup for the existing DREAM NOTEBOOK. It uses the already deployed DREAM HOST and its existing Cloudflare R2 artifact vault. It does not use Kaggle, Hugging Face, or Railway for native DREAM training.

## What stays permanent

- The existing Render DREAM HOST and its notebook URL remain unchanged.
- Cloudflare R2 remains the durable off-machine store for the reviewed training corpus and native model checkpoints.
- The Lightning Studio filesystem is used for the notebook workspace, code, downloaded datasets and local working checkpoints.
- The notebook worker saves native DREAM training checkpoints to R2 every 25 optimizer steps and restores the latest verified checkpoint when the notebook setup cell runs again.
- The worker keeps a persistent per-notebook workspace between worker restarts. The Python kernel's in-memory variables still reset after a full process restart; rerun the setup cell to reconstruct them.

The R2 vault currently contains the connection-test artifact and the Qwen adapter recovery archive. That Qwen archive is a separate fallback and is **not** a native DREAM pretraining checkpoint. The native prepared corpus and native DREAM checkpoint must be created before native training can begin.

## 1. Create the free GPU workspace

1. Open [Lightning AI](https://lightning.ai/) and create or sign into an account.
2. Complete the account verification Lightning requires. Its availability varies by country.
3. Create one Studio and select a single **T4 GPU** if it is offered in your free allowance. T4 has 16 GB VRAM; use the lowest-cost GPU choice shown in your account.
4. Stay on the Free tier. Do not add a payment card, buy credits, disable free-tier sleep settings, or select a paid machine. If the Studio asks for payment or the free GPU allowance is exhausted, stop rather than enabling billing.

Lightning currently advertises up to 80 free GPU hours to start, but this is an allowance, not a permanent GPU promise. Free Studios require a restart every four hours; the exact GPU/credit allowance can vary by machine and account. Studio files persist when the environment sleeps or restarts. Background scripts can continue while the browser is closed, and an active training job prevents idle auto-sleep. See [official pricing](https://studio.lightning.ai/pricing/), [Studio quick start](https://lightning.ai/docs/platform/build/ai-studio), and [background execution](https://lightning.ai/docs/platform/build/ai-studio/background-execution).

## 2. Clone this branch in the Studio terminal

Run:

```bash
git clone --depth 1 --branch dream-permanent-compute https://github.com/mjnr0477/dream-ai.git ~/dream-ai
cd ~/dream-ai
```

## 3. Add the existing secrets in Studio environment settings

Set these variables in the Studio's private environment/secrets settings. Copy the values from the existing Render `dream-host-live` service; do not paste them into chat, GitHub, notebook cells, or this repository.

| Studio variable | Value |
|---|---|
| `DREAM_HOST_URL` | `https://dream-host-live.onrender.com` |
| `DREAM_HOST_NOTEBOOK_WORKER_TOKEN` | The existing Render value with the same name |
| `DREAM_NOTEBOOK_ARTIFACT_TOKEN` | The existing Render `DREAM_HOST_ARTIFACT_TOKEN` value |
| `DREAM_HOST_NOTEBOOK_TRUSTED_MODE` | `1` |
| `DREAM_NOTEBOOK_PERSISTENT_ROOT` | `$HOME/dream-ai-workspace` |
| `DREAM_REPO_BRANCH` | `dream-permanent-compute` |

Do **not** add `HF_TOKEN`, `DREAM_NOTEBOOK_HF_TOKEN`, a Kaggle secret, or a new model-provider API key for native DREAM training.

The worker token authenticates the GPU worker to DREAM HOST. The artifact token must match the existing R2 artifact token configured on DREAM HOST. These are two different secrets.

## 4. Start and verify the connection

From the repository root, run:

```bash
bash scripts/start-lightning-worker.sh
```

The script checks CUDA visibility, calls the existing DREAM HOST health route, authenticates to the existing R2 artifact vault, displays the artifact inventory, and only then starts the worker. It does not print secret values or install/replace PyTorch.

A successful connection prints `Registered with DREAM NOTEBOOK` and reports the GPU device. Open [DREAM NOTEBOOK](https://dream-host-live.onrender.com/notebook); its worker status should show the Lightning GPU. Do not consider it connected until that status confirms a worker and device.

## 5. Start native DREAM training

In DREAM NOTEBOOK, load the **native DREAM from-scratch** training workflow. Put the already reviewed `approved-records.jsonl` in the notebook `datasets/` folder using its dataset tools or the Studio terminal, then run the setup and audit cells before the training cell.

The workflow will stop if the training corpus is missing, unreviewed, lacks rights/privacy/quality approval, or fails provenance checks. Do not mark records approved just to pass the gate. The first native corpus backup is named `dream-native-prepared-corpus.zip`; native training checkpoints are stored in R2 as `dream-native-pretrain-checkpoint.pt` and later instruction checkpoints.

The existing Qwen adapter archive remains untouched. The separate Qwen adapter-improvement notebook still depends on Hugging Face and is not part of this HF-free native training path. Do not switch the live chat service away from its current model until a native checkpoint has passed independent evaluation and a separate release review.

## Limits that remain

- A free GPU is temporary compute, not a permanently allocated dedicated GPU. A free Studio needs four-hour restarts and its GPU allowance is limited.
- The Render notebook queue itself is currently in-memory. R2-backed training checkpoints survive, but a Render host restart can lose an in-flight notebook job record. Re-running the affected notebook cell resumes from the last verified checkpoint.
- This branch prepares the Lightning worker and persistence path. It does not claim that a Studio has been created, that a GPU is connected, or that training has started. Those can only be confirmed after the worker runs and appears in DREAM NOTEBOOK.
