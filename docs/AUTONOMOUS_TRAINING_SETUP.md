# DREAM AI autonomous training trigger

This workflow checks once per day and can submit the existing reviewed adapter-training notebook to Kaggle. It does not run a permanent GPU worker: Kaggle starts a bounded notebook job when GPU quota is available. The notebook's existing four-minute optimizer budget, v3 contributor-consent checks, exact dataset hashes and candidate-only uploads remain in force. The production adapter is never overwritten or automatically promoted.

## One-time GitHub configuration

In the `mjnr0477/dream-ai` repository, open **Settings → Secrets and variables → Actions** and add these repository secrets:

- `KAGGLE_USERNAME`: your Kaggle username.
- `KAGGLE_KEY`: your Kaggle API key. Generate it in Kaggle account settings and keep it private.

Add this repository variable:

- `DREAM_AI_KAGGLE_DATASET`: the private Kaggle Dataset ID in `owner/dataset-slug` form.

The dataset must contain exactly one each of `train.jsonl`, `validation.jsonl`, `manifest.json` and `audit-gate.json`. It must be private and must have passed DREAM's human review/audit pipeline. Do not upload raw conversations, account identifiers, passwords, API keys or unreviewed contributions.

## One-time Kaggle setup

The first workflow run creates the private Kaggle notebook and submits its first run. If `HF_TOKEN` is not yet attached, the notebook stops at its initial token check before loading the model or training. Open **DREAM AI Autonomous Adapter Training** in Kaggle, then use **Add-ons → Secrets** to add `HF_TOKEN` with a Hugging Face token that can write to the private `msjnr0477/dream-ai-adapter` repository. Kaggle's CLI does not expose secret creation, so this one-time step must be done in Kaggle's UI.

After adding the secret, run the workflow once from **GitHub → Actions → DREAM AI autonomous training trigger → Run workflow**. Later daily runs will submit the notebook automatically when the dataset and credentials are available.

## What it does and does not do

- Missing credentials or dataset configuration: exits safely without starting training.
- Missing or duplicate required dataset files: blocks the run.
- Existing Kaggle run still active: skips launching a second one.
- Successful notebook run: stores a new candidate under `candidates/` and training progress separately in the private Hugging Face repository.
- Production: unchanged. Candidate promotion remains a separate release decision after broad independent evaluation.

This is scheduled job orchestration, not continuous 24/7 GPU training. Kaggle quotas and GPU availability remain platform-controlled. The workflow must not be used to bypass the dataset's approval gates.
