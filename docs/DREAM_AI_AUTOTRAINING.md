# DREAM AI — automatic GPU training

## What runs automatically

- GitHub Actions checks the private Kaggle training kernel once per day (09:15 East Africa time).
- If the previous run is queued or running, it does not submit another GPU job.
- If the previous run completed, the next scheduled run submits another bounded adapter-improvement cycle.
- If a run failed or was cancelled, automatic retries pause. Review the private Kaggle logs, fix the cause, then manually dispatch the workflow with `force_retry=true`.
- The Kaggle notebook restores the matching adapter, optimizer and RNG progress from DREAM HOST's private R2 vault, verifies hashes, then saves a new resumable checkpoint and candidate.
- The live chat adapter is not overwritten. Candidates require review and are never promoted automatically.

Training only starts when the private Kaggle account dataset contains exactly one `train.jsonl`, `validation.jsonl`, `manifest.json` and `audit-gate.json`, and the notebook's consent, rights, privacy, human-review and evaluation checks pass.

## One-time credentials

Add these as **GitHub repository Actions secrets** in `mjnr0477/dream-ai`:

- `KAGGLE_USERNAME`: the Kaggle account username.
- `KAGGLE_API_TOKEN`: a current Kaggle API token from https://www.kaggle.com/settings/api.

The private Kaggle notebook must also have these Kaggle Secrets configured once, before its first GPU run:

- `HF_TOKEN`: access to the private DREAM adapter repository.
- `DREAM_NOTEBOOK_ARTIFACT_TOKEN`: access to the private DREAM HOST R2 artifact vault.

Do not put tokens in the repository, notebook source, workflow YAML or chat.

## Limits

This is automatic and resumable, not a permanently running free GPU. The scheduled controller talks directly to Kaggle's API; it does not require the interactive DREAM NOTEBOOK page or its browser connector to remain open. Kaggle controls free GPU availability and enforces quota and session limits. The permanent assets are the private training data, verified R2 checkpoints, optimizer/RNG state, candidate history and scheduler. When Kaggle stops a session, a later scheduled run can restore the last checkpoint that finished uploading.

GitHub scheduled workflows run from the repository's default branch. GitHub may disable scheduled workflows in a public repository after 60 days without repository activity, so the schedule needs occasional activity or re-enabling in Actions.
