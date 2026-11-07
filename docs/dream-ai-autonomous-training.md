# DREAM AI autonomous adapter training

This automation replaces the repeated manual Kaggle training steps with a scheduled control plane. GitHub Actions checks the DREAM AI learning store every day and, when a qualifying reviewed dataset exists, uses Kaggle's GPU notebook API to launch the existing resumable Qwen3 LoRA training process.

## What it does

1. Reads only public.dream_ai_training_examples rows with explicit v3 consent and both status and review_status set to approved.
2. Requires a recorded human review of rights, privacy, quality and training approval, plus at least 32 examples from 8 independent source sessions.
3. Creates deterministic, source-group-isolated train/validation splits and runs the existing privacy, license and provenance audit gates.
4. Uploads those files to an owner-only private Kaggle dataset and launches a private Tesla T4 notebook.
5. Continues an interrupted run only when the exact dataset hash, audit-gate hash and parent-adapter hash match the saved private progress checkpoint.
6. Starts each new dataset cycle from the latest candidate whose held-out validation passed, so approved learning can accumulate between cycles. The original production adapter remains the rollback baseline and is never changed by this job.
7. Uploads a result to candidates/<candidate-id> in the private Hugging Face adapter repository and records the run in Supabase.

The live adapter files at the root of msjnr0477/dream-ai-adapter are never replaced by this workflow. A candidate is not active merely because training or validation completed. Independent evaluation and an explicit promotion decision remain required.

## Required one-time GitHub Actions secrets

Open the DREAM AI repository on GitHub, then go to Settings → Secrets and variables → Actions → New repository secret. Create these three secrets:

| Secret | Value |
|---|---|
| DREAM_AI_SUPABASE_SERVICE_ROLE_KEY | The service-role key from the Dream education Supabase project ldpcawixzxcolbmakrfc. Do not use the Mariona/VYBORA project. |
| KAGGLE_JSON | The complete contents of the kaggle.json file downloaded from Kaggle account settings → API. It contains the Kaggle username and API key. |
| HF_TOKEN | A Hugging Face fine-grained token with read/write access only to msjnr0477/dream-ai-adapter. |

The Supabase project URL is already configured in the workflow. Never commit credentials or paste them into source files, issues or chat.

Kaggle API-pushed kernels do not inherit interactive notebook secrets reliably. The workflow therefore places HF_TOKEN in a private, owner-controlled Kaggle input dataset so the private training notebook can download the current adapter and upload a candidate. The token is not printed or committed to GitHub. Use a dedicated fine-grained token; rotate it if the private Kaggle dataset is ever shared or its access changes.

## Reviewing new contributions

Only a person can confirm the source rights and inspect the actual exchange for private information, errors and quality. A user's Share with AI action submits an example for review; it does not grant automatic training approval.

Run this command in a trusted local environment after setting these two environment variables privately:

- DREAM_AI_SUPABASE_URL=https://ldpcawixzxcolbmakrfc.supabase.co
- DREAM_AI_SUPABASE_SERVICE_ROLE_KEY=<the Dream education project service-role key>

Then run:

    python scripts/review_dream_ai_pending.py --limit 20

The script fetches only pending v3 opt-ins, presents each exchange for the four explicit review decisions, and syncs only approved records to the private training table. Approved examples and a PII-free decision ledger are stored under ~/.dream-ai-learning/ by default. Keep that directory private and never commit it.

## Schedule and limits

- The workflow checks the database daily at 02:17 UTC and can also be started with GitHub Actions → DREAM AI autonomous adapter training → Run workflow.
- It does nothing when the dataset has fewer than 32 approved examples, fewer than 8 independent source groups, a failed audit, or no new training data.
- The Kaggle notebook runs a bounded four-minute optimizer segment and stores a resumable private checkpoint. If a segment reaches its time budget, a later daily run can resume the exact same run.
- Kaggle GPU availability, free quotas, GitHub Actions availability and third-party service limits are not guaranteed. This is recurring automation, not an unconditional promise of uninterrupted training forever.
- The workflow continues to use Qwen3-1.7B as the foundation model and updates a DREAM-specific LoRA adapter. It does not yet train a new foundation model from random initialization.
- New training cycles continue from the most recent technically validated private candidate and use the cumulative set of approved examples. A candidate remains non-active and cannot become the live adapter automatically.

## Current starting state

At the time this automation was prepared, the DREAM AI store contained two pending legacy v1/v2 examples and zero approved v3 examples. Those legacy examples are deliberately excluded. The 10-record Kaggle experiment from a single assistant-generated session is also excluded. The first automatic GPU run will occur only after sufficient new, independently sourced v3 contributions have been explicitly reviewed and approved.
