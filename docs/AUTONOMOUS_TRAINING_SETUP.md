# DREAM AI automated training — retired

**Status as of 2026-11-14: no scheduled third-party training workflow is active.**

The previous scheduled adapter-training triggers have been retired. They depended on external GPU accounts and private model-store credentials, and did not provide a permanent, owner-controlled DREAM runtime.

The latest preflight checks confirmed that:
- the Kaggle launcher stopped because its API token was not configured; no GPU job was submitted;
- the adapter-training workflow skipped data preparation because a dedicated DREAM AI Supabase project was not configured;
- no active Hugging Face Jobs were present;
- the connected Supabase account did not contain a dedicated DREAM AI project.

These checks did not read or reuse the Mariona, Dream Education, or Teacher databases. Do not configure another application's URL or service-role key for DREAM AI.

## Current training policy

Do not restore a scheduled external-provider workflow as a workaround. Training may resume only when all of the following are true:

1. A genuinely connected, operator-controlled CUDA GPU is verified at runtime.
2. The exact pretraining and/or SFT corpus has passed human rights/licensing, privacy, quality and training-consent review.
3. Source-group-isolated train, validation and held-out test splits are bound to complete manifests and audit hashes.
4. A single session-wide time budget limits all training stages to no more than five minutes.
5. Checkpoints preserve optimizer and RNG state on persistent storage, and candidate outputs remain separate from production.
6. Checkpoint integrity, held-out evaluation and independent release review pass before any candidate is promoted.

The native DREAM trainers are in `model/train.py` and `model/sft.py`. Their individual `--max-seconds` flags are not, by themselves, a shared session-wide budget. A bounded native session controller and a verified persistent GPU host are still required before unattended training can be enabled.

CI's tiny synthetic CPU pretraining/SFT smoke tests verify code paths only. They are not real model training and must not be reported as GPU progress.
