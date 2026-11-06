# Preparing DREAM AI's opt-in learning examples

This workflow prepares candidate examples from the private DREAM AI learning store. It does **not** automatically approve contributions or start training.

## 1. Export only the dedicated DREAM AI table

In the Supabase Dashboard for the Dream AI learning store project, export only:

`public.dream_ai_training_examples`

Do not export or attach Dream Education learner tables, `learning_history`, `learning_assessments`, learner profiles or the whole project database. Save the export privately on a trusted device. Use CSV or JSON; never make the export a public Kaggle dataset.

The exporter will accept only rows with:
- `consented = true`
- `consent_version = dream-learning-opt-in-v2`
- `review_status = pending` and `status = pending`
- a valid `metadata.source_group_sha256` produced by the trusted Edge Function
- valid prompt/response text and a valid creation timestamp

Legacy v1 contributions remain stored and pending, but are intentionally excluded because they have no session grouping fingerprint. Invalid rows are skipped with reason counts; the report does not print prompts, answers, database row IDs or raw session UUIDs.

## 2. Create unapproved SFT candidates

From the repository root:

```sh
python data/export_dream_ai_optins.py \
  --input /private/path/dream-ai-training-examples.csv \
  --output data/dream-ai-optin-candidates.jsonl \
  --report data/dream-ai-optin-export-report.json
```

Keep both output files private. Candidate approval fields are always false. The exporter hashes database row identifiers into deterministic candidate IDs and uses only the stored one-way session-group fingerprint for group boundaries.

Review the aggregate report. An export containing zero eligible records is expected until new v2 contributions are made and reviewed.

## 3. Human-review every candidate

Use the existing reviewer, which requires separate affirmative answers for rights, privacy, quality and explicit training approval:

```sh
python data/review_candidates.py \
  --input data/dream-ai-optin-candidates.jsonl \
  --output data/dream-ai-optin-approved.jsonl \
  --decisions data/dream-ai-optin-review-decisions.jsonl \
  --limit 25
```

Read the displayed exchange and its linked terms. Check that the contributor had the right to submit the content; check every answer for privacy, safety, correctness and quality. Do not approve a row merely because the user clicked the contribution button. Rejected or skipped examples are not eligible for training.

## 4. Build leakage-aware training and validation splits

```sh
python data/prepare_dream_ai_sft_splits.py \
  --input data/dream-ai-optin-approved.jsonl \
  --output-dir data/dream-ai-optin-sft-prepared
```

The splitter requires at least four independent session groups by default. It assigns entire session groups to either training or validation, deterministically, and writes:
- `train.jsonl`
- `validation.jsonl`
- `manifest.json`

It refuses records without all four review flags, proper HTTPS provenance, an explicit permitted rights basis, valid source dates, alternating user/assistant messages, unique IDs and unique normalized dialogue content. The manifest contains SHA-256 hashes and privacy-preserving group fingerprints rather than raw group values.

A browser-tab session group is a leakage-control boundary, not proof of unique contributor identity. Separate tabs or sessions may belong to the same person; human reviewers must consider repeat or coordinated content.

## 5. Audit and train a candidate model

Run the existing corpus/privacy audit independently on both supervised splits:

```sh
python model/audit_corpus.py --input data/dream-ai-optin-sft-prepared/train.jsonl --out data/dream-ai-optin-sft-prepared/train-audit.json
python model/audit_corpus.py --input data/dream-ai-optin-sft-prepared/validation.jsonl --out data/dream-ai-optin-sft-prepared/validation-audit.json

python data/verify_dream_ai_sft_audits.py \
  --manifest data/dream-ai-optin-sft-prepared/manifest.json \
  --train-audit data/dream-ai-optin-sft-prepared/train-audit.json \
  --validation-audit data/dream-ai-optin-sft-prepared/validation-audit.json \
  --output data/dream-ai-optin-sft-prepared/audit-gate.json
```

The audit gate passes only when both reports match the exact split-file hashes in the manifest and contain zero parse errors, findings, possible-PII flags and unreviewed records. Any finding blocks fine-tuning until the underlying candidate is resolved and the splits and audits are regenerated.

Then use a genuinely pretrained DREAM checkpoint and its exact pretraining manifest and held-out test file with `model/sft.py`. Follow `model/BASELINE.md` for the required arguments, checkpoint audit, task evaluation, multilingual evaluation and release-gate checks. An SFT candidate does not become active automatically.

Only use a **private** Kaggle Dataset for approved training records. Never upload service-role/secret keys, raw session IDs, user account details or the full database export. Kaggle is a temporary notebook training environment, not an always-on GPU server. Download the checkpoint, manifests and evaluation reports before the session expires.

## Current state and non-claims

The previously submitted v1 contribution remains pending and is not eligible for this exporter. The pipeline does not train from a single opt-in. No production-trained DREAM checkpoint or automatic self-training job is claimed by this workflow.
