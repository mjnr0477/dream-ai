# DREAM AI Opt-in Learning Pipeline — Design

**Date:** 2026-11-06  
**Status:** Approved implementation direction by project owner’s standing instruction to proceed autonomously.  
**Scope:** One contained increment: private opt-in exchange curation and leakage-aware SFT preparation.

## Intent

Turn explicitly shared DREAM AI exchanges into a safe, reproducible source of supervised fine-tuning candidates. The pipeline must never equate a share click with approval to train. It must preserve provenance, consent, privacy review, quality review, rights review and human approval as separate gates.

## Current state

- The application sends one user prompt and its corresponding DREAM response only after the user clicks the per-answer share control.
- Supabase stores the row in `dream_ai_training_examples` with `consented=true`, `review_status=pending` and `status=pending`.
- The learning tables are server-only; anonymous and authenticated clients do not have table grants.
- The current table contains one v1 opt-in record. It has no source-group fingerprint and remains pending.
- `data/review_candidates.py` already requires four explicit human confirmations: rights, privacy, quality and training approval.
- `model/sft.py` already checks review flags, provenance, duplicate content, source-group leakage and the exact held-out pretraining test file.

## Goals

1. Version the user-facing contribution notice so it explicitly describes possible use in DREAM model training after human review.
2. Attach a random, temporary browser-session group ID to each v2 contribution; hash it in the trusted Edge Function and store only the hash.
3. Export v2 pending opt-ins into candidate SFT JSONL only when the data has the required consent and provenance fields.
4. Keep all approval flags false in candidate exports. Reuse the existing interactive reviewer rather than creating automatic approval.
5. Deterministically split approved SFT examples by source group into separate train and validation files, and write a hash-based manifest.
6. Refuse to create splits until there are at least four independent source groups.
7. Keep all learner tables, model weights, runtime configuration and production model activation unchanged.

## Non-goals

- No automatic approval or unattended training.
- No training on the existing v1 record.
- No upload of conversation histories, saved memory, account IDs or raw session IDs.
- No new Supabase learner-table access or public read policies.
- No use of a service-role key in Kaggle, browser code or GitHub.
- No automatic activation of a candidate checkpoint and no claim that the model is frontier-level.

## Approaches considered

1. **Train directly from opt-in rows.** Rejected: an opt-in click is not a substitute for rights, privacy, safety and quality review, and there is no split-leakage protection.
2. **Give the training notebook broad Supabase service-role access.** Rejected: this would expose unrelated Dream Education records to the training environment and would expand the blast radius of a compromised notebook.
3. **Private export → unapproved candidates → existing human review → source-group-isolated SFT splits.** Selected: it reuses the project’s current review and SFT guards, keeps the Supabase store private, requires no new broad credentials in Kaggle, and makes the dataset reproducible.

## Architecture and data contract

### Collection

- The browser creates one UUID in `sessionStorage` and reuses it only for the current browser tab/session.
- The browser submits the existing exact prompt/response pair, the v2 consent version and this temporary UUID.
- The Node server validates the request and forwards it; it never stores the raw UUID.
- The learning Edge Function validates the UUID and stores `SHA256("dream-ai-session-group-v1\0" + UUID)` inside the existing `metadata` JSONB field.
- The raw UUID is not persisted. Older v1 clients remain accepted for compatibility, but v1 records do not receive a group fingerprint and cannot enter this pipeline.

### Candidate export

`data/export_dream_ai_optins.py` reads a private JSONL export of `dream_ai_training_examples`. It emits only v2, consented, pending rows with a valid 64-character source-group fingerprint and valid text/date fields. It never changes the database.

Each output candidate has the shape expected by `data/review_candidates.py` and `model/sft.py`: one user message, one assistant message, source/permission references, source date, language (or `und` when not declared), explicit-permission rights basis, source-group identity and all four approval flags set to false.

Rows that fail eligibility are skipped with a non-sensitive reason code. Candidate content is written only to the requested local output file; it is not printed to logs.

### Human review

The owner runs the existing interactive reviewer. A row only enters the approved JSONL output after four separate affirmative answers for the exact candidate. A reviewer must inspect the actual content and underlying rights; the exporter cannot decide those questions.

### SFT preparation

`data/prepare_dream_ai_sft_splits.py` accepts only human-approved SFT records. It checks all four flags, HTTPS provenance, dates, roles, duplicate IDs/content and independent source groups. It assigns complete source groups deterministically to train or validation; no group can appear in both. It requires at least four groups by default and produces a manifest with dataset hash, split hashes, record counts and domain-separated group fingerprints.

The existing `model/sft.py` remains the final training guard and must still receive the pretraining manifest and exact pretraining test file. A successful SFT run is a candidate only; existing checkpoint audit and evaluation/release gates still apply.

## Privacy and safety constraints

- v1 opt-ins remain pending and are not eligible for export.
- No personal or account identifiers are stored as source-group values.
- Session grouping is a leakage aid, not a unique-person identifier; examples from separate sessions may still belong to the same person and need reviewer judgment.
- Candidate files are private training data and must only be placed in private Kaggle datasets.
- A source-group fingerprint is pseudonymous, not anonymous.
- No model training occurs until all human review flags are true and the independent split gate succeeds.
