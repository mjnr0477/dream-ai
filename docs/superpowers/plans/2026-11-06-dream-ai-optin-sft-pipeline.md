# DREAM AI Opt-in SFT Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a privacy-conscious path from v2 opt-in exchanges to human-approved, source-group-isolated DREAM SFT datasets.

**Architecture:** The browser adds a temporary session grouping UUID to the exact exchange only; the Supabase Edge Function stores only a SHA-256 fingerprint. A local importer turns eligible v2 rows into unapproved SFT candidates, the existing reviewer approves them, and a deterministic splitter creates train/validation files for the existing SFT trainer.

**Tech Stack:** Node.js 20, browser JavaScript, Supabase Edge Functions (Deno), Python 3.11 standard library, existing DREAM review/SFT scripts.

**Spec:** `docs/superpowers/specs/2026-11-06-dream-ai-optin-sft-pipeline-design.md`

## Global Constraints

- Never train directly from a share click; all approval flags remain false until the existing human reviewer confirms them.
- Keep `dream_ai_training_examples` server-only; do not add anon/authenticated grants or read policies.
- Do not read, modify or export Dream Education learner records.
- Never put a service-role key in browser, Kaggle or GitHub code.
- Do not store raw session IDs, account IDs or full chat histories.
- v1 consent records remain ineligible for training.
- Require at least four independent source groups for SFT train/validation splitting.
- Do not modify model activation or bypass `audit_checkpoint.py`, `validate_release.py`, or the native runtime identity checks.

## Review Focus

- v1 records and old clients: remain pending and cannot be exported.
- Missing/invalid session group: collection may remain compatible, but no SFT candidate is emitted.
- Malformed, duplicate or oversized records: export safely rejects/skips with a reason code, never logs raw content.
- Same source group represented multiple times: all its records stay in one split.
- Too few independent groups: splitting refuses rather than creating a misleading validation set.

---

### Task 1: Add exporter and split tests first

**Files:**
- Create: `data/test_dream_ai_optin_pipeline.py`
- Create: `data/dream_ai_optin_pipeline.py` (initial testable interface only)

**Interfaces:**
- `export_optin_rows(rows) -> (candidates, report)`
- `prepare_sft_splits(records, min_source_groups=4) -> (train, validation, manifest)`

- [ ] Write tests for v2 eligibility, v1 exclusion, missing source-group hash, false approval flags, malformed dates, and private report codes.
- [ ] Write tests for approved-only splitting, required review flags, group isolation, deterministic output, duplicate rejection and minimum group count.
- [ ] Run the data tests and confirm they fail on the missing behavior, not on test setup.

### Task 2: Implement the opt-in candidate exporter

**Files:**
- Modify: `data/dream_ai_optin_pipeline.py`
- Create: `data/export_dream_ai_optins.py`

**Interfaces:**
- `export_optin_rows(rows)` returns candidate records and counts only; it does not approve or train anything.
- CLI accepts a Supabase JSON/JSONL export and writes candidate JSONL plus a non-sensitive report.

- [ ] Implement v2-only consent checks and validate the content text, timestamp and 64-character source-group fingerprint.
- [ ] Build the SFT message pair and trusted permission/provenance references.
- [ ] Set `rights_confirmed`, `privacy_reviewed`, `quality_reviewed`, and `approval_granted` to false on every candidate.
- [ ] Deduplicate normalized content and ensure the report never includes the prompt, response or raw group identifier.
- [ ] Run the exporter tests and check output examples.

### Task 3: Implement leakage-aware SFT splitting

**Files:**
- Modify: `data/dream_ai_optin_pipeline.py`
- Create: `data/prepare_dream_ai_sft_splits.py`

**Interfaces:**
- `prepare_sft_splits(records, min_source_groups=4)` returns deterministic train/validation record arrays and a SHA-256 manifest.

- [ ] Require every record to have all four true review flags and valid provenance.
- [ ] Reject duplicate IDs and normalized dialogue content.
- [ ] Assign source groups deterministically so no group crosses train/validation.
- [ ] Refuse fewer than four unique groups by default.
- [ ] Write train.jsonl, validation.jsonl and manifest.json with hashes/counts/fingerprints.
- [ ] Run focused tests and confirm determinism under shuffled input.

### Task 4: Add v2 consent and privacy-preserving source grouping

**Files:**
- Modify: `public/app.js`
- Modify: `public/index.html`
- Modify: `server.js`
- Replace: Supabase `dream-ai-learning-ingest-v3` Edge Function code
- Create: `public/learning-data-terms.html`

**Interfaces:**
- Browser submits `consentVersion: "dream-learning-opt-in-v2"` and a sessionStorage UUID as `sourceGroupId`.
- Server validates the v2 source-group UUID and forwards it; raw ID is not logged or stored.
- Edge Function hashes the UUID with the `dream-ai-session-group-v1\0` domain prefix and writes only the hash into `metadata.source_group_sha256`.
- Old v1 submissions remain accepted for compatibility but are ineligible for export.

- [ ] Add visible v2 consent wording and link to the contribution terms.
- [ ] Add safe sessionStorage UUID generation and per-exchange v2 submission.
- [ ] Validate and forward v1/v2 consent versions without logging prompt/response/session identifiers.
- [ ] Update Edge Function to accept v1/v2 and hash a valid v2 UUID before insert.
- [ ] Verify the deployed function and confirm the existing pending record is unchanged.

### Task 5: Connect the preparation flow to CI and documentation

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `data/README.md`
- Create: `docs/dream-ai-optin-sft-workflow.md`

- [ ] Run data pipeline tests in CI.
- [ ] Document private Supabase JSON export, candidate import, four-question human review, split generation and Kaggle SFT invocation.
- [ ] Make clear that session grouping is not person identity, that data files must remain private, and that one sample cannot train a credible model.
- [ ] Run the complete application and model test suites.
- [ ] Review the final diff, CI run, Edge Function state and Supabase row counts before merging/deploying.

