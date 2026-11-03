# DREAM training data

This directory is reserved for datasets created or lawfully licensed for DREAM.

Do not put private user conversations, passwords, personal records, copyrighted books
without permission, or scraped personal data here.

## Data acceptance rules

Every dataset release must include:
- source and collection date;
- HTTPS source URL and HTTPS license text URL;
- explicit rights basis: `original-authorship`, `public-domain`, `open-license` or `explicit-permission`;
- written permission or an explicit license permitting the intended use;
- language and subject labels;
- human-confirmed rights, privacy and quality review for each record;
- personal-data review and redaction record;
- deduplication and quality-review report;
- train/validation/test split with no cross-split leakage;
- dataset version and SHA-256 manifest.

Each JSONL record must include `source_url` and `license_url` as absolute HTTPS URLs, a `rights_basis` value from the allowed set above, and `source_date` in strict `YYYY-MM-DD` form and set `rights_confirmed`, `privacy_reviewed` and `quality_reviewed` to `true`. The generated manifest reports record counts by language, license and rights basis, plus a source catalog containing source URLs and license URLs and the source-date range; inspect these before training. These flags record a human review; the script cannot independently verify legal rights or detect all personal information.

## Supervised instruction records

For instruction fine-tuning, each JSONL record uses the same provenance and review fields as pretraining records, plus a `messages` array. Messages must alternate `user` and `assistant`, may begin with one `system` message, and must end with an assistant response. Example shape:

```json
{"id":"sft-001","source":"DREAM-authored dialogue","source_group":"dream-tutor-v1","source_url":"https://example.org/dialogue","license":"CC0-1.0","license_url":"https://creativecommons.org/publicdomain/zero/1.0/","rights_basis":"original-authorship","source_date":"2026-01-15","rights_confirmed":true,"privacy_reviewed":true,"quality_reviewed":true,"language":"en","messages":[{"role":"user","content":"Explain fractions."},{"role":"assistant","content":"A fraction represents part of a whole..."}]}
```

Use distinct source groups, independently checked answers, and separate training/validation files. Pass the pretraining `manifest.json` and exact held-out `test.txt` file to SFT: it verifies their hash matches and its training and validation source groups must also be disjoint from the pretraining held-out test groups. The SFT checkpoint records the exact held-out test hash, and evaluation refuses any other corpus. SFT also rejects duplicate record IDs and normalized dialogue content across its training/validation files. The script verifies the review flags and data hashes; it cannot establish the truth of the underlying attestations.

## First target domains

1. General language and instruction following.
2. Mathematics with worked, independently checked solutions.
3. Programming examples with runnable tests and license provenance.
4. Teaching explanations and learner misconceptions.
5. Kiswahili and English, with additional Kenyan languages only where data quality and permission are adequate.
6. Kenyan curriculum and local-context content with authoritative source citations and version dates.

## Important

The files `smoke-test.jsonl`, `sft-smoke-train.jsonl` and `sft-smoke-validation.jsonl` are tiny, synthetic CI fixtures. They exist only to verify that preprocessing, random-initialization pretraining, held-out evaluation, assistant-only SFT and checkpoint loading work end to end. They are not a useful training corpus and must never be presented as model-quality evidence or production intelligence. A capable model requires a much larger, diverse, reviewed corpus and substantial compute.
