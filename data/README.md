# DREAM training data

This directory is reserved for datasets created or lawfully licensed for DREAM.

Do not put private user conversations, passwords, personal records, copyrighted books
without permission, or scraped personal data here.

## Data acceptance rules

Every dataset release must include:
- source description, an HTTPS source-page URL and collection date;
- an HTTPS license/permission reference and a declared rights basis (`original-authorship`, `public-domain`, `open-license` or `explicit-permission`);
- written permission or an explicit license permitting the intended use;
- language and subject labels;
- human-confirmed rights, privacy and quality review for each record;
- personal-data review and redaction record;
- deduplication and quality-review report;
- deterministic source-group-stratified train/validation/test splits with no cross-split leakage; languages need at least three independent source groups to be represented in all three splits;
- dataset version and SHA-256 manifest.

Before preparation, run `python model/audit_corpus.py --input data/approved-records.jsonl --out eval-results/corpus-audit.json`; inspect its metadata, duplicate-content and possible-PII findings. The Colab workflow blocks on provenance errors, possible PII and incomplete human reviews. Run the same audit separately on both SFT splits. This is heuristic triage, not legal or privacy approval.

Each JSONL record must include `source_url`, `license_url`, `rights_basis` and `source_date` in strict `YYYY-MM-DD` form, and set `rights_confirmed`, `privacy_reviewed` and `quality_reviewed` to `true`. The preparation gate requires absolute HTTPS source and license URLs and rejects an unverified rights basis. The generated manifest fingerprints record and source-group identifiers with domain-separated SHA-256 values (`identifier_format: sha256-domain-separated-v1`) instead of publishing caller-supplied raw IDs. SFT supports older manifests without this marker by fingerprinting their legacy raw source-group IDs before leakage checks. These fingerprints are pseudonymous, not anonymized; never put secrets or personal data in identifier fields. The manifest reports record counts by language and license plus the source-date range; inspect these before training. It also reports each language's independent source-group count and split coverage. Languages with at least three independent groups are assigned at least one group to each split; rarer languages are marked as not yet eligible for three-way held-out evaluation. These flags record a human review; the script cannot independently verify legal rights or detect all personal information.

## FineWeb2 multilingual candidate import

The official [HuggingFaceFW/FineWeb-2 dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) describes a multilingual Common Crawl-derived collection, its ODC-By 1.0 collection license, source metadata and known limitations. The dataset is very large; do not attempt to download the full corpus to a phone or a small free service. Start with a small, language-balanced local export.

Convert a local JSONL export into review candidates:

```sh
python model/import_fineweb2_candidates.py \
  --input data/fineweb2-small-export.jsonl \
  --output data/fineweb2-candidates.jsonl \
  --summary data/fineweb2-candidates-summary.json \
  --languages eng,swa,amh,ibo,yo \
  --max-per-language 1000
```

The importer preserves each row's URL, date, language and language-script metadata, uses the source hostname as a leakage-group key, removes duplicate IDs/text, and selects a deterministic maximum per language. It deliberately sets `rights_confirmed`, `privacy_reviewed`, `quality_reviewed` and `approval_granted` to false. These are review candidates only and are not accepted by `prepare_corpus.py`.

**Important licensing boundary:** ODC-By describes the dataset collection; it does not automatically settle the copyright, privacy or reuse rights of every underlying webpage. FineWeb-2 also documents Common Crawl terms, possible personal/sensitive information, opt-out handling and language-classification limitations. Human reviewers must verify the actual terms and source-level rights, check opt-outs and personal data, and assess quality before approving any record. Do not bulk-change review flags to bypass this gate. The importer does not provide legal advice or train the model.

## Supervised instruction records

For instruction fine-tuning, each JSONL record uses the same provenance and review fields as pretraining records, plus a `messages` array. Messages must alternate `user` and `assistant`, may begin with one `system` message, and must end with an assistant response. Example shape:

```json
{"id":"sft-001","source":"DREAM-authored dialogue","source_group":"dream-tutor-v1","source_url":"https://example.org/dream-tutor-v1","license":"CC0-1.0","license_url":"https://creativecommons.org/publicdomain/zero/1.0/","rights_basis":"original-authorship","source_date":"2026-01-15","rights_confirmed":true,"privacy_reviewed":true,"quality_reviewed":true,"language":"en","messages":[{"role":"user","content":"Explain fractions."},{"role":"assistant","content":"A fraction represents part of a whole..."}]}
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
