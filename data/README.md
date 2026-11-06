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

Corpus preparation deduplicates text using Unicode NFKC normalization, whitespace folding and case folding; the pre-ingestion audit uses the same normalization policy to flag equivalent content. This catches compatibility-form duplicates as well as ordinary whitespace/case variants, but does not claim semantic or paraphrase-level deduplication.\n\nBefore preparation, run `python model/audit_corpus.py --input data/approved-records.jsonl --out eval-results/corpus-audit.json`; inspect its metadata, duplicate-content and possible-PII findings. The Colab workflow blocks on provenance errors, possible PII and incomplete human reviews. Run the same audit separately on both SFT splits. This is heuristic triage, not legal or privacy approval.

Each JSONL record must include `source_url`, `license_url`, `rights_basis` and `source_date` in strict `YYYY-MM-DD` form, and set `rights_confirmed`, `privacy_reviewed` and `quality_reviewed` to `true`. The preparation gate requires absolute HTTPS source and license URLs and rejects an unverified rights basis. The generated manifest reports record counts by language and license plus the source-date range; inspect these before training. Record IDs and source-group IDs in the manifest are domain-separated SHA-256 fingerprints (`identifier_format: sha256-domain-separated-v1`), not raw caller-supplied identifiers; the SFT leakage checks compare the same fingerprints. These fingerprints reduce accidental raw-ID disclosure but are pseudonymous, not anonymized: low-entropy IDs may be guessed, so never put secrets or personal data in identifier fields. Older manifests without this field remain supported by hashing their legacy raw group IDs before comparison. The manifest also reports each language's independent source-group count and split coverage. Languages with at least three independent groups are assigned at least one group to each split; rarer languages are marked as not yet eligible for three-way held-out evaluation. These flags record a human review; the script cannot independently verify legal rights or detect all personal information.

## Commercial-ready license compatibility

DREAM's training pipeline uses the versioned policy in `model/license_policy.py` before pretraining and native SFT. This is conservative automated triage, not legal advice or a substitute for checking the source page and rights holder. The policy assumes DREAM may later be operated as a paid service, so non-commercial-only material is not accepted into this corpus by default.

The current standard allow-list is:
- CC0 1.0, with no attribution condition under that dedication;
- CC BY 4.0, with attribution and change-notice obligations;
- Apache-2.0 and MIT, where the specific item is actually released under that license;
- Public Domain Mark 1.0 only when the record is explicitly marked public domain and a human has confirmed the applicable rights basis.

The gate blocks Creative Commons NonCommercial terms (NC) unless a separate HTTPS permission record expressly covers commercial AI model training and a human confirms that permission. ShareAlike (SA), NoDerivatives (ND), unknown licenses and collection-only license claims are held for review unless separate permission resolves the issue. A dataset collection license alone does not establish rights in every underlying web page. A generic `rights_confirmed: true` flag cannot override a known non-commercial restriction.

The gate also writes a text-free `train.provenance.jsonl`, `validation.provenance.jsonl` and `test.provenance.jsonl` beside the prepared text splits. These sidecars preserve source names, source URLs, license identifiers, license URLs, dates, languages and attribution requirements. Record and source-group identifiers are SHA-256 fingerprints; URL query strings and fragments are stripped; training text and raw IDs are not copied. Their hashes and counts are recorded in `manifest.json`. Keep the original approved JSONL export privately because the sidecars are not a replacement for the review record.

Examples:
- [OpenAssistant OASST1 dataset card](https://huggingface.co/datasets/OpenAssistant/oasst1) declares Apache-2.0; its individual conversations still require privacy and quality review.
- [OpenStax's official licensing information](https://help.openstax.org/s/article/Licensing-information-of-OpenStax-textbooks) states CC BY-NC-SA. DREAM must not use those textbook records for a potentially commercial model unless OpenStax separately grants the required permission.
- [Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/) permits commercial reuse with attribution; [BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) restricts use to non-commercial purposes and carries ShareAlike conditions.

## DREAM AI opt-in exchange candidates

Optional user contributions are stored in the separate `dream_ai_training_examples` table. v3 submissions explicitly disclose that DREAM and models trained on the submitted exchange may be used in free or paid offerings and commercial services. v2 and earlier contributions remain compatible with ingestion but are excluded from the commercial-ready training exporter unless separate commercial training permission is documented. v2/v3 submissions include a temporary browser-tab session identifier; the trusted Edge Function stores only a domain-separated SHA-256 fingerprint in `metadata.source_group_sha256`. The raw session ID and database row ID are not used as identifiers in the training export.

Use `docs/dream-ai-optin-sft-workflow.md` for the exact sequence: private export of this one DREAM AI table, candidate-only export, four-question manual review, source-group-isolated train/validation split, independent audit and native DREAM SFT. The exporter accepts only v3 consent or a documented separate commercial-training permission, deliberately excludes v1/v2 submissions by default, keeps every candidate approval flag false, hashes database row IDs, and emits only non-sensitive skip counts in its report. The splitter refuses fewer than four independent source groups by default. Before SFT, run the hash-bound audit gate with `data/verify_dream_ai_sft_audits.py`; it refuses stale reports, possible-PII detections, unreviewed rows, parse errors or any unresolved audit finding.

Do not export Dream Education learner tables or expose any of the Dream AI training tables to anonymous/authenticated API roles. A contribution click is not training approval. Do not upload raw database exports, service-role keys or secrets to Kaggle.

## FineWeb2 multilingual candidate import

The official [HuggingFaceFW/FineWeb-2 dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) describes a multilingual Common Crawl-derived collection, its ODC-By 1.0 collection license, source metadata and known limitations. The dataset is very large; do not attempt to download the full corpus to a phone or a small free service. Start with a small, language-balanced local export.

For a small deterministic streaming sample in Colab or another Python environment, first install the optional Hugging Face Datasets dependency and export a few high-confidence Swahili documents (the subset is `swh_Latn`):

```sh
python -m pip install datasets
python data/export_fineweb2_sample.py --subsets swh_Latn --max-per-subset 1000 --scan-per-subset 20000 --output data/fineweb2-small-export.jsonl
```

The exporter pins FineWeb2 revision v2.1.1, streams the dataset, shuffles with a fixed seed, applies a language-confidence threshold, and preserves each document's URL, crawl date, language and script. For a broader multilingual seed, specify verified language-script subsets separated by commas; unavailable subsets fail instead of being silently substituted.

Convert the local JSONL export into review candidates:

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

Use distinct source groups, independently checked answers, and separate training/validation files. Pass the pretraining `manifest.json` and exact held-out `test.txt` file to SFT: it verifies their hash matches and its training and validation source groups must also be disjoint from the pretraining held-out test groups. The SFT checkpoint records the exact held-out test hash, and evaluation refuses any other corpus. SFT also rejects duplicate record IDs and normalized dialogue content within each split and across its training/validation files. The script verifies the review flags and data hashes; it cannot establish the truth of the underlying attestations.

## First target domains

1. General language and instruction following.
2. Mathematics with worked, independently checked solutions.
3. Programming examples with runnable tests and license provenance.
4. Teaching explanations and learner misconceptions.
5. Kiswahili and English, with additional Kenyan languages only where data quality and permission are adequate.
6. Kenyan curriculum and local-context content with authoritative source citations and version dates.

## Important

The files `smoke-test.jsonl`, `sft-smoke-train.jsonl` and `sft-smoke-validation.jsonl` are tiny, synthetic CI fixtures. They exist only to verify that preprocessing, random-initialization pretraining, held-out evaluation, assistant-only SFT and checkpoint loading work end to end. They are not a useful training corpus and must never be presented as model-quality evidence or production intelligence. A capable model requires a much larger, diverse, reviewed corpus and substantial compute.


## OASST1 instruction-tuning candidates

The flat OASST1 ready-message export can also be converted into conversation-shaped SFT candidates. The converter reconstructs each assistant reply's full root-to-leaf path through `parent_id`, requires alternating prompter/assistant roles, and rejects a path if any ancestor fails the conservative review, privacy, deletion or synthetic-content filters.

After downloading the same pinned `2023-04-12_oasst_ready.messages.jsonl.gz` file described in [OASST1.md](OASST1.md), run:

```sh
python data/import_oasst1_sft_candidates.py \
  --input data/2023-04-12_oasst_ready.messages.jsonl.gz \
  --output data/oasst1-sft-candidates.jsonl \
  --report data/oasst1-sft-candidate-report.json
```

The output uses the conversation format expected by `model/sft.py`, but it is deliberately **not approved**: rights, privacy, quality and SFT approval flags remain false. Review every candidate and the upstream terms before approval. All conversations from one message tree share the same `source_group`; split by that group so branches and repeated prefixes from a single conversation tree cannot leak across training and validation. The importer omits OASST user IDs and does not itself split or train the data.


## Manual candidate review

After creating an unapproved candidate JSONL file, use the interactive reviewer before preparing a training corpus:

```sh
python data/review_candidates.py \
  --input data/oasst1-candidates.jsonl \
  --output data/approved-records.jsonl \
  --decisions data/candidate-review-decisions.jsonl \
  --limit 25
```

The reviewer displays each candidate and requires four separate affirmative human decisions: rights for this exact use, privacy inspection, quality inspection, and explicit training approval. Any negative answer rejects the record. There is no automatic approval. Re-run the command to review the next batch; the decision ledger prevents already-reviewed IDs from being prompted again and stores one-way hashes of IDs and decisions, not raw IDs or record text. Approved records are appended without overwriting prior work. Correct invalid provenance before reviewing again.

The reviewer is an aid, not a legal or privacy classifier. The reviewer must inspect the actual source terms and content. The existing corpus audit, provenance checks, deduplication and held-out split rules remain mandatory.

### SFT conversation review and splitting

The same reviewer accepts complete conversation candidates with a `messages` array:

```sh
python data/review_candidates.py \
  --input data/oasst1-sft-candidates.jsonl \
  --output data/approved-sft-records.jsonl \
  --decisions data/sft-review-decisions.jsonl \
  --limit 25
```

After at least two independent source groups have been reviewed and approved, create deterministic train/validation files without splitting a source group across both partitions:

```sh
python data/split_sft.py \
  --input data/approved-sft-records.jsonl \
  --train-out data/sft-train.jsonl \
  --validation-out data/sft-validation.jsonl \
  --manifest data/sft-split-manifest.json \
  --validation-fraction 0.2 --seed 42
```

The splitter applies the validation fraction to distinct source groups (not individual rows). It requires every review flag to be true, validates complete alternating user/assistant conversations, rejects duplicate IDs and duplicate dialogue content, and keeps source groups disjoint. Existing outputs are protected unless `--overwrite` is explicitly supplied. The notebook's later corpus audit and SFT held-out checks remain mandatory.


**Privacy note:** interactive review prints candidate content for human inspection. Clear the review-cell outputs before sharing or saving a notebook copy that should not retain candidate text. The decision ledger itself stores IDs and decisions, not record text.
