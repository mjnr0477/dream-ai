# Prepare the one private Kaggle dataset for DREAM

The Kaggle training notebook expects exactly one attached Kaggle Dataset with a file named approved-records.jsonl. This guide prepares that upload directory without inventing training data or bypassing human approval.

## 1. Create review candidates (not approved data)

Use the existing FineWeb2 candidate workflow in data/README.md. The FineWeb2 exporter and importer deliberately leave all human-review flags false. Candidate data is not training-ready. Review source-level terms, the underlying page rights, opt-outs, privacy/personal data and content quality.

## 2. Complete the human review

Run the existing interactive reviewer against the candidate JSONL. It asks separately about rights, privacy, quality and explicit approval for every record. Only records receiving four affirmative decisions are written to the approved output.

```sh
python data/review_candidates.py \
  --input data/fineweb2-candidates.jsonl \
  --output data/approved-records.jsonl \
  --decisions data/candidate-review-decisions.jsonl \
  --limit 25
```

Repeat in batches until enough eligible records from at least three independent source groups have been reviewed. Never bulk-set review flags to true.

## 3. Build the Kaggle upload directory

From the repository root:

```sh
python data/package_kaggle_dataset.py \
  --input data/approved-records.jsonl \
  --output build/dream-kaggle-dataset
```

The packager calls the existing canonical model/prepare_corpus.py validator. It refuses missing metadata, non-HTTPS provenance URLs, invalid dates or rights basis, any false/missing review or approval flag, duplicate IDs, insufficient independent source groups, and unsafe split eligibility. It does not delete non-empty output directories.

The resulting directory contains approved-records.jsonl (the exact file required by the Kaggle notebook), dataset-manifest.json (SHA-256 and validated corpus summary), and README.md (privacy and validation notes).

## 4. Upload privately to Kaggle

Create a Kaggle Dataset, set its visibility to Private, upload the contents of build/dream-kaggle-dataset, and attach that one Dataset to the notebook with Add Input. The notebook will still run its own corpus privacy and provenance audit before training.

## Important boundary

No script can truthfully perform a human rights, privacy or quality review on your behalf. A valid JSON structure and true flags are not proof that the underlying attestations are accurate. FineWeb2 is Common Crawl-derived; its collection licence does not automatically grant rights to every underlying webpage. Do not train on records unless the actual source terms and content have been checked for the intended use.
