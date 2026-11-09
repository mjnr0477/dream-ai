# DREAM evaluation datasets

## Current built-in sets

Dataset sizes below are verified against the checked-in JSONL files on this branch.

| Dataset | Cases | Coverage | Review status |
|---|---:|---|---|
| `core-v1` | 71 | Reasoning, mathematics, coding, teaching, safety, tool honesty, multilingual prompts and other general capabilities | Rubric-based; outputs require human scoring |
| `african-languages-v1` | 20 | Two cases each for Kiswahili, Yoruba, Hausa, Amharic, Igbo, Somali, Kinyarwanda, Luganda, Zulu and Xhosa | All 20 cases are `pending_native_review` |

The core set also includes two prompts each for Arabic, Hindi, Chinese, Japanese, Korean, French, Spanish, Portuguese, Kiswahili, Yoruba, Hausa, Amharic, Bengali and Russian. These are coverage prompts, not a validated proficiency certification.

## Reproducible evaluation

Run `npm test` to execute the application suite and the structural validators for built-in evaluation datasets. To score a live native-runtime checkpoint, configure `DREAM_MODEL_URL` and `DREAM_MODEL_NAME`, then run `DREAM_EVAL_SET=core-v1 node scripts/run-evals.mjs` or select `african-languages-v1`.

The runner records a UUID, UTC start time, evaluation set, model name, checkpoint SHA-256, training-step count reported by the runtime, tokenizer, architecture, case ID, prompt, rubric, raw output and latency. Human scores are initially null. Score each rubric dimension using a documented scale and retain reviewer qualification and adjudication notes in a separate controlled review record. Do not use model self-scoring as ground truth.

For reproducibility, compare runs only when the evaluation-set revision, checkpoint hash, runtime identity, generation settings and scoring rubric are known. Report per-language results and sample counts; do not hide weak or missing language coverage inside one aggregate score.

## Benchmark contamination preflight

Before using a benchmark to compare a checkpoint, scan every available pretraining and SFT train, validation and test split:

```sh
python model/audit_benchmark_contamination.py \
  --benchmark evals/core-v1.jsonl \
  --corpus data/prepared/train.txt data/prepared/validation.txt data/prepared/test.txt \
          data/sft-train.jsonl data/sft-validation.jsonl \
  --out eval-results/core-v1-contamination.json
```

Repeat with `evals/african-languages-v1.jsonl`. Replace the example corpus paths with the exact files used for the checkpoint. The command checks NFKC-normalized, case-folded, whitespace-normalized prompt, rubric and long reference substrings (minimum 40 characters). It exits non-zero when it finds an overlap and writes only task IDs and SHA-256 fingerprints, not matched training text or raw record IDs. A no-match result is only an exact/normalized overlap screen: it cannot detect paraphrases, shared source material, semantic overlap or memorization. Human source review remains mandatory. This audit does not approve data, change review flags or train a model.

## Translation and comprehension protocol

The existing built-in cases test multilingual response generation and a small number of translation/explanation tasks, but they do not yet constitute a professionally validated translation benchmark. A reliable translation benchmark must use source/target-aligned segments, preserve named entities and numbers, identify direction and locale/script, and be independently translated and adjudicated by qualified bilingual reviewers. Score adequacy, meaning preservation, terminology, fluency and harmful omissions separately. Comprehension cases should test factual extraction, inference and instruction following separately from translation quality.

Do not claim measured translation accuracy from prompts alone. Until aligned gold references and qualified human ratings are available, report the benchmark as draft coverage only.

## Contamination and leakage controls

- Keep benchmark prompts and reference answers out of training corpora and instruction-tuning exports.
- Record benchmark version and SHA-256 digest before each evaluation release.
- Use source and task provenance, not only record IDs, to identify overlap with training material.
- Search exact and normalized text overlaps against every training/validation corpus; investigate paraphrase or source-level overlap manually.
- Keep held-out benchmark sets access-controlled where practical and rotate compromised items.
- Never split near-duplicate items or source families across training and evaluation.
- Record exclusions and contamination findings; do not silently remove difficult items after seeing model results.

## Data and privacy boundaries

Evaluation datasets must be original, explicitly licensed or otherwise authorized for the intended use. Do not import private user conversations, personal records or secrets. Minimize identifiers and keep raw candidate responses in controlled evaluation storage. A clean structural validator does not establish translation correctness, legal rights, consent, factuality or language proficiency.

## Current limitations

- African-language prompts are pending native-speaker review; prompt authorship is not proof of naturalness or correctness.
- Current coverage is small and mostly arithmetic and short conceptual prompts.
- No validated aligned translation gold set or human-scored multilingual comprehension results are claimed here.
- Structural checks validate JSON, required fields, IDs and arithmetic answer keys; they do not validate linguistic quality.
- Benchmark contamination checks require corpus-level overlap scans and human source review before any benchmark can be called uncontaminated.
- Creating or validating a dataset does not train DREAM AI.

## Probabilistic forecast evaluation

DREAM can score a set of resolved forecasts independently of the language model. Each JSONL record represents one event with a mutually exclusive, exhaustive outcome set and one probability per outcome:

```json
{
  "id": "match-2026-11-09-city-united",
  "domain": "sports",
  "model": "DREAM AI",
  "modelVersion": "forecast-engine-v1",
  "horizon": "kickoff-to-full-time",
  "issuedAt": "2026-11-08T12:00:00Z",
  "resolvedAt": "2026-11-09T21:00:00Z",
  "outcomes": ["home", "draw", "away"],
  "probabilities": [0.52, 0.27, 0.21],
  "actualOutcome": "home",
  "baselineProbabilities": [0.44, 0.29, 0.27]
}
```

Use timezone-qualified ISO 8601 timestamps. Probabilities must be finite, between 0 and 1, aligned with outcome labels, and sum to 1 within 1e-6. Baseline probabilities are optional, but must be present for every row or none. Only resolved results belong in a scoring file. Keep one model version (and one checkpoint hash, when available) per scoring run.

Run:

```sh
npm run score:forecasts -- eval-results/resolved-forecasts.jsonl
# Optionally choose the summary destination:
npm run score:forecasts -- eval-results/resolved-forecasts.jsonl eval-results/forecast-summary.json
```

The scorer reports multiclass Brier score (mean sum of squared class errors per event), log loss, tie-aware top-outcome accuracy, expected calibration error, reliability bins, per-domain results, per-horizon results, and optional Brier skill score against a supplied baseline. Lower Brier, log loss and calibration error are better; positive Brier skill means the model beats the stated baseline. It clips zero actual-outcome probabilities at 1e-15 for finite log loss.

This scorer does not create predictions, verify that an event outcome is true, validate the completeness of the outcome set, or prove that a model is calibrated. Forecast records and resolution evidence must be independently maintained and audited. Evaluate against chronological held-out periods, include simple baselines, keep unresolved events out of the score, and report sample counts. Do not promote a forecasting model based on a small or cherry-picked sample.

### Chronological forecast backtesting requirements

- Freeze each probability vector, model/version, checkpoint digest, issue timestamp and forecast horizon before the event resolves. Never revise the stored probability after observing the outcome.
- Use chronological rolling-origin or strictly held-out future periods. At the issue time, exclude future observations, post-event news, final line-ups, closing prices or any other information not yet available.
- Keep event identity unique across partitions. Do not let the same match, weather episode, market interval or near-duplicate event appear in both tuning and evaluation.
- Score each domain and forecast horizon separately before any aggregate. Publish the number of resolved events, missing records, coverage dates and outcome-label definitions alongside every score.
- Compare against predeclared, reproducible baselines appropriate to the domain (for example, historical base rates or a simple persistence forecast). Do not choose a baseline after viewing DREAM's score.
- Assess calibration and discrimination separately. Reliability bins and expected calibration error are sample-sensitive; include bin counts and do not claim calibration from a handful of cases.
- Retain all issued forecasts, including abstentions, failures and low-confidence cases, under a documented inclusion rule. Do not cherry-pick only forecasts that resolved cleanly or agree with the prediction.
- Independent resolution records must cite authoritative outcome evidence and record who or what resolved the event. The scoring tool checks structure and arithmetic; it does not certify the truth of the event result.

A forecast evaluation is a measurement exercise, not a source of new forecasts or evidence that the underlying model is generally intelligent. Only sustained out-of-time performance against strong baselines, with appropriate sample sizes and uncertainty intervals, can support a claim of predictive value.
