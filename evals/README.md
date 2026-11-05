# DREAM evaluation datasets

## Current built-in sets

| Dataset | Cases | Coverage | Review status |
|---|---:|---|---|
| `core-v1` | 71 | Reasoning, mathematics, coding, teaching, safety, tool honesty, multilingual prompts and other general capabilities | Rubric-based; outputs require human scoring |
| `african-languages-v1` | 20 | Two cases each for Kiswahili, Yoruba, Hausa, Amharic, Igbo, Somali, Kinyarwanda, Luganda, Zulu and Xhosa | All 20 cases are `pending_native_review` |

The core set also includes two prompts each for Arabic, Hindi, Chinese, Japanese, Korean, French, Spanish, Portuguese, Kiswahili, Yoruba, Hausa, Amharic, Bengali and Russian. These are coverage prompts, not a validated proficiency certification.

## Reproducible evaluation

Run `npm test` to execute the application suite and the structural validators for built-in evaluation datasets. To score a live native-runtime checkpoint, configure `DREAM_MODEL_URL` and `DREAM_MODEL_NAME`, then run `DREAM_EVAL_SET=core-v1 node scripts/run-evals.mjs` or select `african-languages-v1`.

The runner records a UUID, UTC start time, evaluation set, model name, checkpoint SHA-256, training-step count reported by the runtime, tokenizer, architecture, case ID, prompt, rubric, raw output and latency. Human scores are initially null. Score each rubric dimension using a documented scale and retain reviewer qualification and adjudication notes in a separate controlled review record. Do not use model self-scoring as ground truth.

For reproducibility, compare runs only when the evaluation-set revision, checkpoint hash, runtime identity, generation settings and scoring rubric are known. Report per-language results and sample counts; do not hide weak or missing language coverage inside one aggregate score.

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
