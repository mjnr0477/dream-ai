# DREAM AI

DREAM AI is an independent AI research and application project, separate from Dream Education, Dream Wi-Fi and VYBORA.

## Project status

DREAM has its own decoder-only transformer architecture, byte-tokenizer default, optional byte-fallback BPE tokenizer, random initialization, training script, inference script and native model-serving API in `model/`. This is the beginning of a from-scratch model, not a borrowed foundation-model checkpoint.

**No production-trained DREAM checkpoint is being claimed yet.** CI now proves that a tiny model can train from random initialization, complete assistant-only instruction tuning and pass held-out smoke evaluation. Those synthetic tests validate the pipeline only; they do not establish useful model intelligence. The Node web app and model runtime are separate. The Render web service hosts the application; training and inference run on suitable compute that we control.

## Current application

- Responsive mobile-first chat workspace.
- General, coding and learning modes.
- Browser-local conversation history, export and copy.
- Node API with bounded request validation, two-request model concurrency protection with retry guidance, and health status.
- Provider-independent DREAM engine adapter.
- Automated Node tests and original evaluation dataset.

The web interface is not itself a trained model. Until a trained native DREAM checkpoint is available and connected, it must not pretend that it generated intelligent answers.

## Task-level evaluation

After training an instruction checkpoint, run the small authored diagnostic suite:

```sh
python model/evaluate_tasks.py \
  --checkpoint checkpoints/dream-bpe-instruct.pt \
  --benchmark evals/dream-core-v1.jsonl \
  --out eval-results/dream-core-v1.json
```

The report includes deterministic exact-match results by category and language, model/checkpoint hashes and every generated answer. This 15-task starter set is a debugging aid, not a comprehensive reasoning, factuality, safety or multilingual benchmark; its score cannot establish production readiness or superiority over other models. Expand it with independently authored and reviewed tests before making capability claims.

## Native DREAM model

See [model/BASELINE.md](model/BASELINE.md) for the architecture, commands and release gates. For free-tier training, use the [DREAM from-scratch Colab notebook](https://colab.research.google.com/github/mjnr0477/dream-ai/blob/main/training/dream_train_colab.ipynb) or the [Kaggle training notebook](https://github.com/mjnr0477/dream-ai/blob/main/training/dream_train_kaggle.ipynb). Both require a real, reviewed corpus; GPU access is not guaranteed. Download the checkpoint and evaluation report before the session ends.

- `model/dream_transformer.py`: DREAM's own transformer blocks, causal attention, embeddings, UTF-8 byte tokenizer, inference-time key/value caching when the sequence fits the context window, and UTF-8-constrained output-byte sampling.
- `model/dream_bpe.py`: deterministic byte-fallback BPE tokenizer, integrated with native pretraining, SFT, checkpoint auditing, evaluation, generation and serving. Existing byte-tokenizer checkpoints remain supported.
- `model/benchmark_multilingual_tokenizer.py`: measures BPE token efficiency per language on held-out UTF-8 text, checks train/test text overlap and verifies exact Unicode round trips. This measures compression only, not language understanding.
- `model/train.py`: trains from random initialization and optionally uses activation checkpointing to reduce GPU memory at the cost of extra recomputation; the setting is recorded and enforced on resume.
- `model/evaluate.py`: reconstructs the checkpoint tokenizer and reports held-out cross-entropy per token and per represented UTF-8 byte, bits-per-byte, perplexity-per-byte, corpus/checkpoint hashes and a sample generation.
- `model/evaluate_multilingual.py`: verifies manifest hashes, source-group isolation and declared per-language train/validation/test coverage, then evaluates each language-specific held-out test file independently for pretrained or SFT checkpoints. Reports whether each language has at least three independent source groups and coverage in all three splits; legacy manifests are explicitly marked as lacking this evidence. No pooled score is emitted.
- `model/generate.py`: local checkpoint inference.
- `model/benchmark_generation.py`: compares cached greedy generation with exact full-prefix recomputation, verifies identical token output, and reports local median latency and throughput. The result is hardware/checkpoint-specific, not a model-quality score.
- `model/audit_checkpoint.py`: read-only preflight audit for checkpoint tokenizer/configuration, trained-step metadata, exact tensor keys/shapes/dtypes, finite weights, strict state loading and SHA-256 identity.
- `model/serve.py`: serves DREAM checkpoints through a local HTTP API; `/health` reports the checkpoint SHA-256, completed training steps, tokenizer and architecture so benchmark results can be tied to the exact native model; the JSON protocol does not call an external AI.
- `Dockerfile.model-cpu` and `model/DEPLOYMENT.md`: reproducible CPU container and secure instructions for connecting a trained native checkpoint to the Render web application. This is an initial integration path, not a GPU-scale production deployment.
- `training/dream_train_kaggle.ipynb`: Kaggle notebook workflow for the same native from-scratch training pipeline. Attach a private Kaggle Dataset containing the reviewed corpus as an input; checkpoints and the artifact ZIP are written under `/kaggle/working`. Kaggle's free GPU availability is limited and may queue; see the [Kaggle notebook guide](https://www.kaggle.com/docs/notebooks) and [GPU usage guidance](https://www.kaggle.com/docs/efficient-gpu-usage).
- `scripts/run-evals.mjs`: verifies native runtime identity and checkpoint provenance before running the human-reviewed benchmark, then records a run ID, checkpoint hash, training steps, architecture, tokenizer and latency with every result. The summarizer refuses to combine rows from different runs or checkpoints.
- `model/test_*.py`: tests cover transformer behavior (including cached-vs-full generation equivalence, context-window fallback and strict UTF-8 output validation), corpus preparation, training/checkpoint resume, SFT masking and provenance, notebook code cells, and the local authenticated HTTP API. CI also runs a full tiny pretraining → held-out evaluation → SFT → held-out evaluation sequence.
- `data/export_fineweb2_sample.py` and `model/import_fineweb2_candidates.py`: deterministic, confidence-filtered FineWeb2 streaming export with provenance-preserving, unapproved candidate conversion; source-page rights still require review.
- `data/import_oasst1_candidates.py` and `data/OASST1.md`: optional pinned OASST1 pretraining-text candidate importer; filters deleted, synthetic, unreviewed and flagged records, omits user IDs, and keeps all DREAM approval flags false until human review.
- `data/import_oasst1_sft_candidates.py`: reconstructs complete OASST1 conversation paths for SFT candidates, applies the safety filter to every ancestor, preserves tree-level split groups, omits user IDs and never pre-approves records. OASST1 does not include Swahili and is only one possible seed source.
- `data/README.md`: strict data provenance, privacy and licensing rules. Corpus manifests expose domain-separated SHA-256 fingerprints instead of raw record/source-group identifiers, and SFT uses those fingerprints for leakage checks while retaining compatibility with legacy manifests. These are pseudonymous identifiers, not anonymization; never place secrets or personal data in ID fields. SFT rejects source groups that appear in the pretraining held-out test split, duplicate record IDs or normalized dialogue content across its train/validation splits; it verifies the manifest against the exact held-out test file, and its evaluator requires that exact test-corpus hash.

Install and run the model locally:

```sh
python -m pip install -r model/requirements.txt
python -m unittest discover -s model -p 'test_*.py'
python model/prepare_corpus.py --input data/approved-records.jsonl
python model/train.py --data data/prepared/train.txt --validation-data data/prepared/validation.txt --steps 2000 --eval-batches 10 --batch-size 4 --grad-accum-steps 8 --gradient-checkpointing
# Resume an interrupted run from its latest atomic checkpoint:
python model/train.py --data data/prepared/train.txt --validation-data data/prepared/validation.txt --steps 2000 --resume checkpoints/dream-small.pt
python model/audit_checkpoint.py --checkpoint checkpoints/dream-small.pt --out eval-results/checkpoint-audit.json
python model/evaluate.py --checkpoint checkpoints/dream-small.pt --data data/prepared/test.txt --out eval-results/dream-small.json
# Report held-out metrics separately for every language in the corpus manifest:
python model/evaluate_multilingual.py --checkpoint checkpoints/dream-small.pt --manifest data/prepared/manifest.json --out eval-results/dream-multilingual.json
# Then instruction-tune on separate, reviewed conversation JSONL files:
python model/sft.py --checkpoint checkpoints/dream-small.pt --train-data data/sft-train.jsonl --validation-data data/sft-validation.jsonl --pretraining-manifest data/prepared/manifest.json --pretraining-test-data data/prepared/test.txt --steps 1000 --batch-size 2 --grad-accum-steps 4 --out checkpoints/dream-instruct.pt
python model/generate.py --checkpoint checkpoints/dream-small.pt --prompt "USER: Hello DREAM.\nASSISTANT:"
```

The training corpus path above is an example, not a file already present. Each JSONL record must include `source_date` in `YYYY-MM-DD` form and contain `rights_confirmed: true`, `privacy_reviewed: true` and `quality_reviewed: true` after actual human review. The corpus manifest reports license/language distributions, source-date coverage and per-language source-group counts by split. Splitting is deterministic and source-group isolated; any language represented by at least three independent source groups is guaranteed to appear in train, validation and test. Languages with fewer than three independent groups are explicitly marked as not yet eligible for three-way evaluation. The script verifies declarations, not underlying legal claims. Do not train on data without appropriate rights and privacy review. Do not market a small research checkpoint as a frontier model.

## Evaluation

- `evals/core-v1.jsonl`: 59 original DREAM evaluation prompts and human scoring rubrics. The suite includes 28 explicit multilingual cases: two independent tasks each in Arabic, Hindi, Chinese, Japanese, Korean, French, Spanish, Portuguese, Kiswahili, Yoruba, Hausa, Amharic, Bengali and Russian. The validator enforces at least two cases per language so language-level averages are not based on a single prompt, alongside reasoning, coding, science, safety, privacy, security and reliability.
- `npm run validate:evals`: validates the evaluation set.
- `npm run eval`: records outputs and latency from a configured self-hosted model runtime for human review.
- Human reviewers score each result from 0 to 4 using its rubric, filling `humanScore` and `reviewerNotes` in the JSONL file. `npm run summarize:evals -- eval-results/scored.jsonl eval-results/summary.json` reports honest overall, category and language averages, latency, and unscored counts; missing scores are never inferred.

The current benchmark is an initial seed, not a certification of quality. For held-out BPE efficiency experiments, prepare separate JSONL files containing `language` and `text` fields, then run `python model/benchmark_multilingual_tokenizer.py --train-data data/tokenizer-train.jsonl --test-data data/tokenizer-test.jsonl`. The test text must not overlap the tokenizer-training text. This experiment does not change the active byte-tokenizer model.

## Local parallel-corpus candidate import

For a local translation export with a JSON object per line and a nested `translation` object, the candidate importer can create deterministic SFT review files. It does not fetch datasets or approve them:

```sh
python model/import_parallel_pairs.py \
  --input data/local-parallel-export.jsonl \
  --source-lang dav --target-lang swa \
  --source-name "Dataset name" \
  --source-url https://example.org/dataset \
  --license "claimed license label" \
  --license-url https://example.org/license \
  --source-date 2025-01-15 \
  --output-dir data/parallel-candidates
```

Each input record must contain `translation.dav` and `translation.swa`, plus a genuine `source_group` or `story_id`. The importer refuses to invent independent groups from pair hashes. External group identifiers are hashed before writing candidate files; pair IDs are generated from content. The outputs are `sft-train-candidates.jsonl`, `sft-validation-candidates.jsonl` and `import-summary.json`.

All review flags remain false and `approval_granted` remains false. Source names, URLs, dates and license labels are caller-supplied claims, not verified facts; `rights_basis` is explicitly `unverified`. A human must independently establish licensing, attribution, provenance, consent, privacy and translation quality before editing any review flags. Run the corpus audit after import. This workflow creates candidates only and does not train the model.

## Runtime architecture

The Node app connects to a DREAM model server through the independent adapter:
- `DREAM_MODEL_URL`: native server API base, including the `/v1` path (for example `http://127.0.0.1:8090/v1` for a same-machine setup or `https://your-private-model-host/v1` for a secured remote runtime).
- `DREAM_MODEL_NAME`: model identifier, currently `dream-native`.
- `DREAM_MODEL_TOKEN`: optional private bearer token; required when exposing the model server beyond localhost.

The native server defaults to `127.0.0.1:8081` and exposes `GET /health` plus `POST /v1/chat/completions`. It serializes inference to bound concurrency, validates request size/history/sampling settings, and refuses checkpoints without completed training steps. For a same-machine setup, run `python model/serve.py`, then configure the Node app with `DREAM_MODEL_URL=http://127.0.0.1:8081/v1`, `DREAM_MODEL_NAME=dream-native`, and `DREAM_MODEL_TOKEN` matching `DREAM_RUNTIME_TOKEN`. Set `DREAM_RUNTIME_HOST`, `DREAM_RUNTIME_PORT`, and a `DREAM_RUNTIME_TOKEN` of at least 24 characters before binding beyond localhost. The built-in HTTP server is a development/runtime baseline, not a hardened public deployment: use HTTPS, a reverse proxy, network restrictions and resource limits before remote exposure. Never put tokens in GitHub or browser JavaScript.

## Compute and deployment

Render's free web service is not a GPU training environment. Keep the Node app on Render and run model training/inference on a separate machine or host that we control. A future deployment must be based on measured memory, latency and availability, not a claim that Render free can train a large model.

### Run the free Kaggle training workflow

1. Download `training/dream_train_kaggle.ipynb` from this repository and import it as a Kaggle notebook.
2. In Notebook Settings, select a GPU accelerator and enable Internet. The notebook stops immediately if CUDA is unavailable.
3. Create a **private** Kaggle Dataset containing the human-reviewed `approved-records.jsonl`, then attach it with **Add Input**.
4. Run the notebook cells in order. For optional instruction tuning, attach a private dataset containing both `sft-train.jsonl` and `sft-validation.jsonl`.
5. Save a notebook version and download `dream-training-artifacts.zip` from the notebook Output panel. Keep it private if it contains reviewed training text.

Kaggle documents free P100 notebook accelerators, but GPU access can be limited or queued and weekly usage quotas vary. See the [notebook guide](https://www.kaggle.com/docs/notebooks) and [GPU usage limits](https://www.kaggle.com/docs/efficient-gpu-usage).

## API

- `GET /api/health/live` is a fast process-only liveness probe and does not depend on model availability. It is suitable for the hosting platform health check.
- `GET /api/health` reports application availability separately from model configuration and live native-checkpoint readiness. A ready runtime must identify as the native DREAM transformer and report positive completed training steps plus a 64-character checkpoint SHA-256. This is a runtime/provenance check, not a quality certification.
- `GET /api/capabilities` returns a versioned, server-owned capability contract. It distinguishes implemented application features from tools that are not configured or not implemented; it never advertises disabled tools as ready. Tool execution is disabled by default and external actions require approval.
- `POST /api/chat`
- Supported modes: `general`, `coding`, `learning`

## Security principles

1. Keep DREAM independent from other projects and external AI providers.
2. Keep model credentials server-side.
3. Validate requests and bound resource usage.
4. Isolate any future code execution and require explicit permissions for tools.
5. Never claim a model is trained or an action was completed without evidence.
6. Treat external content as untrusted, use least-privilege tool access, and require explicit confirmation for destructive or external actions.

## African-language evaluation expansion

`evals/african-languages-v1.jsonl` adds 20 targeted evaluation prompts across Kiswahili, Yoruba, Hausa, Amharic, Igbo, Somali, Kinyarwanda, Luganda, Zulu and Xhosa. Each language has a multi-step arithmetic task and a concept-explanation task, so the set checks more than simple translation.

Run its structural checks with:

```sh
npm run validate:african-evals
```

All cases are explicitly marked `pending_native_review`. The prompts and rubrics are engineering drafts, not certified translations. Before using this set to make language-quality claims or as supervised training data, have qualified native speakers review and correct every prompt and rubric, record the review provenance, and score model outputs independently. A passing validator proves only dataset structure and coverage; it does not prove model fluency or competence.

To run this suite against a configured, provenance-verified native DREAM runtime and save outputs for human scoring:

```sh
DREAM_EVAL_SET=african-languages-v1 npm run eval
```

The runner accepts only the named built-in evaluation sets; it does not accept arbitrary file paths. Its output remains unscored until a human reviewer completes the rubrics.
