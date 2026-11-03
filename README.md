# DREAM AI

DREAM AI is an independent AI research and application project, separate from Dream Education, Dream Wi-Fi and VYBORA.

## Project status

DREAM has its own decoder-only transformer architecture, byte tokenizer, random initialization, training script, inference script and native model-serving API in `model/`. This is the beginning of a from-scratch model, not a borrowed foundation-model checkpoint.

**No production-trained DREAM checkpoint is being claimed yet.** CI now proves that a tiny model can train from random initialization, complete assistant-only instruction tuning and pass held-out smoke evaluation. Those synthetic tests validate the pipeline only; they do not establish useful model intelligence. The Node web app and model runtime are separate. The Render web service hosts the application; training and inference run on suitable compute that we control.

## Current application

- Responsive mobile-first chat workspace.
- General, coding and learning modes.
- Browser-local conversation history, export and copy.
- Node API with bounded request validation and health status.
- Provider-independent DREAM engine adapter.
- Automated Node tests and original evaluation dataset.

The web interface is not itself a trained model. Until a trained native DREAM checkpoint is available and connected, it must not pretend that it generated intelligent answers.

## Native DREAM model

See [model/BASELINE.md](model/BASELINE.md) for the architecture, commands and release gates. For a phone-friendly, free-tier training workflow, open the [DREAM from-scratch Colab notebook](https://colab.research.google.com/github/mjnr0477/dream-ai/blob/feature/bpe-pipeline-integration/training/dream_train_colab.ipynb). Colab GPU availability and session limits are controlled by Google; download the checkpoint and evaluation report before the session ends.

- `model/dream_transformer.py`: DREAM's own transformer blocks, causal attention and embeddings.
- `model/dream_bpe.py`: deterministic byte-fallback BPE tokenizer (`dream-bpe-v1`) with lossless UTF-8 round trips. Pretraining can opt in with `--tokenizer bpe --tokenizer-vocab-size 2048`; BPE is fitted only on the training split and serialized inside checkpoints. Existing byte checkpoints remain supported.
- `model/benchmark_tokenizer.py`: fits BPE merges on training text and reports token reduction only on a separate held-out corpus, with hashes for both corpora and the tokenizer.
- `model/audit_corpus.py`: performs privacy-conscious pre-ingestion triage for JSONL corpora; it flags metadata gaps, duplicate content and likely contact details without exposing raw text or granting approval.
- `model/train.py`: trains from random initialization.
- `model/evaluate.py`: produces a held-out corpus report with cross-entropy and per-token metrics, corpus/checkpoint hashes and a sample generation; legacy byte checkpoints also retain the original per-byte report keys.
- `model/generate.py`: local checkpoint inference.
- `model/serve.py`: serves DREAM checkpoints through a local HTTP API; the JSON protocol does not call an external AI.
- `model/test_*.py`: tests cover transformer behavior, corpus preparation, training/checkpoint resume, SFT masking and provenance, notebook code cells, and the local authenticated HTTP API. CI also runs a full tiny pretraining → held-out evaluation → SFT → held-out evaluation sequence.
- `data/README.md`: strict data provenance, privacy and licensing rules. SFT rejects source groups that appear in any pretraining split, duplicate record IDs or normalized dialogue content across its train/validation splits; it verifies the manifest against the exact held-out test file, and its evaluator requires that exact test-corpus hash.

Install and run the model locally:

```sh
python -m pip install -r model/requirements.txt
python -m unittest discover -s model -p 'test_*.py'
python model/audit_corpus.py --input data/approved-records.jsonl --out eval-results/corpus-audit.json
python model/prepare_corpus.py --input data/approved-records.jsonl
# Measure tokenizer efficiency on the untouched held-out test split:
python model/benchmark_tokenizer.py --data data/prepared/train.txt --validation-data data/prepared/test.txt --vocab-size 2048
python model/train.py --data data/prepared/train.txt --validation-data data/prepared/validation.txt --manifest data/prepared/manifest.json --steps 2000 --eval-batches 10 --batch-size 4 --grad-accum-steps 8
# Resume an interrupted run from its latest atomic checkpoint:
python model/train.py --data data/prepared/train.txt --validation-data data/prepared/validation.txt --manifest data/prepared/manifest.json --steps 2000 --resume checkpoints/dream-small.pt
python model/evaluate.py --checkpoint checkpoints/dream-small.pt --data data/prepared/test.txt --manifest data/prepared/manifest.json --out eval-results/dream-small.json
# Then instruction-tune on separate, reviewed conversation JSONL files:
python model/sft.py --checkpoint checkpoints/dream-small.pt --train-data data/sft-train.jsonl --validation-data data/sft-validation.jsonl --pretraining-manifest data/prepared/manifest.json --pretraining-test-data data/prepared/test.txt --steps 1000 --batch-size 2 --grad-accum-steps 4 --out checkpoints/dream-instruct.pt
python model/generate.py --checkpoint checkpoints/dream-small.pt --prompt "USER: Hello DREAM.
ASSISTANT:"
```

The training corpus path above is an example, not a file already present. Each JSONL record must include `source_date` in `YYYY-MM-DD` form and contain `rights_confirmed: true`, `privacy_reviewed: true` and `quality_reviewed: true` after actual human review. The corpus manifest reports license/language distributions and source-date coverage. The script verifies these declarations, not the underlying legal claims. Do not train on data without appropriate rights and privacy review. Do not market a small research checkpoint as a frontier model.

## Evaluation

- `evals/core-v1.jsonl`: 40 original DREAM evaluation prompts and human scoring rubrics spanning reasoning, coding, science, multilingual performance, safety, privacy, security and reliability.
- `npm run validate:evals`: validates the evaluation set.
- `npm run eval`: records outputs and latency from a configured self-hosted model runtime for human review.

For native DREAM checkpoint evaluation, run `python model/run_task_benchmark.py --checkpoint checkpoints/dream-instruct.pt --out eval-results/native-task-benchmark.json`, review every generated answer against its rubric, and fill each `rubric_scores` entry with `0` or `1`. Then run `python model/score_task_benchmark.py --input eval-results/native-task-benchmark.json --out eval-results/human-score-summary.json` to calculate overall and per-category scores plus scoring coverage. The scorer rejects malformed or incomplete score values and never grades answers automatically; unscored rubric items remain visible in coverage.

The current benchmark is an initial seed, not a certification of quality. Provenance-verified training checkpoints require `model/evaluate.py --manifest` to match the exact held-out test file and original manifest; the report records the manifest hash and held-out source-group count.

## Runtime architecture

The Node app connects to a DREAM model server through the independent adapter:
- `DREAM_MODEL_URL`: native server API base, including the `/v1` path (for example `http://127.0.0.1:8090/v1` for a same-machine setup or `https://your-private-model-host/v1` for a secured remote runtime).
- `DREAM_MODEL_NAME`: model identifier, currently `dream-native`.
- `DREAM_MODEL_TOKEN`: optional private bearer token; required when exposing the model server beyond localhost.

The native server defaults to `127.0.0.1:8081` and exposes `GET /health` plus `POST /v1/chat/completions`. It caps active HTTP workers, serializes inference, validates request size/history/sampling settings, and refuses checkpoints without completed training steps. Native generation uses per-layer key/value caching while the context window is stable, then rebuilds the cache when the learned absolute-position window slides. For a same-machine setup, run `python model/serve.py`, then configure the Node app with `DREAM_MODEL_URL=http://127.0.0.1:8081/v1`, `DREAM_MODEL_NAME=dream-native`, and `DREAM_MODEL_TOKEN` matching `DREAM_RUNTIME_TOKEN`. Set `DREAM_RUNTIME_HOST`, `DREAM_RUNTIME_PORT`, and a `DREAM_RUNTIME_TOKEN` of at least 24 characters before binding beyond localhost. The Node adapter permits plain HTTP only for loopback model hosts; remote model URLs must use HTTPS so bearer credentials are not sent in cleartext. The built-in HTTP server is a development/runtime baseline, not a hardened public deployment: use HTTPS, a reverse proxy, network restrictions and resource limits before remote exposure. Never put tokens in GitHub or browser JavaScript.

## Compute and deployment

Render's free web service is not a GPU training environment. Keep the Node app on Render and run model training/inference on a separate machine or host that we control. A future deployment must be based on measured memory, latency and availability, not a claim that Render free can train a large model.

## API

- `GET /api/health`
- `POST /api/chat`
- Supported modes: `general`, `coding`, `learning`

## Security principles

1. Keep DREAM independent from other projects and external AI providers.
2. Keep model credentials server-side.
3. Validate requests and bound resource usage.
4. Isolate any future code execution and require explicit permissions for tools.
5. Never claim a model is trained or an action was completed without evidence.
