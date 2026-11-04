# DREAM native model

Status: independently designed and randomly initialized architecture implemented; end-to-end pretraining, held-out evaluation, assistant-only SFT and SFT evaluation all pass on tiny synthetic CI fixtures. No production-quality or generally capable DREAM checkpoint exists yet.

## Architecture decision

DREAM will begin with its own decoder-only transformer implemented in PyTorch:
- UTF-8 byte tokenizer (256 token IDs) remains the backwards-compatible default; an optional deterministic byte-fallback BPE tokenizer can be trained from the training split and saved inside checkpoints;
- learned token and position embeddings;
- causal multi-head self-attention;
- pre-normalized transformer blocks with GELU feed-forward layers;
- tied token embedding and language-model output weights;
- autoregressive sampling, an inference-only per-layer key/value cache for one-token decoding, UTF-8-constrained byte sampling, and native checkpoint format.

The first configurable experiment defaults to 6 layers, width 256, 8 attention heads and 256-byte context. Gradient accumulation allows a larger effective batch without holding every micro-batch in memory at once. This is a small research model, not a competitive production assistant. The architecture and all initial weights are DREAM-owned code/initialization; the checkpoint will only become a trained DREAM artifact after training and evaluation.

## Training and inference

- `model/train.py` trains from random initialization on an authorized UTF-8 text corpus.
- `model/sft.py` performs assistant-response-only supervised instruction fine-tuning from a trained DREAM checkpoint; prompt tokens are masked from the loss.
- `model/generate.py` runs local inference from a DREAM checkpoint. Generation uses per-layer key/value caching when the prompt plus requested output fits the learned absolute-position context; if it would overflow, it retains the existing sliding-window recomputation path so positional behavior is unchanged. Sampling constrains both byte-token and BPE-token output to valid UTF-8 sequences (including continuation-byte ranges that exclude overlong encodings, surrogates and values above U+10FFFF); this guarantees valid Unicode encoding, not language fluency or factual quality.
- `model/benchmark_generation.py` compares cached greedy decoding against a full-prefix recomputation reference, first requiring exact token-ID parity, then reporting median latency, generated tokens/second and speedup ratio for the selected checkpoint and device. Run it only when the requested output fits inside the configured context window; results are local performance measurements, not model-quality evidence.
- `model/estimate_model.py` reports the exact parameter count for the current tied-embedding architecture, FP32/FP16 parameter storage, and an explicitly labeled FP32 Adam training-memory floor. It excludes activations, CUDA workspaces and allocator overhead, so it is a planning lower bound rather than a VRAM guarantee. Run `python model/estimate_model.py --out eval-results/model-budget.json`; override `--vocab-size`, `--block-size`, `--d-model`, `--heads` and `--layers` to compare configurations. Its tests cross-check the formula against an instantiated model.
- `model/audit_checkpoint.py` performs a read-only checkpoint preflight: validates the byte tokenizer, model config, positive training-step metadata, exact state-dict keys/shapes/dtypes, finite floating-point weights and strict model loading. It emits a SHA-256 identity report and does not prove that training data, rights attestations or model quality are valid.
- `model/evaluate.py` reconstructs the tokenizer from the checkpoint and reports held-out cross-entropy per token and per represented UTF-8 byte, bits-per-byte, perplexity-per-byte, SHA-256 provenance and a qualitative sample. BPE byte-normalized loss divides summed token NLL by the actual UTF-8 bytes represented by sampled target tokens.
- `model/serve.py` exposes the native checkpoint over a local HTTP endpoint. Its chat-completions-shaped JSON is only a transport protocol; no external model is called.
- `model/test_dream_transformer.py`, `model/test_train.py` and `model/test_serve.py` test model behavior (including cached/full-logit equivalence, greedy output parity, context-window fallback and valid UTF-8 output constraints), checkpoint/resume and local HTTP serving.
- `data/README.md` defines dataset provenance, privacy, licensing and quality gates.

Install the model dependencies from `model/requirements.txt`. Run the training script from the repository root with a sufficiently large, high-quality corpus:

```sh
python -m pip install -r model/requirements.txt
python model/prepare_corpus.py --input data/approved-records.jsonl
python model/train.py --data data/prepared/train.txt --validation-data data/prepared/validation.txt --steps 2000 --eval-batches 10 --batch-size 4 --grad-accum-steps 8
python model/audit_checkpoint.py --checkpoint checkpoints/dream-small.pt --out eval-results/checkpoint-audit.json
# Resume the saved run later, keeping the same corpus, architecture and optimizer settings:
python model/train.py --data data/prepared/train.txt --validation-data data/prepared/validation.txt --steps 2000 --resume checkpoints/dream-small.pt
python model/evaluate.py --checkpoint checkpoints/dream-small.pt --data data/prepared/test.txt --out eval-results/dream-small.json
# After pretraining, prepare separate reviewed train/validation conversation JSONL files:
python model/sft.py --checkpoint checkpoints/dream-small.pt --train-data data/sft-train.jsonl --validation-data data/sft-validation.jsonl --pretraining-manifest data/prepared/manifest.json --pretraining-test-data data/prepared/test.txt --steps 1000 --batch-size 2 --grad-accum-steps 4 --out checkpoints/dream-instruct.pt
python model/evaluate.py --checkpoint checkpoints/dream-instruct.pt --data data/prepared/test.txt --out eval-results/dream-instruct.json
python model/generate.py --checkpoint checkpoints/dream-small.pt --prompt "USER: Explain photosynthesis.\nASSISTANT:"
```

To serve a trained checkpoint locally:

```sh
DREAM_CHECKPOINT=checkpoints/dream-instruct.pt python model/serve.py
# In the Node app, set DREAM_MODEL_URL=http://127.0.0.1:8081/v1,
# DREAM_MODEL_NAME=dream-native and DREAM_MODEL_TOKEN to the same runtime token.
```

The server binds to localhost by default, serializes model generation to bound inference concurrency, enforces request-size, history, authentication and sampling limits, and refuses checkpoints without completed training steps. Set `DREAM_RUNTIME_HOST`, `DREAM_RUNTIME_PORT` and a `DREAM_RUNTIME_TOKEN` of at least 24 characters before binding to a non-local interface. The Node adapter's `DREAM_MODEL_TOKEN` must match that token. The built-in server is a development/runtime baseline, not a hardened public internet deployment.

## Multilingual BPE tokenizer and checkpoint compatibility

`model/dream_bpe.py` implements a deterministic byte-fallback BPE tokenizer. Its trainer maintains weighted adjacent-pair counts and an inverted sequence index so each merge updates only affected unique text pieces instead of rescanning every piece. It can now be used end-to-end in native DREAM pretraining, supervised fine-tuning, checkpoint auditing, evaluation, local generation and serving. Existing `utf8-byte-v1` checkpoints and the default byte-tokenizer training path remain compatible.

Train a BPE checkpoint by selecting the tokenizer explicitly:

```sh
python model/train.py --data data/prepared/train.txt --validation-data data/prepared/validation.txt --tokenizer bpe --vocab-size 1024 --steps 2000 --out checkpoints/dream-bpe.pt
python model/audit_checkpoint.py --checkpoint checkpoints/dream-bpe.pt
python model/evaluate.py --checkpoint checkpoints/dream-bpe.pt --data data/prepared/test.txt --out eval-results/dream-bpe.json
python model/generate.py --checkpoint checkpoints/dream-bpe.pt --prompt "USER: Habari dunia.\\nASSISTANT:"
python model/serve.py --checkpoint checkpoints/dream-bpe.pt
```

The tokenizer is trained only from the training text, embedded in the checkpoint, and checked on resume. SFT restores the base checkpoint tokenizer and carries its exact metadata forward. Evaluation and serving reconstruct the same tokenizer from the checkpoint; evaluation reports both token-normalized and UTF-8-byte-normalized cross-entropy. The tokenizer uses byte fallback for unseen scripts, but compression/token reduction is not proof of language fluency or model quality. Use a separate reviewed held-out test corpus and native-speaker evaluation before making multilingual capability claims.

## Data and evaluation

No dataset is silently assumed or bundled as training data. Every pretraining and instruction record must carry a strict ISO `source_date` (`YYYY-MM-DD`) and human-confirmed rights, privacy and quality review flags; the script checks these attestations, deduplicates records and keeps source groups isolated across train/validation/test splits. The manifest records each split's exact record IDs, source-group IDs, UTF-8 byte count, language/license coverage, source-date range and content hash, in addition to corpus-wide coverage. These flags are not a substitute for actual legal and privacy review. See `data/README.md`.

Instruction data uses JSONL records with the same provenance/review fields plus alternating `messages` entries (`user`, `assistant`; optional initial `system`). SFT masks user/system/prompt tokens and optimizes assistant response tokens only. Keep training and validation conversation files separate by content hash. SFT rejects source groups present in the pretraining held-out test split, duplicate record IDs and normalized dialogue content across SFT train/validation files. It verifies the manifest against the exact held-out test file, and instruction-checkpoint evaluation must use that exact test-corpus SHA-256.

The 45-case DREAM application evaluation set covers reasoning, mathematics, coding, teaching, Kenyan context, English/Kiswahili, factual calibration, uncertainty, context-following, safety, security, privacy, fairness, research methods and structured output. Fourteen tagged language prompts explicitly cover Arabic, Hindi, Chinese, Japanese, Korean, French, Spanish, Portuguese, Kiswahili, Yoruba, Hausa, Amharic, Bengali and Russian. Tokenizer round-trip tests also cover these writing systems and emoji. This remains a starter benchmark, not enough to certify model quality. The native checkpoint evaluator measures next-token prediction and reports token/UTF-8-byte-normalized cross-entropy on held-out text; it does not establish reasoning, instruction following, factuality or safety. The next requirement is a reviewed multilingual training corpus, independent native-speaker scoring and tests for memorization, factuality, bias and safety.

## Compute reality and honest release gates

The free Render web service hosts the Node application; it is not the training or model-serving machine. This first model is intentionally small enough to experiment with on a suitable CPU/GPU machine, but meaningful training still needs adequate RAM, compute time and a substantial authorized corpus.

A successful training run must save:
- model checkpoint and architecture config;
- exact dataset version/hash and split;
- random seed, optimizer, learning rate, micro-batch size, gradient accumulation, effective batch size, evaluation batch count and step count;
- train and validation loss/perplexity;
- hardware and runtime details;
- evaluation report and known limitations.

Do not call a random-weight model intelligent, do not claim a trained checkpoint before one exists, and do not ship this research-scale model as production-grade without evidence.

## Roadmap

1. [x] Validate the architecture, preprocessing, random-initialization training, held-out evaluation, assistant-only SFT and checkpoint loading on synthetic CI smoke tests.
2. Build and audit the first licensed DREAM corpus.
3. Train repeatable checkpoints from random initialization.
4. [x] Expand the starter benchmark with 14 tagged multilingual cases and Unicode tokenizer tests; obtain independent human evaluation.
5. Improve tokenizer, context length, data mixture and model scale based on evidence.
6. Add retrieval and tools as separately permissioned capabilities, not as substitutes for model intelligence.
7. Optimize inference and explore practical offline deployments.
