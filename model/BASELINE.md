# DREAM native model

Status: independently designed and randomly initialized architecture implemented; end-to-end pretraining, held-out evaluation, assistant-only SFT and SFT evaluation all pass on tiny synthetic CI fixtures. No production-quality or generally capable DREAM checkpoint exists yet.

## Architecture decision

DREAM will begin with its own decoder-only transformer implemented in PyTorch:
- fixed UTF-8 byte tokenizer (256 token IDs, no borrowed vocabulary);
- learned token and position embeddings;
- causal multi-head self-attention;
- pre-normalized transformer blocks with GELU feed-forward layers;
- tied token embedding and language-model output weights;
- autoregressive sampling and native checkpoint format.

The first configurable experiment defaults to 6 layers, width 256, 8 attention heads and 256-byte context. Gradient accumulation allows a larger effective batch without holding every micro-batch in memory at once. This is a small research model, not a competitive production assistant. The architecture and all initial weights are DREAM-owned code/initialization; the checkpoint will only become a trained DREAM artifact after training and evaluation.

## Training and inference

- `model/train.py` trains from random initialization on an authorized UTF-8 text corpus.
- `model/sft.py` performs assistant-response-only supervised instruction fine-tuning from a trained DREAM checkpoint; prompt tokens are masked from the loss.
- `model/generate.py` runs local inference from a DREAM checkpoint.
- `model/evaluate.py` reports held-out cross-entropy, bits-per-byte, perplexity-per-byte, SHA-256 provenance and a qualitative sample.
- `model/serve.py` exposes the native checkpoint over a local HTTP endpoint. Its chat-completions-shaped JSON is only a transport protocol; no external model is called.
- `model/test_dream_transformer.py`, `model/test_train.py` and `model/test_serve.py` test model behavior, checkpoint/resume and local HTTP serving.
- `data/README.md` defines dataset provenance, privacy, licensing and quality gates.

Install the model dependencies from `model/requirements.txt`. Run the training script from the repository root with a sufficiently large, high-quality corpus:

```sh
python -m pip install -r model/requirements.txt
python model/prepare_corpus.py --input data/approved-records.jsonl
python model/train.py --data data/prepared/train.txt --validation-data data/prepared/validation.txt --steps 2000 --eval-batches 10 --batch-size 4 --grad-accum-steps 8
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

## Data and evaluation

No dataset is silently assumed or bundled as training data. Every pretraining and instruction record must carry human-confirmed rights, privacy and quality review flags; the script checks these attestations, deduplicates records and keeps source groups isolated across train/validation/test splits. These flags are not a substitute for actual legal and privacy review. See `data/README.md`.

Instruction data uses JSONL records with the same provenance/review fields plus alternating `messages` entries (`user`, `assistant`; optional initial `system`). SFT masks user/system/prompt tokens and optimizes assistant response tokens only. Keep training and validation conversation files separate by content hash. SFT also rejects source groups present in the pretraining held-out test split, and SFT verifies that manifest against the exact held-out test file, and instruction-checkpoint evaluation must use that exact test-corpus SHA-256.

The 12-case initial DREAM application evaluation set is a starting point, not enough to certify model quality. The native checkpoint evaluator measures next-byte prediction on held-out text; it does not establish reasoning, instruction following, factuality or safety. We must expand it, add independent human-reviewed scoring and test for memorization, factuality, bias, safety and multilingual performance.

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
4. Expand benchmark coverage and human evaluation.
5. Improve tokenizer, context length, data mixture and model scale based on evidence.
6. Add retrieval and tools as separately permissioned capabilities, not as substitutes for model intelligence.
7. Optimize inference and explore practical offline deployments.
