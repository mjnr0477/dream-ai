# DREAM native model

Status: first independently designed and randomly initialized architecture implemented. No trained DREAM checkpoint exists yet.

## Architecture decision

DREAM will begin with its own decoder-only transformer implemented in PyTorch:
- fixed UTF-8 byte tokenizer (256 token IDs, no borrowed vocabulary);
- learned token and position embeddings;
- causal multi-head self-attention;
- pre-normalized transformer blocks with GELU feed-forward layers;
- tied token embedding and language-model output weights;
- autoregressive sampling and native checkpoint format.

The first configurable experiment defaults to 6 layers, width 256, 8 attention heads and 256-byte context. This is a small research model, not a competitive production assistant. The architecture and all initial weights are DREAM-owned code/initialization; the checkpoint will only become a trained DREAM artifact after training and evaluation.

## Training and inference

- `model/train.py` trains from random initialization on an authorized UTF-8 text corpus.
- `model/generate.py` runs local inference from a DREAM checkpoint.
- `model/serve.py` exposes the native checkpoint over a local HTTP endpoint. Its chat-completions-shaped JSON is only a transport protocol; no external model is called.
- `model/test_dream_transformer.py` tests tokenizer round-trip, causal attention, tensor shapes, loss and generation.
- `data/README.md` defines dataset provenance, privacy, licensing and quality gates.

Install the model dependencies from `model/requirements.txt`. Run the training script from the repository root with a sufficiently large, high-quality corpus:

```sh
python -m pip install -r model/requirements.txt
python model/train.py --data data/your-approved-corpus.txt --steps 2000
python model/generate.py --checkpoint checkpoints/dream-small.pt --prompt "USER: Explain photosynthesis.\nASSISTANT:"
```

To serve a trained checkpoint locally:

```sh
DREAM_CHECKPOINT=checkpoints/dream-small.pt python model/serve.py
```

The server binds to localhost by default. Set `DREAM_SERVER_TOKEN` before binding to a non-local interface.

## Data and evaluation

No dataset is silently assumed or bundled as training data. Every corpus must have a recorded source, permission/license, privacy review, deduplication, quality checks and leakage-safe train/validation/test splits. See `data/README.md`.

The 12-case initial DREAM evaluation set is a starting point, not enough to certify model quality. We must expand it, add independent human-reviewed scoring and test for memorization, factuality, bias, safety and multilingual performance.

## Compute reality and honest release gates

The free Render web service hosts the Node application; it is not the training or model-serving machine. This first model is intentionally small enough to experiment with on a suitable CPU/GPU machine, but meaningful training still needs adequate RAM, compute time and a substantial authorized corpus.

A successful training run must save:
- model checkpoint and architecture config;
- exact dataset version/hash and split;
- random seed, optimizer, learning rate and step count;
- train and validation loss/perplexity;
- hardware and runtime details;
- evaluation report and known limitations.

Do not call a random-weight model intelligent, do not claim a trained checkpoint before one exists, and do not ship this research-scale model as production-grade without evidence.

## Roadmap

1. Validate the architecture and training loop on a small synthetic smoke test.
2. Build and audit the first licensed DREAM corpus.
3. Train repeatable checkpoints from random initialization.
4. Expand benchmark coverage and human evaluation.
5. Improve tokenizer, context length, data mixture and model scale based on evidence.
6. Add retrieval and tools as separately permissioned capabilities, not as substitutes for model intelligence.
7. Optimize inference and explore practical offline deployments.
