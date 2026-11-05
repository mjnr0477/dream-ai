# DREAM model provider configuration

DREAM's own transformer remains the default backend. The provider abstraction also supports an explicitly selected OpenAI-compatible chat-completions endpoint, including a local inference server or a hosted HTTPS endpoint. This is a transport/API compatibility layer; it does not imply that DREAM owns, trains or verifies the weights served by that endpoint.

## Configuration

Required server-side environment variables:

- `DREAM_MODEL_PROVIDER=native` (default) or `openai-compatible`.
- `DREAM_MODEL_URL`: base URL ending in `/v1`, or a full `/chat/completions` URL.
- `DREAM_MODEL_NAME`: exact model identifier sent to the endpoint.
- `DREAM_MODEL_TOKEN`: optional bearer token; keep it in server-side secrets only.
- `DREAM_MODEL_TIMEOUT_MS`: optional integer from 1000 to 120000; defaults to 60000.

Local example:

```sh
DREAM_MODEL_PROVIDER=openai-compatible
DREAM_MODEL_URL=http://127.0.0.1:8000/v1
DREAM_MODEL_NAME=local-instruct
```

Hosted example:

```sh
DREAM_MODEL_PROVIDER=openai-compatible
DREAM_MODEL_URL=https://your-model-host.example/v1
DREAM_MODEL_NAME=your-deployed-model
DREAM_MODEL_TOKEN=<server-side-secret>
```

Remote HTTP, URL-embedded credentials, redirects and unsupported provider names are rejected. HTTP is allowed only for loopback development endpoints. The adapter does not silently switch providers after a failure; callers receive a bounded, actionable error. Native mode retains the existing native-runtime identity and trained-checkpoint verification.

## Model routing and capability claims

Routing is explicit and single-backend in this iteration: `DREAM_MODEL_PROVIDER` selects the configured adapter. No automatic provider fallback or speculative capability inference is performed. Hosted adapter readiness is reported as configured-but-unverified because there is no universal health contract across compatible servers. A successful chat request proves endpoint response only, not model quality or provenance.

## Evaluation and low-cost models

Use the existing native task benchmark and human-scored evaluation workflow for model-quality comparisons. Record exact provider/model ID, endpoint deployment revision, checkpoint hash when available, dataset version, hardware, latency and human rubric scores. Never compare synthetic smoke checkpoints as if they were trained models. Small quantized models can reduce memory and serving cost, but choose one only after measuring task quality, peak RAM/VRAM, tokens/second, first-token latency and concurrency on the actual target hardware.

## Fine-tuning and data controls

The native training and SFT pipeline remains separate from inference adapters. Fine-tuning requires a rights-reviewed, privacy-reviewed, quality-reviewed corpus, source-group-isolated train/validation/test splits, duplicate/leakage checks, reproducible manifests and held-out human evaluation. Provider routing does not grant permission to export prompts or user data to a hosted endpoint; deployment operators must disclose the selected provider and apply their privacy/retention policy.
