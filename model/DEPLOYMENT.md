# Deploy the native DREAM model runtime

This deploys DREAM's own PyTorch checkpoint. It does not connect an outside AI provider. The container in this repository is CPU-first and is intended for the current small research architecture and initial integration testing. It is not a high-throughput production GPU deployment.

## 1. Obtain a trained checkpoint

Train with the Colab or Kaggle notebook and download the artifact ZIP. Extract **only** the chosen checkpoint, normally `checkpoints/dream-bpe-instruct.pt` after instruction tuning. If instruction tuning was skipped, `checkpoints/dream-bpe.pt` is a pretraining checkpoint, not an instruction-tuned assistant. Keep the ZIP and source corpus private; the bundle may contain your training text.

Before serving, run the repository's checkpoint audit and evaluation/release-gate commands from the README. A passing technical gate verifies artifact consistency, not answer quality.

## 2. Build the CPU container

From the repository root on a machine with Docker:

```sh
docker build -f Dockerfile.model-cpu -t dream-native-runtime:local .
```

The image installs CPU-only PyTorch. For larger models or production traffic, build a separate image from a compatible CUDA-enabled PyTorch base on a GPU host; do not assume this CPU image can use CUDA.

## 3. Run it with an external, read-only checkpoint

Create a protected environment file so the secret is not written directly into the Docker command or shell history:

```sh
umask 077
TOKEN="$(openssl rand -hex 32)"
printf '%s\n' \
  'DREAM_CHECKPOINT=/models/dream-instruct.pt' \
  'DREAM_MODEL_NAME=dream-native-v1' \
  'DREAM_RUNTIME_HOST=0.0.0.0' \
  'DREAM_RUNTIME_PORT=8081' \
  'DREAM_CPU_THREADS=2' \
  "DREAM_RUNTIME_TOKEN=$TOKEN" > /absolute/path/dream-runtime.env
unset TOKEN
chmod 600 /absolute/path/dream-runtime.env
```

Then run the container, replacing the checkpoint and environment-file paths:

```sh
docker run --detach --name dream-native \
  --publish 8081:8081 \
  --mount type=bind,src=/absolute/path/dream-bpe-instruct.pt,dst=/models/dream-instruct.pt,readonly \
  --env-file /absolute/path/dream-runtime.env \
  --restart unless-stopped \
  dream-native-runtime:local
```

`DREAM_CPU_THREADS` limits PyTorch CPU parallelism (default 2 in the container); set it to the actual CPU capacity of the host to avoid thread oversubscription. The current runtime processes one generation at a time; excess simultaneous generation requests receive HTTP 503 (`DREAM is busy`) instead of accumulating an unbounded inference queue. The server refuses non-loopback binding without a token of at least 24 characters. Keep the token private and use HTTPS/TLS at the hosting proxy. Do not expose port 8081 directly to the public internet without TLS and access controls.

## 4. Verify the native checkpoint

```sh
curl --fail http://127.0.0.1:8081/health
```

Confirm the response reports `engine: dream-native-transformer`, the expected model name, positive completed training steps, a 64-character checkpoint SHA-256, parameter count, tokenizer and architecture. The endpoint intentionally does not disclose the secret token.

Test a chat completion:

```sh
curl --fail http://127.0.0.1:8081/v1/chat/completions \
  -H 'Authorization: Bearer REPLACE_WITH_RANDOM_SECRET' \
  -H 'Content-Type: application/json' \
  -d '{"model":"dream-native-v1","messages":[{"role":"user","content":"Explain 3x + 5 = 20 step by step in Kiswahili."}],"temperature":0}'
```

## 5. Connect the Render web application

Set these environment variables on the existing DREAM AI web service:

- `DREAM_MODEL_URL=https://YOUR-HTTPS-MODEL-HOST/v1`
- `DREAM_MODEL_NAME=dream-native-v1`
- `DREAM_MODEL_TOKEN=the-same-random-secret`

Use the HTTPS URL of the deployed model service, not the example placeholder. Redeploy the web application after setting the variables. Then open `/api/health`; it should report `modelReady: true` only after the model host passes identity and checkpoint-provenance checks.

The web application rejects remote model URLs that use plain HTTP or contain embedded username/password credentials. HTTPS is mandatory for remote hosts; plain HTTP is permitted only for local loopback development (`localhost`, `127.0.0.1`, or `::1`). Keep the runtime token in the Render environment settings, never in the URL.

The existing Render Node application and native model host are separate services. The free Render web service does not provide a persistent GPU training machine or a durable model checkpoint volume. Do not upload a private training bundle or expose a checkpoint through a public static directory.

## Limitations

- The current native architecture is a small research model. Serving it successfully does not make it competitive with leading general-purpose models.
- CPU inference may be slow under concurrent use. Measure latency and memory before inviting users.
- A real, rights-reviewed corpus, instruction tuning, held-out task benchmarks, safety evaluation and reliable GPU infrastructure remain necessary.
- This guide does not create a trained checkpoint or deploy a model host automatically; it makes the deployment repeatable once suitable compute and a checkpoint exist.
