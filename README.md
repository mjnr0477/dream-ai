# DREAM AI

DREAM AI is an independent AI assistant project, deployed separately from Dream Education, Dream Wi-Fi and VYBORA.

## Live service

https://dream-ai-live.onrender.com

## Project ownership and model status

DREAM AI is the application, orchestration layer and model-runtime interface. It is not a chat wrapper that requires Groq or another hosted AI provider.

**Important:** this release creates the independent DREAM model-runtime boundary, but it does not yet contain trained model weights. Until a model runtime is connected, chat returns an honest setup message rather than pretending that a trained model answered. The next model milestone is to select an open-weight foundation model, evaluate it, adapt it with DREAM datasets and serve it through infrastructure we control.

## Current capabilities

- Responsive, mobile-first chat workspace
- General, coding and learning assistant modes
- Local conversation history stored in this browser/device
- Export conversations as text and copy assistant responses
- Node.js chat API and health endpoint
- Independent DREAM model runtime adapter
- Optional private runtime token (no provider-specific API dependency)
- Request validation, bounded message history and runtime timeout
- Automated Node test suite and GitHub Actions workflow

Live web research, file analysis, authenticated accounts, cloud-synced memory, autonomous tools and a trained DREAM model are not yet implemented.

## Architecture

- **DREAM AI application:** user interface, API, conversation validation and mode orchestration.
- **DREAM engine adapter:** a provider-independent interface that sends inference requests to a model runtime we control.
- **DREAM model:** the open-weight model we select, evaluate and progressively adapt. Model weights and training data must be versioned separately from the web application.
- **DREAM infrastructure:** a model-serving host with enough CPU/GPU memory for the chosen model. The current free Render web service hosts the application; it is not suitable for running a large language model.

The runtime currently speaks the common chat-completions wire format so a self-hosted inference server such as llama.cpp can serve the model. This is a transport protocol, not a dependency on a third-party AI company. The model host and weights remain under our control.

## Connect a self-hosted DREAM model runtime

Set these private environment variables on the Render service:

- `DREAM_MODEL_URL`: base URL of the self-hosted model server, for example `https://your-private-model-host.example/v1`
- `DREAM_MODEL_NAME`: exact model identifier accepted by that server
- `DREAM_MODEL_TOKEN`: optional bearer token if the model server requires authentication

The adapter appends `/chat/completions` unless the URL already ends with that path. Keep model credentials private. Never put them in GitHub or browser JavaScript.

Do not use the placeholder URL above as a real endpoint. A real model host must first be provisioned and secured. Prefer HTTPS, restrict access to the DREAM AI service, set usage limits, and never expose an unauthenticated model server to the public internet.

## Model development roadmap

1. Select an open-weight base model that fits our realistic compute budget and target devices.
2. Establish evaluation datasets for reasoning, coding, factuality, instruction following, safety and Kenyan/local context.
3. Build a repeatable fine-tuning pipeline with licensed and consented data.
4. Train adapters first where practical; compare against the untouched base model using held-out evaluations.
5. Version model weights, datasets, training configs and evaluation reports.
6. Serve the chosen model on infrastructure we control and connect it through the DREAM engine adapter.
7. Improve inference cost, latency, privacy, reliability and offline options over time.

Training a competitive foundation model from zero is a major compute and data investment. We should not claim DREAM owns a trained model until we have produced, versioned and evaluated actual model weights.

## Requirements and local setup

- Node.js 20.12 or newer

Run `npm start` and open http://localhost:3000. For development, run `npm run dev`. The server automatically reads a local `.env` file when present.

## Deploy on Render

- Runtime: Node
- Build command: `npm install`
- Start command: `npm start`
- Health endpoint: `/api/health`
- Repository: https://github.com/mjnr0477/dream-ai
- Branch: main

## API

- `GET /api/health` — application and DREAM runtime configuration status
- `POST /api/chat` — accepts a messages array and optional mode
- Supported modes: `general`, `coding`, `learning`

## Security principles

1. Keep DREAM AI independent from the user's other applications.
2. Keep model credentials on the server.
3. Validate requests and bound resource usage.
4. Use explicit permissions and isolated execution for future tools and code execution.
5. Never claim web research, file access, code execution or other actions that have not actually happened.
6. Distinguish application availability from actual model availability.
