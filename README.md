# Dream AI

Dream AI is an independent AI assistant project. It has its own repository and is intentionally separate from Dream Education, Dream Wi-Fi and VYBORA.

## Current foundation

- Responsive chat interface
- Node.js HTTP server and JSON chat API
- Health endpoint
- Optional OpenAI-compatible chat-completions provider adapter
- Honest starter mode when no model provider is configured
- Request validation, body-size limits and provider timeout
- Built-in Node test runner

## Requirements

- Node.js 20 or newer

## Run locally

```bash
npm start
```

Open http://localhost:3000.

For development with automatic server restart:

```bash
npm run dev
```

## Connect a language model

Copy `.env.example` to `.env` and set:

- `AI_BASE_URL`: provider API base URL (or full chat-completions endpoint)
- `AI_API_KEY`: secret API key
- `AI_MODEL`: model identifier

Never commit real API keys. On Render, configure these as private environment variables in the service settings.

The provider adapter uses the common `/chat/completions` request format. A connected provider supplies the language-model inference; it is not a model trained by this repository. Training and serving Dream AI's own model is a later, separate engineering phase requiring suitable datasets and compute.

## Deploy

Create a Render Web Service connected to this repository:

- Runtime: Node
- Build command: `npm install`
- Start command: `npm start`
- Health check path: `/api/health`

Set model credentials only in Render environment settings, never in source files.

## API

- `GET /api/health` — service and model configuration status
- `POST /api/chat` — accepts `{ "messages": [{ "role": "user", "content": "Hello" }] }`

## Project principles

1. Keep Dream AI independent from the user's other applications.
2. Never expose provider secrets to the browser.
3. Validate requests and bound resource usage.
4. Test changes before deployment.
5. Be explicit about whether a response came from a connected model or starter mode.

## Roadmap

1. Verify the core service and tests.
2. Connect a suitable model provider or open-weight model.
3. Add persistent accounts and conversation memory with privacy controls.
4. Add retrieval, tools and agent workflows with permission boundaries.
5. Evaluate open models and build a realistic fine-tuning/training pipeline.
