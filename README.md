# Dream AI

Dream AI is an independent AI assistant project, deployed separately from Dream Education, Dream Wi-Fi and VYBORA.

## Live service
https://dream-ai-live.onrender.com

## Current capabilities
- Responsive, mobile-first chat workspace
- General, coding and learning assistant modes
- Local conversation history stored in this browser/device
- Export conversations as text and copy assistant responses
- Node.js chat API and health endpoint
- Optional OpenAI-compatible model provider adapter
- Request validation, bounded message history and provider timeout
- Automated Node test suite and GitHub Actions workflow

## Important model status
The service can host real model responses after a provider is configured. Until then it deliberately uses a clearly labelled starter response. The repository does not yet contain a trained language model, and a UI mode does not itself create new model capabilities. Live web research, file analysis, agent execution, user accounts and cloud-synced memory are not yet implemented.

## Requirements and local setup
- Node.js 20.12 or newer

Run npm start and open http://localhost:3000. For development, run npm run dev. The server automatically reads a local .env file when present.

## Connect a language model
Set these private environment variables in the Render service dashboard:
- AI_BASE_URL: provider base URL, for example an OpenAI-compatible API base URL
- AI_API_KEY: secret provider key
- AI_MODEL: exact model identifier supported by that provider

Never put a real key in GitHub, browser JavaScript, or a public issue. After saving the variables, redeploy the service. The backend sends requests to the provider; the browser never receives the key.

The provider must support the chat-completions request format. Check the provider's current pricing, free-tier limits, privacy policy and model capabilities before connecting it. A connected provider supplies inference; it is not a model trained by this repository.

## Recommended no-cost model connection

The current provider adapter works with Groq's OpenAI-compatible chat API. To try the available free tier, create a Groq API key and add these values as private environment variables on the Render service:

- AI_BASE_URL: `https://api.groq.com/openai/v1`
- AI_API_KEY: your Groq API key (keep it private)
- AI_MODEL: `openai/gpt-oss-120b`

Groq's published Free Plan limits currently list this model at 30 requests per minute and 1,000 requests per day, with token limits; the exact limits shown in your Groq console take precedence. Free access is rate-limited and is not unlimited production capacity. Groq says inference inputs and outputs are not retained by default, with limited retention possible for reliability and abuse investigations. Review the current terms before sending sensitive information.

## Deploy on Render
- Runtime: Node
- Build command: npm install
- Start command: npm start
- Health check path: /api/health
- Repository: https://github.com/mjnr0477/dream-ai
- Branch: main

## API
- GET /api/health — service and model configuration status
- POST /api/chat — accepts a messages array and optional mode
- Supported modes: general, coding, learning

## Security and project principles
1. Keep Dream AI independent from the user's other applications.
2. Keep provider credentials on the server.
3. Validate requests and bound resource usage.
4. Use explicit permissions and isolated execution for future tools and code execution.
5. Never claim web research, file access, code execution or other actions that have not actually happened.
6. Be clear about whether a response came from a connected model or starter mode.

## Development roadmap
1. Verify deployment and automated tests.
2. Connect a selected model provider.
3. Add grounded web search and document retrieval.
4. Add authenticated accounts and encrypted cloud conversation storage.
5. Add a permissioned agent/tool runtime with isolated code execution.
6. Evaluate open-weight models and the realistic cost of training or fine-tuning a Dream model.