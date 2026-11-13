# Full-application answer evaluation

The direct model evaluator in `scripts/run-evals.mjs` is a low-level model smoke test. It bypasses the public application prompt and automatic research flow.

Use `scripts/run-chat-evals.mjs` when measuring what a user actually receives. It checks `/api/health`, requires the model runtime to be ready, then sends each synthetic evaluation prompt through the public `/api/chat` route. Requests are sequential and paced to avoid overwhelming the free service. HTTP 429 and 503 responses receive at most two bounded retries.

## Run a first pass

```bash
DREAM_APP_URL=https://dream-ai-live.onrender.com DREAM_EVAL_LIMIT=12 npm run eval:chat
```

The evaluator defaults to 12 cases, selected round-robin across categories before taking second examples, and waits 3.5 seconds between cases. Increase the limit after the first pass. The core set currently contains 80 cases across 32 categories, including everyday assistance, health information, Kenyan legal information, business, relationships, careers, controversial-but-safe analysis, coding, research and 14 multilingual groups. Run the full set with `DREAM_EVAL_LIMIT=100`.

## African and multilingual pass

```bash
DREAM_APP_URL=https://dream-ai-live.onrender.com DREAM_EVAL_SET=african-languages-v1 DREAM_EVAL_LIMIT=20 npm run eval:chat
```

Optional controls:

- `DREAM_EVAL_PAUSE_MS`: delay between calls; default 3500, range 0–10000.
- `DREAM_EVAL_TIMEOUT_MS`: per-request timeout; default 90000, range 1000–120000.

## Multi-turn coverage

A case may contain either a single `prompt` or a synthetic `messages` transcript. Multi-turn transcripts must alternate user and assistant messages, start and end with a user message, and contain at most 15 messages. Both the full-application and direct-model evaluators send the complete transcript and retain it in the unscored result so reviewers can inspect whether DREAM preserved earlier constraints.

The core set includes a multi-turn context-following case. These transcripts are authored benchmark fixtures, not private user conversations, and must never be copied into training data.

The application URL must be an HTTPS origin. HTTP is accepted only for localhost/loopback testing. Do not put credentials, tokens, a path, query string or fragment in the URL.

## Review and score

Each run creates an unscored JSONL file under `eval-results/`. Inspect each output against its case rubric and add `humanScore` as an integer from 0 to 4:

- 0: incorrect, unsafe or no answer
- 1: major errors or mostly incomplete
- 2: partly correct; important gaps remain
- 3: correct and useful with minor gaps
- 4: accurate, complete and well-reasoned

Then summarize the scored file:

```bash
node scripts/summarize-evals.mjs eval-results/scored-run.jsonl eval-results/summary.json
```

The report separates categories and languages, and preserves application/runtime metadata and automatic-research status. It does not use an uncalibrated AI judge or invent a quality score.

The runner sends only the repository's synthetic prompts. It does not submit private user conversations or automatically modify model weights. Keep raw evaluation outputs local unless they have been reviewed for accidental sensitive content.
