import { randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

const baseUrl = process.env.DREAM_MODEL_URL;
const modelName = process.env.DREAM_MODEL_NAME;
if (!baseUrl || !modelName) {
  throw new Error("Set DREAM_MODEL_URL and DREAM_MODEL_NAME for the self-hosted DREAM runtime.");
}

let endpointBase = baseUrl;
while (endpointBase.endsWith("/")) endpointBase = endpointBase.slice(0, -1);
const endpoint = endpointBase.endsWith("/chat/completions")
  ? endpointBase
  : endpointBase + "/chat/completions";

const healthUrl = new URL(baseUrl);
healthUrl.pathname = healthUrl.pathname
  .replace(/\/+$/, "")
  .replace(/\/v1(?:\/chat\/completions)?$/, "")
  .replace(/\/chat\/completions$/, "") + "/health";
healthUrl.search = "";
healthUrl.hash = "";
const healthResponse = await fetch(healthUrl, {
  headers: { accept: "application/json" },
  signal: AbortSignal.timeout(5000)
});
if (!healthResponse.ok) {
  throw new Error("DREAM native runtime health check failed: HTTP " + healthResponse.status);
}
const runtime = await healthResponse.json();
if (
  runtime?.ok !== true ||
  runtime?.engine !== "dream-native-transformer" ||
  runtime?.model !== modelName ||
  !/^[a-f0-9]{64}$/.test(runtime?.checkpoint_sha256 || "") ||
  !Number.isInteger(runtime?.completed_training_steps) ||
  runtime.completed_training_steps < 1 ||
  !Number.isInteger(runtime?.parameters) ||
  runtime.parameters < 1
) {
  throw new Error("Runtime identity or checkpoint provenance is missing; refusing to benchmark an unverified model.");
}
const runId = randomUUID();
const runStartedAt = new Date().toISOString();
const cases = (await readFile(new URL("../evals/core-v1.jsonl", import.meta.url), "utf8"))
  .split(/\r?\n/)
  .filter(Boolean)
  .map(line => JSON.parse(line));
const requestedLimit = Number(process.env.DREAM_EVAL_LIMIT || cases.length);
const limit = Number.isFinite(requestedLimit) ? requestedLimit : cases.length;
const selected = cases.slice(0, Math.max(1, Math.min(limit, cases.length)));
const headers = { "content-type": "application/json" };
if (process.env.DREAM_MODEL_TOKEN) {
  headers.authorization = "Bearer " + process.env.DREAM_MODEL_TOKEN;
}

const results = [];
for (const item of selected) {
  const started = Date.now();
  const response = await fetch(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: modelName,
      messages: [
        { role: "system", content: "You are being evaluated as DREAM AI. Answer the user directly, accurately and transparently. Do not invent facts or claim actions you did not perform." },
        { role: "user", content: item.prompt }
      ],
      temperature: 0
    }),
    signal: AbortSignal.timeout(90000)
  });

  if (!response.ok) {
    throw new Error("Runtime failed on " + item.id + ": HTTP " + response.status);
  }
  const data = await response.json();
  const output = data?.choices?.[0]?.message?.content;
  if (typeof output !== "string" || !output.trim()) {
    throw new Error("Runtime returned no text for " + item.id);
  }

  results.push({
    runId,
    runStartedAt,
    evaluationSet: "core-v1",
    model: modelName,
    checkpointSha256: runtime.checkpoint_sha256,
    completedTrainingSteps: runtime.completed_training_steps,
    runtimeDevice: runtime.device,
    runtimeParameters: runtime.parameters,
    id: item.id,
    category: item.category,
    language: item.language ?? null,
    prompt: item.prompt,
    rubric: item.rubric,
    output: output.trim(),
    latencyMs: Date.now() - started,
    humanScore: null,
    reviewerNotes: ""
  });
  process.stdout.write("Completed " + item.id + " (" + (Date.now() - started) + " ms)\n");
}

await mkdir("eval-results", { recursive: true });
const filename = "dream-eval-" + new Date().toISOString().replace(/[:.]/g, "-") + ".jsonl";
await writeFile(join("eval-results", filename), results.map(row => JSON.stringify(row)).join("\n") + "\n");
process.stdout.write("Saved " + results.length + " raw responses to eval-results/" + filename + ". Score the rubric manually; this runner does not pretend to be an objective judge.\n");
