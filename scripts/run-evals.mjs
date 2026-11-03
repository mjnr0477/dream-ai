import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

const baseUrl = process.env.DREAM_MODEL_URL;
const modelName = process.env.DREAM_MODEL_NAME;
if (!baseUrl || !modelName) {
  throw new Error("Set DREAM_MODEL_URL and DREAM_MODEL_NAME for the self-hosted DREAM runtime.");
}

const endpointBase = baseUrl.replace(/\/+$/, "");
const endpoint = endpointBase.endsWith("/chat/completions")
  ? endpointBase
  : \`\${endpointBase}/chat/completions\`;
const cases = (await readFile(new URL("../evals/core-v1.jsonl", import.meta.url), "utf8"))
  .split(/\r?\n/)
  .filter(Boolean)
  .map(line => JSON.parse(line));
const limit = Number(process.env.DREAM_EVAL_LIMIT || cases.length);
const selected = cases.slice(0, Math.max(1, Math.min(limit, cases.length)));
const headers = { "content-type": "application/json" };
if (process.env.DREAM_MODEL_TOKEN) {
  headers.authorization = \`Bearer \${process.env.DREAM_MODEL_TOKEN}\`;
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
    throw new Error(\`Runtime failed on \${item.id}: HTTP \${response.status}\`);
  }
  const data = await response.json();
  const output = data?.choices?.[0]?.message?.content;
  if (typeof output !== "string" || !output.trim()) {
    throw new Error(\`Runtime returned no text for \${item.id}\`);
  }

  results.push({
    id: item.id,
    category: item.category,
    prompt: item.prompt,
    rubric: item.rubric,
    output: output.trim(),
    latencyMs: Date.now() - started,
    humanScore: null,
    reviewerNotes: ""
  });
  process.stdout.write(\`Completed \${item.id} (\${Date.now() - started} ms)\\n\`);
}

await mkdir("eval-results", { recursive: true });
const filename = \`dream-eval-\${new Date().toISOString().replace(/[:.]/g, "-")}.jsonl\`;
await writeFile(join("eval-results", filename), results.map(row => JSON.stringify(row)).join("\\n") + "\\n");
process.stdout.write(\`Saved \${results.length} raw responses to eval-results/\${filename}. Score the rubric manually; this runner does not pretend to be an objective judge.\\n\`);
