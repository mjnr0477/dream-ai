import { randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const DEFAULT_LIMIT = 12;
const MAX_LIMIT = 100;
const DEFAULT_TIMEOUT_MS = 90000;
const MAX_RETRIES = 2;

export function normalizeAppOrigin(value) {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error("Set DREAM_APP_URL to the public DREAM AI application URL.");
  }
  let url;
  try { url = new URL(value.trim()); }
  catch { throw new Error("DREAM_APP_URL must be a valid absolute URL."); }

  const hostname = url.hostname.toLowerCase().replace(/^\[|\]$/g, "");
  const loopback = hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1";
  if (url.protocol !== "https:" && !(url.protocol === "http:" && loopback)) {
    throw new Error("DREAM_APP_URL must use HTTPS unless it targets loopback for local testing.");
  }
  if (url.username || url.password || url.search || url.hash) {
    throw new Error("DREAM_APP_URL must not contain credentials, query parameters or fragments.");
  }
  if (url.pathname !== "/" && url.pathname !== "") {
    throw new Error("DREAM_APP_URL must be an origin only, without an application path.");
  }
  return url.origin;
}

export function modeForEvaluationCategory(category) {
  if (category === "coding" || category === "debugging" || category === "security") return "coding";
  if (category === "teaching" || category === "planning" || category === "learning") return "learning";
  if (category === "research-method" || category === "research") return "research";
  return "general";
}

function validateCases(cases) {
  if (!Array.isArray(cases) || cases.length === 0) {
    throw new Error("The evaluation set must contain at least one case.");
  }
  const ids = new Set();
  for (const [index, item] of cases.entries()) {
    if (!item || typeof item.id !== "string" || !item.id.trim()) {
      throw new Error("Missing evaluation id at case " + (index + 1));
    }
    if (ids.has(item.id)) throw new Error("Duplicate evaluation id: " + item.id);
    ids.add(item.id);
    if (typeof item.category !== "string" || !item.category.trim()) {
      throw new Error("Missing category for " + item.id);
    }
    if (typeof item.prompt !== "string" || !item.prompt.trim() || item.prompt.length > 8000) {
      throw new Error("Invalid prompt for " + item.id);
    }
    if (!Array.isArray(item.rubric) || item.rubric.length === 0 ||
        item.rubric.some(point => typeof point !== "string" || !point.trim())) {
      throw new Error("Missing scoring rubric for " + item.id);
    }
  }
}

export function selectEvaluationCases(cases, limit = DEFAULT_LIMIT) {
  if (!Number.isInteger(limit) || limit < 1 || limit > MAX_LIMIT) {
    throw new Error("Evaluation limit must be an integer from 1 to " + MAX_LIMIT + ".");
  }
  validateCases(cases);
  const groups = new Map();
  for (const item of cases) {
    if (!groups.has(item.category)) groups.set(item.category, []);
    groups.get(item.category).push(item);
  }
  const categories = [...groups.keys()].sort((a, b) => a.localeCompare(b));
  const selected = [];
  for (let round = 0; selected.length < Math.min(limit, cases.length); round += 1) {
    let added = false;
    for (const category of categories) {
      const item = groups.get(category)[round];
      if (!item) continue;
      selected.push(item);
      added = true;
      if (selected.length >= Math.min(limit, cases.length)) break;
    }
    if (!added) break;
  }
  return selected;
}

function retryDelay(response, attempt) {
  const fallback = [1500, 3000][attempt] ?? 3000;
  let value = "";
  try { value = response.headers?.get?.("retry-after") || ""; } catch {}
  if (!value) return fallback;
  const seconds = Number(value);
  const requested = Number.isFinite(seconds)
    ? seconds * 1000
    : Date.parse(value) - Date.now();
  if (!Number.isFinite(requested) || requested <= 0) return fallback;
  return Math.min(15000, Math.max(500, requested));
}

async function readJson(response, label) {
  try { return await response.json(); }
  catch { throw new Error(label + " returned invalid JSON."); }
}

export async function runChatEvaluations({
  appUrl,
  cases,
  limit = DEFAULT_LIMIT,
  pauseMs = 1000,
  timeoutMs = DEFAULT_TIMEOUT_MS,
  fetchImpl = fetch,
  sleepImpl = ms => new Promise(resolve => setTimeout(resolve, ms))
}) {
  const origin = normalizeAppOrigin(appUrl);
  validateCases(cases);
  if (!Number.isInteger(limit) || limit < 1 || limit > MAX_LIMIT) {
    throw new Error("Evaluation limit must be an integer from 1 to " + MAX_LIMIT + ".");
  }
  if (!Number.isInteger(pauseMs) || pauseMs < 0 || pauseMs > 10000) {
    throw new Error("Evaluation pause must be an integer from 0 to 10000 milliseconds.");
  }
  if (!Number.isInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 120000) {
    throw new Error("Evaluation timeout must be an integer from 1000 to 120000 milliseconds.");
  }

  const healthResponse = await fetchImpl(origin + "/api/health", {
    method: "GET",
    headers: { accept: "application/json" },
    signal: AbortSignal.timeout(timeoutMs)
  });
  if (!healthResponse.ok) {
    throw new Error("DREAM AI health check failed: HTTP " + healthResponse.status);
  }
  const health = await readJson(healthResponse, "DREAM AI health check");
  if (health?.ok !== true || health?.modelReady !== true) {
    throw new Error("DREAM AI application or model runtime is not ready; refusing to score unavailable responses.");
  }

  const runId = randomUUID();
  const runStartedAt = new Date().toISOString();
  const selected = selectEvaluationCases(cases, limit);
  const results = [];

  for (const item of selected) {
    let response;
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt += 1) {
      const started = Date.now();
      try {
        response = await fetchImpl(origin + "/api/chat", {
          method: "POST",
          headers: { "content-type": "application/json", accept: "application/json" },
          body: JSON.stringify({
            mode: modeForEvaluationCategory(item.category),
            messages: [{ role: "user", content: item.prompt }]
          }),
          signal: AbortSignal.timeout(timeoutMs)
        });
      } catch (error) {
        throw new Error("DREAM chat request failed for " + item.id + ": " + (error?.message || "network error"));
      }

      if ((response.status === 429 || response.status === 503) && attempt < MAX_RETRIES) {
        await sleepImpl(retryDelay(response, attempt));
        continue;
      }
      if (!response.ok) {
        const detail = await readJson(response, "DREAM chat error response").catch(() => ({}));
        throw new Error("DREAM chat failed for " + item.id + ": HTTP " + response.status +
          (typeof detail?.error === "string" ? " (" + detail.error.slice(0, 180) + ")" : ""));
      }

      const payload = await readJson(response, "DREAM chat response");
      if (typeof payload?.reply !== "string" || !payload.reply.trim()) {
        throw new Error("DREAM returned an empty reply for " + item.id + ".");
      }
      const autoResearch = payload.autoResearch && typeof payload.autoResearch === "object"
        ? payload.autoResearch
        : null;
      results.push({
        runId,
        runStartedAt,
        evaluationSource: "full-application-chat",
        evaluationSet: process.env.DREAM_EVAL_SET || "core-v1",
        applicationOrigin: origin,
        applicationVersion: health.version || null,
        runtimeProvider: health.provider || null,
        id: item.id,
        category: item.category,
        language: item.language ?? null,
        mode: modeForEvaluationCategory(item.category),
        prompt: item.prompt,
        rubric: item.rubric,
        output: payload.reply.trim(),
        source: payload.source || null,
        autoResearchStatus: autoResearch?.status || null,
        autoResearchSourceCount: Number.isInteger(autoResearch?.sourceCount) ? autoResearch.sourceCount : null,
        autoResearchProviders: autoResearch?.providers || null,
        latencyMs: Date.now() - started,
        humanScore: null,
        reviewerNotes: ""
      });
      process.stdout.write("Completed " + item.id + " (" + (Date.now() - started) + " ms)\n");
      break;
    }

    if (pauseMs > 0 && results.length < selected.length) {
      await sleepImpl(pauseMs);
    }
  }

  return {
    runId,
    runStartedAt,
    applicationOrigin: origin,
    health,
    results
  };
}

async function main() {
  const appUrl = process.env.DREAM_APP_URL;
  const evaluationSet = process.env.DREAM_EVAL_SET || "core-v1";
  const files = {
    "core-v1": "../evals/core-v1.jsonl",
    "african-languages-v1": "../evals/african-languages-v1.jsonl"
  };
  const file = files[evaluationSet];
  if (!file) throw new Error("DREAM_EVAL_SET must be core-v1 or african-languages-v1.");

  const rows = (await readFile(new URL(file, import.meta.url), "utf8"))
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line, index) => {
      try { return JSON.parse(line); }
      catch { throw new Error("Invalid JSON on evaluation line " + (index + 1)); }
    });
  const rawLimit = process.env.DREAM_EVAL_LIMIT;
  const limit = rawLimit === undefined || rawLimit === "" ? DEFAULT_LIMIT : Number(rawLimit);
  if (!Number.isInteger(limit) || limit < 1 || limit > MAX_LIMIT) {
    throw new Error("DREAM_EVAL_LIMIT must be an integer from 1 to " + MAX_LIMIT + ".");
  }
  const pauseMs = Number(process.env.DREAM_EVAL_PAUSE_MS || 1000);
  const timeoutMs = Number(process.env.DREAM_EVAL_TIMEOUT_MS || DEFAULT_TIMEOUT_MS);

  const run = await runChatEvaluations({ appUrl, cases: rows, limit, pauseMs, timeoutMs });
  await mkdir("eval-results", { recursive: true });
  const filename = "dream-chat-eval-" + run.runStartedAt.replace(/[:.]/g, "-") + ".jsonl";
  await writeFile(
    join("eval-results", filename),
    run.results.map(row => JSON.stringify(row)).join("\n") + "\n",
    "utf8"
  );
  process.stdout.write(
    "Saved " + run.results.length + " full-application responses to eval-results/" + filename +
    ". Responses are unscored; review each rubric and add humanScore values from 0 to 4.\n"
  );
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => {
    process.stderr.write(error.message + "\n");
    process.exitCode = 1;
  });
}
