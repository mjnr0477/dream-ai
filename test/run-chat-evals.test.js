import test from "node:test";
import assert from "node:assert/strict";
import {
  normalizeAppOrigin,
  modeForEvaluationCategory,
  selectEvaluationCases,
  runChatEvaluations
} from "../scripts/run-chat-evals.mjs";

const cases = [
  {
    id: "math-small",
    category: "mathematics",
    prompt: "What is 7 times 8? Show one step.",
    rubric: ["Answers 56", "Shows a correct step"]
  },
  {
    id: "coding-small",
    category: "coding",
    prompt: "Write a JavaScript function that adds two numbers.",
    rubric: ["Provides runnable JavaScript"]
  }
];

function jsonResponse(value, status = 200, headers = {}) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json", ...headers }
  });
}

test("normalizes only safe application origins", () => {
  assert.equal(normalizeAppOrigin("https://dream-ai-live.onrender.com/"), "https://dream-ai-live.onrender.com");
  assert.equal(normalizeAppOrigin("http://127.0.0.1:3000"), "http://127.0.0.1:3000");
  assert.throws(() => normalizeAppOrigin("http://dream-ai-live.onrender.com"), /must use HTTPS/);
  assert.throws(() => normalizeAppOrigin("https://user:pass@example.com"), /credentials/);
  assert.throws(() => normalizeAppOrigin("https://example.com/api"), /origin only/);
  assert.throws(() => normalizeAppOrigin("https://example.com/?token=x"), /query parameters/);
});

test("maps evaluation categories to the correct DREAM chat modes", () => {
  assert.equal(modeForEvaluationCategory("coding"), "coding");
  assert.equal(modeForEvaluationCategory("debugging"), "coding");
  assert.equal(modeForEvaluationCategory("security"), "coding");
  assert.equal(modeForEvaluationCategory("teaching"), "learning");
  assert.equal(modeForEvaluationCategory("planning"), "learning");
  assert.equal(modeForEvaluationCategory("research-method"), "research");
  assert.equal(modeForEvaluationCategory("research"), "research");
  assert.equal(modeForEvaluationCategory("mathematics"), "general");
});

test("evaluates through the full chat route and retains research diagnostics", async () => {
  const requests = [];
  const run = await runChatEvaluations({
    appUrl: "https://dream-ai-live.onrender.com",
    cases,
    limit: 2,
    pauseMs: 0,
    fetchImpl: async (url, options = {}) => {
      if (String(url).endsWith("/api/health")) {
        return jsonResponse({ ok: true, modelReady: true, provider: "native", version: "0.3.0" });
      }
      requests.push({ url: String(url), body: JSON.parse(options.body) });
      return jsonResponse({
        reply: requests.length === 1 ? "56, because 7 groups of 8 make 56." : "function add(a, b) { return a + b; }",
        mode: requests.at(-1).body.mode,
        source: "dream-model-runtime",
        autoResearch: { status: "sources-found", sourceCount: 2, providers: { wikipedia: "ok" } }
      });
    }
  });

  assert.equal(requests.length, 2);
  assert.ok(requests.every(item => item.url.endsWith("/api/chat")));
  assert.equal(requests[0].body.mode, "general");
  assert.equal(requests[1].body.mode, "coding");
  assert.deepEqual(requests[0].body.messages, [{ role: "user", content: cases[0].prompt }]);
  assert.equal(run.results.length, 2);
  assert.equal(run.results[0].output, "56, because 7 groups of 8 make 56.");
  assert.equal(run.results[0].autoResearchStatus, "sources-found");
  assert.equal(run.results[0].autoResearchSourceCount, 2);
  assert.equal(run.results[0].runtimeProvider, "native");
  assert.equal(run.results[0].evaluationSource, "full-application-chat");
  assert.equal(run.results[0].humanScore, null);
  assert.ok(run.results[0].runId);
  assert.equal(run.results[0].applicationOrigin, "https://dream-ai-live.onrender.com");
});

test("selects one case per category before taking second examples", () => {
  const expanded = [
    { ...cases[0], id: "math-1" },
    { ...cases[0], id: "math-2" },
    { ...cases[0], id: "math-3" },
    { ...cases[1], id: "coding-1" },
    { ...cases[1], id: "coding-2" }
  ];
  const selected = selectEvaluationCases(expanded, 3);
  assert.deepEqual(selected.map(item => item.id), ["coding-1", "math-1", "coding-2"]);
  assert.deepEqual(new Set(selected.slice(0, 2).map(item => item.category)), new Set(["coding", "mathematics"]));
});

test("retries bounded 429 and 503 responses before scoring", async () => {
  let calls = 0;
  const waits = [];
  const run = await runChatEvaluations({
    appUrl: "https://dream-ai-live.onrender.com",
    cases: cases.slice(0, 1),
    limit: 1,
    pauseMs: 0,
    sleepImpl: async ms => { waits.push(ms); },
    fetchImpl: async url => {
      if (String(url).endsWith("/api/health")) return jsonResponse({ ok: true, modelReady: true });
      calls += 1;
      if (calls === 1) return jsonResponse({ error: "busy" }, 429, { "retry-after": "0.1" });
      return jsonResponse({ reply: "56" });
    }
  });
  assert.equal(calls, 2);
  assert.deepEqual(waits, [500]);
  assert.equal(run.results[0].output, "56");
});

test("refuses to benchmark when the deployed model is not ready", async () => {
  let requests = 0;
  await assert.rejects(
    runChatEvaluations({
      appUrl: "https://dream-ai-live.onrender.com",
      cases: cases.slice(0, 1),
      fetchImpl: async () => {
        requests += 1;
        return jsonResponse({ ok: true, modelReady: false, modelStatus: "unavailable" });
      }
    }),
    /model runtime is not ready/
  );
  assert.equal(requests, 1);
});

test("rejects empty replies instead of counting them as successful answers", async () => {
  await assert.rejects(
    runChatEvaluations({
      appUrl: "https://dream-ai-live.onrender.com",
      cases: cases.slice(0, 1),
      limit: 1,
      pauseMs: 0,
      fetchImpl: async url => String(url).endsWith("/api/health")
        ? jsonResponse({ ok: true, modelReady: true })
        : jsonResponse({ reply: "   " })
    }),
    /empty reply/
  );
});

test("validates duplicate ids, empty rubrics and unsafe run bounds", async () => {
  await assert.rejects(
    runChatEvaluations({
      appUrl: "https://dream-ai-live.onrender.com",
      cases: [cases[0], cases[0]],
      fetchImpl: async () => { throw new Error("must not fetch"); }
    }),
    /Duplicate evaluation id/
  );
  await assert.rejects(
    runChatEvaluations({
      appUrl: "https://dream-ai-live.onrender.com",
      cases: [{ ...cases[0], rubric: [] }],
      fetchImpl: async () => { throw new Error("must not fetch"); }
    }),
    /scoring rubric/
  );
  await assert.rejects(
    runChatEvaluations({
      appUrl: "https://dream-ai-live.onrender.com",
      cases,
      limit: 101,
      fetchImpl: async () => { throw new Error("must not fetch"); }
    }),
    /limit must be an integer/
  );
});
