import test, { before, after } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import http from "node:http";
import net from "node:net";

let port;
let modelPort;
let learningIpCounter = 210;
let base;
let server;
let model;
const modelRequests = [];
let modelReply = "Test response from the DREAM model runtime.";
let modelEngine = "dream-native-transformer";
let modelHasCheckpoint = true;
let runtimeModelName = "dream-test-model";
let holdModelResponses = false;
const heldModelResponses = [];

async function getEphemeralPorts(count) {
  const probes = Array.from({ length: count }, () => net.createServer());
  await Promise.all(probes.map(probe => new Promise((resolve, reject) => {
    probe.once("error", reject);
    probe.listen(0, "127.0.0.1", resolve);
  })));
  const ports = probes.map(probe => probe.address().port);
  await Promise.all(probes.map(probe => new Promise((resolve, reject) => {
    probe.close(error => error ? reject(error) : resolve());
  })));
  return ports;
}

before(async () => {
  [port, modelPort] = await getEphemeralPorts(2);
  base = `http://127.0.0.1:${port}`;
  model = http.createServer((req, res) => {
    if (req.method === "GET" && req.url === "/health") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(modelHasCheckpoint
        ? { ok: true, engine: modelEngine, model: runtimeModelName, completed_training_steps: 12, checkpoint_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", device: "cpu", parameters: 1000, tokenizer: "DreamByteTokenizer" }
        : { ok: true, engine: modelEngine }));
      return;
    }
    const chunks = [];
    req.on("data", chunk => chunks.push(chunk));
    req.on("end", () => {
      modelRequests.push({
        headers: req.headers,
        url: req.url,
        body: JSON.parse(Buffer.concat(chunks).toString("utf8"))
      });
      const respond = () => {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({
          choices: [{ message: { content: modelReply } }]
        }));
      };
      if (holdModelResponses) heldModelResponses.push(respond);
      else respond();
    });
  });
  await new Promise((resolve, reject) => {
    model.once("error", reject);
    model.listen(modelPort, "127.0.0.1", resolve);
  });

  server = spawn(process.execPath, ["server.js"], {
    env: {
      ...process.env,
      PORT: String(port),
      DREAM_MODEL_PROVIDER: "native",
      DREAM_MODEL_URL: `http://127.0.0.1:${modelPort}/v1`,
      DREAM_MODEL_NAME: "dream-test-model",
      DREAM_MODEL_TOKEN: "",
      DREAM_API_RATE_LIMIT: "3",
      DREAM_LEARNING_INGEST_URL: "",
      DREAM_LEARNING_INGEST_TOKEN: "",
      DREAM_LEARNING_SUPABASE_ANON_KEY: ""
    },
    stdio: "ignore"
  });
  for (let attempt = 0; attempt < 50; attempt++) {
    try {
      const response = await fetch(`${base}/api/health`);
      if (response.ok) return;
    } catch {}
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error("DREAM AI server did not start for tests.");
});

after(async () => {
  if (server && server.exitCode === null && server.signalCode === null) {
    await new Promise(resolve => {
      server.once("exit", resolve);
      server.kill("SIGTERM");
    });
  }
  if (model) await new Promise(resolve => model.close(resolve));
});

async function postResearch(payload, rawBody) {
  return fetch(`${base}/api/research`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: rawBody ?? JSON.stringify(payload)
  });
}

async function postLearning(payload, rawBody) {
  return fetch(`${base}/api/learning/examples`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "cf-connecting-ip": `198.51.100.${learningIpCounter++}`
    },
    body: rawBody ?? JSON.stringify(payload)
  });
}

async function postChat(payload, rawBody) {
  return fetch(`${base}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: rawBody ?? JSON.stringify(payload)
  });
}


test("learning contribution refuses private account prompts even when called directly", async () => {
  for (const prompt of [
    "Check my Gmail inbox.",
    "Show my Facebook Pages.",
    "Review my Instagram account.",
    "Check my Threads profile."
  ]) {
    const response = await postLearning({
      mode: "general",
      prompt,
      response: "Private account data must never be collected for training.",
      consentVersion: "dream-learning-opt-in-v1"
    });
    assert.equal(response.status, 403);
    assert.match((await response.json()).error, /Private account questions cannot be submitted/);
  }
});

test("learning contribution requires a valid temporary source group for v2 consent", async () => {
  const missing = await postLearning({
    mode: "general",
    prompt: "Explain photosynthesis.",
    response: "Photosynthesis is how plants convert light energy into chemical energy.",
    consentVersion: "dream-learning-opt-in-v2"
  });
  assert.equal(missing.status, 400);

  const malformed = await postLearning({
    mode: "general",
    prompt: "Explain photosynthesis.",
    response: "Photosynthesis is how plants convert light energy into chemical energy.",
    consentVersion: "dream-learning-opt-in-v2",
    sourceGroupId: "../../not-a-uuid"
  });
  assert.equal(malformed.status, 400);
});

test("learning contribution v3 requires a valid temporary source group", async () => {
  const missing = await postLearning({
    mode: "general",
    prompt: "Explain photosynthesis.",
    response: "Photosynthesis is how plants convert light energy into chemical energy.",
    consentVersion: "dream-learning-opt-in-v3"
  });
  assert.equal(missing.status, 400);

  const accepted = await postLearning({
    mode: "general",
    prompt: "Explain photosynthesis.",
    response: "Photosynthesis is how plants convert light energy into chemical energy.",
    consentVersion: "dream-learning-opt-in-v3",
    sourceGroupId: "123e4567-e89b-42d3-a456-426614174000"
  });
  assert.equal(accepted.status, 503);
  assert.match((await accepted.json()).error, /not configured/i);
});

test("learning contribution rejects malformed consent-version types instead of downgrading them", async () => {
  const response = await postLearning({
    mode: "general",
    prompt: "Explain photosynthesis.",
    response: "Photosynthesis is how plants convert light energy into chemical energy.",
    consentVersion: { version: "dream-learning-opt-in-v2" }
  });
  assert.equal(response.status, 400);
});

test("learning contribution rejects unknown consent versions", async () => {
  const response = await postLearning({
    mode: "general",
    prompt: "Explain photosynthesis.",
    response: "Photosynthesis is how plants convert light energy into chemical energy.",
    consentVersion: "dream-learning-opt-in-v99",
    sourceGroupId: "123e4567-e89b-42d3-a456-426614174000"
  });
  assert.equal(response.status, 400);
});

test("learning contribution keeps legacy v1 clients compatible but accepts valid v2 input", async () => {
  const validV2 = await postLearning({
    mode: "general",
    prompt: "Explain photosynthesis.",
    response: "Photosynthesis is how plants convert light energy into chemical energy.",
    consentVersion: "dream-learning-opt-in-v2",
    sourceGroupId: "123e4567-e89b-42d3-a456-426614174000"
  });
  assert.equal(validV2.status, 503);
  assert.match((await validV2.json()).error, /not configured/i);

  const legacyV1 = await postLearning({
    mode: "general",
    prompt: "Explain photosynthesis.",
    response: "Photosynthesis is how plants convert light energy into chemical energy."
  });
  assert.equal(legacyV1.status, 503);
});

test("research endpoint rejects malformed and oversized queries before upstream access", async () => {
  for (const payload of [
    { query: "" },
    { query: "   " },
    { query: "x".repeat(201) },
    { query: 42 },
    [],
    null
  ]) {
    const response = await postResearch(payload);
    assert.equal(response.status, 400);
    assert.match((await response.json()).error, /research query|JSON object/i);
  }

  const malformed = await postResearch(null, "{");
  assert.equal(malformed.status, 400);
  assert.match((await malformed.json()).error, /valid JSON/i);
});

test("liveness endpoint stays healthy when the native model is unavailable", async () => {
  modelHasCheckpoint = false;
  try {
    const response = await fetch(`${base}/api/health/live`);
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      ok: true,
      name: "DREAM AI",
      status: "live"
    });
  } finally {
    modelHasCheckpoint = true;
  }
});

test("health endpoint exposes minimal runtime readiness without diagnostics", async () => {
  const response = await fetch(`${base}/api/health`);
  assert.equal(response.status, 200);
  const data = await response.json();
  assert.equal(data.ok, true);
  assert.equal(data.name, "DREAM AI");
  assert.equal(data.modelConfigured, true);
  assert.equal(data.modelReady, true);
  assert.equal(data.version, "0.3.0");
  assert.equal(Object.hasOwn(data, "engine"), false);
  assert.equal(Object.hasOwn(data, "runtimeStatus"), false);
  assert.equal(Object.hasOwn(data, "modelName"), false);
  assert.equal(Object.hasOwn(data, "model"), false);
  assert.equal(data.version, "0.3.0");
});

test("health endpoint refuses readiness without a trained checkpoint identity", async () => {
  modelHasCheckpoint = false;
  try {
    const response = await fetch(`${base}/api/health`);
    const data = await response.json();
    assert.equal(response.status, 200);
    assert.equal(data.modelConfigured, true);
    assert.equal(data.modelReady, false);
    assert.equal(Object.hasOwn(data, "runtimeStatus"), false);
    assert.equal(Object.hasOwn(data, "model"), false);
    const chat = await postChat({
      messages: [{ role: "user", content: "Hello DREAM AI" }]
    });
    assert.equal(chat.status, 502);
    assert.match((await chat.json()).error, /trained-checkpoint identity/i);
  } finally {
    modelHasCheckpoint = true;
  }
});

test("health endpoint refuses a different configured native model", async () => {
  runtimeModelName = "unexpected-native-model";
  try {
    const response = await fetch(`${base}/api/health`);
    const data = await response.json();
    assert.equal(response.status, 200);
    assert.equal(data.modelReady, false);
    assert.equal(Object.hasOwn(data, "runtimeStatus"), false);
    const chat = await postChat({
      messages: [{ role: "user", content: "Hello DREAM AI" }]
    });
    assert.equal(chat.status, 502);
    assert.match((await chat.json()).error, /model name does not match/i);
  } finally {
    runtimeModelName = "dream-test-model";
  }
});

test("chat endpoint validates missing messages", async () => {
  const response = await postChat({});
  assert.equal(response.status, 400);
});

test("chat endpoint rejects malformed JSON", async () => {
  const response = await postChat(null, "{not-json");
  assert.equal(response.status, 400);
});

test("coding endpoint returns validated multi-file artifacts without changing ordinary replies", async () => {
  modelReply = [
    "--- DREAM FILE: src/index.js ---",
    "export const ready = true;",
    "--- END DREAM FILE ---",
    "--- DREAM FILE: src/start.js ---",
    "import { ready } from \"./index.js\";",
    "--- END DREAM FILE ---"
  ].join("\n");
  try {
    const response = await postChat({ mode: "coding", messages: [{ role: "user", content: "Create a two-file JavaScript project." }] });
    assert.equal(response.status, 200);
    const data = await response.json();
    assert.equal(data.mode, "coding");
    assert.equal(data.artifacts.length, 2);
    assert.deepEqual(data.artifacts.map(file => file.path), ["src/index.js", "src/start.js"]);
    assert.equal(data.artifacts[0].content, "export const ready = true;\n");
    const ordinary = await postChat({ mode: "general", messages: [{ role: "user", content: "Explain this code." }] });
    assert.equal(ordinary.status, 200);
    const ordinaryData = await ordinary.json();
    assert.equal(ordinaryData.reply, modelReply);
    assert.equal(Object.hasOwn(ordinaryData, "artifacts"), false);
  } finally { modelReply = "Test response from the DREAM model runtime."; }
});

test("general chat recognizes complete application requests and returns validated project files", async () => {
  const originalReply = modelReply;
  modelReply = [
    "--- DREAM FILE: README.md ---",
    "# Appointment Booking",
    "Run with npm install && npm start.",
    "--- END DREAM FILE ---",
    "--- DREAM FILE: package.json ---",
    JSON.stringify({ name: "appointment-booking", private: true, scripts: { start: "node server.js" } }, null, 2),
    "--- END DREAM FILE ---"
  ].join("\n");
  try {
    const before = modelRequests.length;
    const response = await postChat({
      mode: "general",
      messages: [{ role: "user", content: "Build a complete web app for booking appointments." }]
    });
    assert.equal(response.status, 200);
    const data = await response.json();
    assert.equal(data.mode, "general");
    assert.equal(data.artifacts.length, 2);
    assert.deepEqual(data.artifacts.map(item => item.path), ["README.md", "package.json"]);
    const request = modelRequests.at(-1);
    assert.equal(modelRequests.length, before + 1);
    assert.match(request.body.messages[0].content, /APPLICATION BUILD MODE/);
    assert.match(request.body.messages[0].content, /browser can assemble a complete ZIP|complete, coherent project/);
  } finally {
    modelReply = originalReply;
  }
});

test("coding endpoint ignores malformed artifact metadata without losing the model reply", async () => {
  modelReply = "--- DREAM FILE: ../unsafe.js ---\nalert(0);\n--- END DREAM FILE ---";
  try {
    const response = await postChat({ mode: "coding", messages: [{ role: "user", content: "Generate a file." }] });
    assert.equal(response.status, 200);
    const data = await response.json();
    assert.equal(data.reply, modelReply);
    assert.equal(Object.hasOwn(data, "artifacts"), false);
  } finally { modelReply = "Test response from the DREAM model runtime."; }
});

test("chat endpoint uses the DREAM runtime in all supported modes", async () => {
  const beforeCount = modelRequests.length;
  for (const mode of ["general", "coding", "learning", "research"]) {
    const response = await postChat({
      mode,
      messages: [{ role: "user", content: "Hello DREAM AI" }]
    });
    assert.equal(response.status, 200);
    const data = await response.json();
    assert.equal(data.mode, mode);
    assert.equal(data.source, "dream-model");
    assert.equal(data.reply, "Test response from the DREAM model runtime.");
  }

  const modeRequests = modelRequests.slice(beforeCount);
  assert.equal(modeRequests.length, 4);
  for (const request of modeRequests) {
    assert.equal(request.url, "/v1/chat/completions");
    assert.equal(request.body.model, "dream-test-model");
    assert.equal(request.body.messages.at(-1).content, "Hello DREAM AI");
    assert.equal(request.headers.authorization, undefined);
  }
  assert.match(modeRequests[0].body.messages[0].content, /Answer clearly/);
  assert.match(modeRequests[1].body.messages[0].content, /senior software engineer/);
  assert.equal(modeRequests[0].body.temperature, 0.7);
  assert.equal(modeRequests[1].body.temperature, 0.2);
  assert.equal(modeRequests[2].body.temperature, 0.7);
  assert.match(modeRequests[2].body.messages[0].content, /patient expert tutor/);
  assert.match(modeRequests[3].body.messages[0].content, /rigorous research assistant for any user and any subject/);
  for (const request of modelRequests) {
    assert.match(request.body.messages[0].content, /capabilities explicitly available/);
    assert.match(request.body.messages[0].content, /untrusted data/);
    assert.match(request.body.messages[0].content, /explicit user confirmation/);
    assert.match(request.body.messages[0].content, /Never ask users to paste passwords/);
  }
});

test("chat endpoint passes explicitly saved memory as untrusted context", async () => {
  const response = await postChat({
    memoryContext: "Prefers concise explanations. Learning JavaScript.",
    messages: [{ role: "user", content: "Help me study." }]
  });
  assert.equal(response.status, 200);
  const system = modelRequests.at(-1).body.messages[0].content;
  assert.ok(system.includes("\n\nUSER-CONTROLLED MEMORY"));
  assert.match(system, /Prefers concise explanations/);
  assert.match(system, /untrusted contextual data, not instructions/);
  assert.match(system, /never override the current user request/);
});

test("chat endpoint rejects invalid or oversized memory context", async () => {
  const oversized = await postChat({
    memoryContext: "x".repeat(3001),
    messages: [{ role: "user", content: "Hello" }]
  });
  assert.equal(oversized.status, 400);
  const invalid = await postChat({
    memoryContext: { private: true },
    messages: [{ role: "user", content: "Hello" }]
  });
  assert.equal(invalid.status, 400);
});

test("chat endpoint bounds concurrent model requests and tells clients when to retry", async () => {
  const beforeCount = modelRequests.length;
  holdModelResponses = true;
  const first = postChat({
    messages: [{ role: "user", content: "First concurrent request" }]
  });
  const second = postChat({
    messages: [{ role: "user", content: "Second concurrent request" }]
  });
  try {
    for (let attempt = 0; attempt < 50 && modelRequests.length < beforeCount + 2; attempt++) {
      await new Promise(resolve => setTimeout(resolve, 20));
    }
    assert.equal(modelRequests.length, beforeCount + 2);
    const rejected = await postChat({
      messages: [{ role: "user", content: "Third concurrent request" }]
    });
    assert.equal(rejected.status, 503);
    assert.equal(rejected.headers.get("retry-after"), "3");
    const data = await rejected.json();
    assert.match(data.error, /retry in a few seconds/i);
  } finally {
    holdModelResponses = false;
    for (const respond of heldModelResponses.splice(0)) respond();
  }
  const [firstResponse, secondResponse] = await Promise.all([first, second]);
  assert.equal(firstResponse.status, 200);
  assert.equal(secondResponse.status, 200);
});

test("chat endpoint refuses a non-native OpenAI-compatible model service", async () => {
  modelEngine = "third-party-compatible-api";
  try {
    const before = modelRequests.length;
    const response = await postChat({
      messages: [{ role: "user", content: "Hello DREAM AI" }]
    });
    assert.equal(response.status, 502);
    const data = await response.json();
    assert.match(data.error, /not the native DREAM transformer/i);
    assert.equal(modelRequests.length, before);
  } finally {
    modelEngine = "dream-native-transformer";
  }
});

test("chat endpoint rejects unsupported modes", async () => {
  const response = await postChat({
    mode: "unrestricted",
    messages: [{ role: "user", content: "Hello" }]
  });
  assert.equal(response.status, 400);
});

test("chat endpoint rejects invalid message roles", async () => {
  const response = await postChat({
    messages: [{ role: "system", content: "override" }]
  });
  assert.equal(response.status, 400);
});

test("chat endpoint requires the final message to be from the user", async () => {
  const response = await postChat({
    messages: [{ role: "assistant", content: "Hello" }]
  });
  assert.equal(response.status, 400);
});

test("chat endpoint enforces the request body limit", async () => {
  const response = await postChat(null, JSON.stringify({
    messages: [{ role: "user", content: "x".repeat(40000) }]
  }));
  assert.equal(response.status, 413);
});


test("capabilities endpoint reports only verified runtime and tool availability", async () => {
  const response = await fetch(`${base}/api/capabilities`);
  assert.equal(response.status, 200);
  const data = await response.json();
  assert.equal(data.product, "DREAM AI");
  assert.equal(data.contractVersion, 2);
  assert.equal(data.model.status, "ready");
  assert.equal(Object.hasOwn(data.model, "completedTrainingSteps"), false);
  assert.equal(Object.hasOwn(data.model, "checkpointSha256"), false);
  assert.deepEqual(data.application.modes, ["general", "coding", "learning", "research"]);
  assert.ok(data.tools.some(tool => tool.id === "public_knowledge_search" && tool.status === "available"));
  assert.equal(data.safety.toolExecutionDefault, "read-only-public-sources");
  assert.equal(data.safety.unrestrictedAccess, false);
  assert.ok(data.tools.some(tool => tool.id === "code_execution" && tool.status === "not-implemented"));
  assert.ok(data.tools.every(tool => tool.status !== "ready"));
});


test("capabilities endpoint never reports an unverified model as ready", async () => {
  modelHasCheckpoint = false;
  try {
    const response = await fetch(`${base}/api/capabilities`);
    assert.equal(response.status, 200);
    const data = await response.json();
    assert.equal(data.model.status, "unavailable");
    assert.equal(Object.hasOwn(data.model, "completedTrainingSteps"), false);
    assert.equal(Object.hasOwn(data.model, "checkpointSha256"), false);
  } finally {
    modelHasCheckpoint = true;
  }
});


test("chat and research reject non-JSON and cross-site browser requests", async () => {
  const missingType = await fetch(`${base}/api/chat`, {
    method: "POST",
    body: JSON.stringify({ messages: [{ role: "user", content: "hello" }] })
  });
  assert.equal(missingType.status, 415);

  const textType = await fetch(`${base}/api/research`, {
    method: "POST",
    headers: { "content-type": "text/plain" },
    body: JSON.stringify({ query: "DREAM" })
  });
  assert.equal(textType.status, 415);

  for (const [url, payload] of [
    ["/api/chat", { messages: [{ role: "user", content: "hello" }] }],
    ["/api/research", { query: "DREAM" }]
  ]) {
    const hostile = await fetch(base + url, {
      method: "POST",
      headers: { "content-type": "application/json", origin: "https://attacker.example" },
      body: JSON.stringify(payload)
    });
    assert.equal(hostile.status, 403);
  }

  const crossSite = await fetch(`${base}/api/chat`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      origin: base,
      "sec-fetch-site": "cross-site"
    },
    body: JSON.stringify({ messages: [{ role: "user", content: "hello" }] })
  });
  assert.equal(crossSite.status, 403);
});

test("same-host origin with the wrong scheme is rejected", async () => {
  const response = await fetch(`${base}/api/chat`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      origin: base.replace("http://", "https://")
    },
    body: JSON.stringify({ messages: [{ role: "user", content: "hello" }] })
  });
  assert.equal(response.status, 403);
});

test("shared per-client token bucket ignores forged X-Forwarded-For values", async () => {
  const forwarded = "198.51.100.77, 203.0.113.9";
  for (let i = 0; i < 3; i++) {
    const response = await fetch(`${base}/api/chat`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "cf-connecting-ip": "198.51.100.77",
        "x-forwarded-for": "203.0.113.250, " + forwarded
      },
      body: JSON.stringify({ messages: [{ role: "user", content: `limit test ${i}` }] })
    });
    assert.equal(response.status, 200);
  }

  const blocked = await fetch(`${base}/api/research`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "cf-connecting-ip": "198.51.100.77",
      "x-forwarded-for": "192.0.2.99, 192.0.2.44"
    },
    body: JSON.stringify({ query: "DREAM" })
  });
  assert.equal(blocked.status, 429);
  assert.ok(Number(blocked.headers.get("retry-after")) >= 1);
});

test("security headers are present on API and static responses", async () => {
  const api = await fetch(`${base}/api/health`);
  assert.equal(api.headers.get("x-frame-options"), "DENY");
  assert.equal(api.headers.get("x-content-type-options"), "nosniff");
  assert.equal(api.headers.get("cache-control"), "no-store");
  assert.equal(api.headers.get("permissions-policy"), "camera=(), microphone=(), geolocation=()");

  const page = await fetch(base + "/");
  const csp = page.headers.get("content-security-policy") || "";
  assert.match(csp, /default-src 'self'/);
  assert.match(csp, /script-src 'self'/);
  assert.match(csp, /frame-ancestors 'none'/);
  assert.match(csp, /style-src-attr 'unsafe-inline'/);
});

test("root serves the DREAM AI application", async () => {
  const response = await fetch(`${base}/`);
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type"), /text\/html/);
  assert.match(await response.text(), /DREAM AI/);
});

test("static server blocks encoded traversal attempts", async () => {
  const response = await fetch(`${base}/%2e%2e%2fserver.js`);
  assert.notEqual(response.status, 200);
});

test("learning contribution terms page explains the independent review gate", async () => {
  const response = await fetch(`${base}/learning-data-terms.html`);
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /Only the single question and the corresponding DREAM AI answer/);
  assert.match(html, /only after separate human review/);
  assert.match(html, /non-exclusive, worldwide, royalty-free permission/);
  assert.match(html, /not automatically approve an example/);
  assert.match(html, /raw identifier is not saved in the database/);
  assert.match(html, /non-exclusive, worldwide, royalty-free permission/);
  assert.match(html, /permission from the rights-holder/);
  assert.match(html, /free or paid offerings/);
  assert.match(html, /services that DREAM AI may operate/);
  assert.match(html, /question and any other material you supply/);
  assert.match(html, /answer is generated by DREAM AI/);
  assert.match(html, /VERSION 3/);
  assert.match(html, /free or paid offerings/);
  assert.match(html, /services that DREAM AI may operate/);
  assert.match(html, /VERSION 3/);
});

test("plugin catalog endpoint exposes guest-first plugin metadata and the frontend directory", async () => {
  const homepage = await fetch(`${base}/`);
  assert.equal(homepage.status, 200);
  const html = await homepage.text();
  assert.match(html, /id="plugins-open"/);
  assert.match(html, /id="plugins-dialog"/);
  assert.match(html, /id="plugins-search"/);
  assert.match(html, /id="plugins-category"/);

  const frontend = await fetch(`${base}/app.js`);
  assert.equal(frontend.status, 200);
  const appSource = await frontend.text();
  assert.match(appSource, /function renderPluginCards/);
  assert.match(appSource, /Connect Gmail/);
  assert.match(appSource, /Operator must enable Google OAuth/);
  assert.match(appSource, /function isPrivateAccountPrompt/);
  assert.match(appSource, /cleanPrompt && !isPrivateAccountPrompt\(cleanPrompt\)/);
  assert.match(appSource, /Google did not confirm token revocation/);

  const response = await fetch(`${base}/api/plugins`);
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") || "", /application\/json/);

  const catalog = await response.json();
  assert.equal(catalog.guestAccessSupported, true);
  assert.equal(catalog.userRegistrationRequired, false);
  assert.equal(catalog.userSuppliedApiKeysRequired, false);
  assert.ok(Array.isArray(catalog.categories));
  assert.ok(Array.isArray(catalog.plugins));

  const byId = new Map(catalog.plugins.map(plugin => [plugin.id, plugin]));
  assert.equal(byId.get("bluesky").status, "connected_read_only");
  assert.equal(byId.get("public_github").status, "connected_read_only");
  assert.equal(byId.get("arxiv").status, "connected_read_only");
  assert.equal(byId.get("semantic_scholar").status, "connected_read_only");
  assert.equal(byId.get("facebook_instagram_threads").status, "requires_user_authorization");
  assert.equal(byId.get("gmail").status, "requires_user_authorization");
  assert.equal(catalog.policy.privateAccountAccessRequiresAuthorization, true);
  assert.equal(catalog.policy.writeAndFinancialActionsRequireExplicitConfirmation, true);
});


test("Gmail account status is guest-scoped and private Gmail prompts bypass public research", async () => {
  const statusResponse = await fetch(`${base}/api/account/status`);
  assert.equal(statusResponse.status, 200);
  const status = await statusResponse.json();
  assert.equal(status.googleOAuthConfigured, false);
  assert.equal(status.gmail.connected, false);
  assert.match(statusResponse.headers.get("set-cookie") || "", /HttpOnly/);
  assert.match(statusResponse.headers.get("set-cookie") || "", /SameSite=Lax/);

  const response = await fetch(`${base}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json", "cf-connecting-ip": "203.0.113.222" },
    body: JSON.stringify({ messages: [{ role: "user", content: "Search my Gmail for: project meeting" }] })
  });
  assert.equal(response.status, 200);
  const request = modelRequests.at(-1);
  assert.match(request.body.messages[0].content, /USER-AUTHORIZED PRIVATE ACCOUNT DATA/);
  assert.match(request.body.messages[0].content, /not configured by the DREAM AI operator/);
  assert.doesNotMatch(request.body.messages[0].content, /AUTOMATIC PUBLIC RESEARCH/);

  const disconnect = await fetch(`${base}/api/account/google/disconnect`, {
    method: "POST",
    headers: { "content-type": "application/json", "cf-connecting-ip": "203.0.113.223" },
    body: "{}"
  });
  assert.equal(disconnect.status, 200);
  assert.equal((await disconnect.json()).disconnected, false);
});
