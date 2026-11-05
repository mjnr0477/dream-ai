import test, { before, after } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import http from "node:http";

const port = 3197;
const modelPort = 3198;
const base = `http://127.0.0.1:${port}`;
let server;
let model;
const modelRequests = [];
let modelEngine = "dream-native-transformer";
let modelHasCheckpoint = true;
let runtimeModelName = "dream-test-model";
let holdModelResponses = false;
const heldModelResponses = [];

before(async () => {
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
          choices: [{ message: { content: "Test response from the DREAM model runtime." } }]
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
      DREAM_MODEL_URL: `http://127.0.0.1:${modelPort}/v1`,
      DREAM_MODEL_NAME: "dream-test-model",
      DREAM_MODEL_TOKEN: ""
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

async function postChat(payload, rawBody) {
  return fetch(`${base}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: rawBody ?? JSON.stringify(payload)
  });
}

test("health endpoint identifies the DREAM engine and configured model", async () => {
  const response = await fetch(`${base}/api/health`);
  assert.equal(response.status, 200);
  const data = await response.json();
  assert.equal(data.ok, true);
  assert.equal(data.name, "DREAM AI");
  assert.equal(data.engine, "dream-engine");
  assert.equal(data.modelConfigured, true);
  assert.equal(data.modelReady, true);
  assert.equal(data.runtimeStatus, "ready");
  assert.equal(data.modelName, "dream-test-model");
  assert.equal(data.model.completedTrainingSteps, 12);
  assert.equal(data.model.checkpointSha256, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
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
    assert.equal(data.runtimeStatus, "unavailable");
    assert.equal(data.model, null);
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
    assert.equal(data.runtimeStatus, "unavailable");
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

test("chat endpoint uses the DREAM runtime in all supported modes", async () => {
  for (const mode of ["general", "coding", "learning"]) {
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

  assert.equal(modelRequests.length, 3);
  for (const request of modelRequests) {
    assert.equal(request.url, "/v1/chat/completions");
    assert.equal(request.body.model, "dream-test-model");
    assert.equal(request.body.messages.at(-1).content, "Hello DREAM AI");
    assert.equal(request.headers.authorization, undefined);
  }
  assert.match(modelRequests[0].body.messages[0].content, /Answer clearly/);
  assert.match(modelRequests[1].body.messages[0].content, /senior software engineer/);
  assert.match(modelRequests[2].body.messages[0].content, /patient expert tutor/);
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
  assert.equal(data.contractVersion, 1);
  assert.equal(data.model.status, "ready");
  assert.equal(data.model.completedTrainingSteps, 12);
  assert.deepEqual(data.application.modes, ["general", "coding", "learning"]);
  assert.equal(data.safety.toolExecutionDefault, "disabled");
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
    assert.equal(data.model.completedTrainingSteps, null);
    assert.equal(data.model.checkpointSha256, null);
  } finally {
    modelHasCheckpoint = true;
  }
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
