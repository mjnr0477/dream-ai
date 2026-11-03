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

before(async () => {
  model = http.createServer((req, res) => {
    const chunks = [];
    req.on("data", chunk => chunks.push(chunk));
    req.on("end", () => {
      modelRequests.push({
        headers: req.headers,
        url: req.url,
        body: JSON.parse(Buffer.concat(chunks).toString("utf8"))
      });
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({
        choices: [{ message: { content: "Test response from the DREAM model runtime." } }]
      }));
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
  assert.equal(data.modelName, "dream-test-model");
  assert.equal(data.version, "0.2.0");
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
