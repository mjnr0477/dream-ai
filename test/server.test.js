import test, { before, after } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import http from "node:http";

const port = 3197;
const providerPort = 3198;
const base = `http://127.0.0.1:${port}`;
let server;
let provider;
const providerRequests = [];

before(async () => {
  provider = http.createServer((req, res) => {
    const chunks = [];
    req.on("data", chunk => chunks.push(chunk));
    req.on("end", () => {
      providerRequests.push({
        headers: req.headers,
        body: JSON.parse(Buffer.concat(chunks).toString("utf8"))
      });
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({
        choices: [{ message: { content: "Mock provider response from Dream AI." } }]
      }));
    });
  });
  await new Promise((resolve, reject) => {
    provider.once("error", reject);
    provider.listen(providerPort, "127.0.0.1", resolve);
  });

  server = spawn(process.execPath, ["server.js"], {
    env: {
      ...process.env,
      PORT: String(port),
      AI_BASE_URL: `http://127.0.0.1:${providerPort}/v1`,
      AI_API_KEY: "test-only-key",
      AI_MODEL: "mock-model"
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
  throw new Error("Dream AI server did not start for tests.");
});

after(async () => {
  if (server && server.exitCode === null && server.signalCode === null) {
    await new Promise(resolve => {
      server.once("exit", resolve);
      server.kill("SIGTERM");
    });
  }
  if (provider) await new Promise(resolve => provider.close(resolve));
});

async function postChat(payload, rawBody) {
  return fetch(`${base}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: rawBody ?? JSON.stringify(payload)
  });
}

test("health endpoint reports Dream AI and configured provider", async () => {
  const response = await fetch(`${base}/api/health`);
  assert.equal(response.status, 200);
  const data = await response.json();
  assert.equal(data.ok, true);
  assert.equal(data.name, "Dream AI");
  assert.equal(data.modelConfigured, true);
  assert.equal(data.version, "0.1.1");
});

test("chat endpoint validates missing messages", async () => {
  const response = await postChat({});
  assert.equal(response.status, 400);
});

test("chat endpoint rejects malformed JSON", async () => {
  const response = await postChat(null, "{not-json");
  assert.equal(response.status, 400);
});

test("chat endpoint uses the configured provider in all supported modes", async () => {
  for (const mode of ["general", "coding", "learning"]) {
    const response = await postChat({
      mode,
      messages: [{ role: "user", content: "Hello Dream AI" }]
    });
    assert.equal(response.status, 200);
    const data = await response.json();
    assert.equal(data.mode, mode);
    assert.equal(data.reply, "Mock provider response from Dream AI.");
  }

  assert.equal(providerRequests.length, 3);
  for (const request of providerRequests) {
    assert.equal(request.headers.authorization, "Bearer test-only-key");
    assert.equal(request.body.model, "mock-model");
    assert.equal(request.body.messages.at(-1).content, "Hello Dream AI");
  }
  assert.match(providerRequests[0].body.messages[0].content, /Answer clearly/);
  assert.match(providerRequests[1].body.messages[0].content, /senior software engineer/);
  assert.match(providerRequests[2].body.messages[0].content, /patient expert tutor/);
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

test("root serves the Dream AI application", async () => {
  const response = await fetch(`${base}/`);
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type"), /text\/html/);
  assert.match(await response.text(), /DREAM AI/);
});

test("static server blocks encoded traversal attempts", async () => {
  const response = await fetch(`${base}/%2e%2e%2fserver.js`);
  assert.notEqual(response.status, 200);
});
