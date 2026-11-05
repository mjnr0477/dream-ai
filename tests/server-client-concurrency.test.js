import test, { before, after } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import http from "node:http";
import net from "node:net";

let app;
let model;
let baseUrl;
let modelUrl;
let activeCompletions = 0;
let firstCompletionStarted;
let signalFirstCompletion;

async function freePort() {
  const server = net.createServer();
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const { port } = server.address();
  await new Promise(resolve => server.close(resolve));
  return port;
}

async function waitForServer(url) {
  const deadline = Date.now() + 10000;
  while (Date.now() < deadline) {
    if (app.exitCode !== null) throw new Error("DREAM server exited before becoming ready");
    try {
      const response = await fetch(url + "/api/health/live");
      if (response.ok) return;
    } catch {}
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  throw new Error("DREAM server did not become ready");
}

before(async () => {
  firstCompletionStarted = new Promise(resolve => { signalFirstCompletion = resolve; });
  const modelPort = await freePort();
  model = http.createServer((req, res) => {
    if (req.url === "/health") {
      res.writeHead(200, { "content-type": "application/json" });
      return res.end(JSON.stringify({
        ok: true,
        engine: "dream-native-transformer",
        model: "security-test-model",
        completed_training_steps: 1,
        checkpoint_sha256: "a".repeat(64),
        parameters: 100
      }));
    }
    if (["/chat/completions", "/v1/chat/completions"].includes(req.url) && req.method === "POST") {
      activeCompletions += 1;
      if (activeCompletions === 1) signalFirstCompletion();
      return setTimeout(() => {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ choices: [{ message: { content: "Test response" } }] }));
        activeCompletions -= 1;
      }, 900);
    }
    res.writeHead(404).end();
  });
  await new Promise(resolve => model.listen(modelPort, "127.0.0.1", resolve));
  modelUrl = `http://127.0.0.1:${modelPort}`;

  const appPort = await freePort();
  baseUrl = `http://127.0.0.1:${appPort}`;
  app = spawn(process.execPath, ["server.js"], {
    cwd: process.cwd(),
    env: {
      ...process.env,
      PORT: String(appPort),
      DREAM_MODEL_URL: modelUrl,
      DREAM_MODEL_NAME: "security-test-model",
      DREAM_MODEL_TOKEN: "test-only-token"
    },
    stdio: "ignore"
  });
  await waitForServer(baseUrl);
});

after(async () => {
  if (app && app.exitCode === null) app.kill("SIGTERM");
  if (model) await new Promise(resolve => model.close(resolve));
});

function chat(clientIp) {
  return fetch(baseUrl + "/api/chat", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      origin: baseUrl,
      "cf-connecting-ip": clientIp,
      "x-forwarded-for": "203.0.113.250"
    },
    body: JSON.stringify({
      mode: "general",
      messages: [{ role: "user", content: "Hello" }]
    })
  });
}

test("live chat route enforces one in-flight request per client", { timeout: 15000 }, async () => {
  const firstRequest = chat("198.51.100.101");
  await firstCompletionStarted;

  const sameClientChat = await chat("198.51.100.101");
  assert.equal(sameClientChat.status, 429);
  assert.ok(Number(sameClientChat.headers.get("retry-after")) >= 1);

  const sameClientResearch = await fetch(baseUrl + "/api/research", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      origin: baseUrl,
      "cf-connecting-ip": "198.51.100.101",
      "x-forwarded-for": "203.0.113.250"
    },
    body: JSON.stringify({ query: "DREAM security" })
  });
  assert.equal(sameClientResearch.status, 429);

  const otherClient = await chat("198.51.100.102");
  assert.equal(otherClient.status, 200);
  assert.equal((await otherClient.json()).reply, "Test response");

  const firstResponse = await firstRequest;
  assert.equal(firstResponse.status, 200);
  assert.equal((await firstResponse.json()).reply, "Test response");

  const retryAfterRelease = await chat("198.51.100.101");
  assert.equal(retryAfterRelease.status, 200);
});
