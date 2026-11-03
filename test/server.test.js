import test, { before, after } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";

const port = 3197;
const base = `http://127.0.0.1:${port}`;
let server;

before(async () => {
  server = spawn(process.execPath, ["server.js"], {
    env: {
      ...process.env,
      PORT: String(port),
      AI_BASE_URL: "",
      AI_API_KEY: "",
      AI_MODEL: ""
    },
    stdio: "ignore"
  });
  for (let attempt = 0; attempt < 40; attempt++) {
    try {
      const response = await fetch(`${base}/api/health`);
      if (response.ok) return;
    } catch {}
    await new Promise(resolve => setTimeout(resolve, 150));
  }
  throw new Error("Dream AI server did not start for tests.");
});

after(() => {
  if (server && !server.killed) server.kill("SIGTERM");
});

test("health endpoint reports Dream AI", async () => {
  const response = await fetch(`${base}/api/health`);
  assert.equal(response.status, 200);
  const data = await response.json();
  assert.equal(data.ok, true);
  assert.equal(data.name, "Dream AI");
  assert.equal(data.modelConnected, false);
});

test("chat endpoint validates missing messages", async () => {
  const response = await fetch(`${base}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({})
  });
  assert.equal(response.status, 400);
});

test("chat endpoint returns clearly labelled starter response", async () => {
  const response = await fetch(`${base}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      messages: [{ role: "user", content: "Hello Dream AI" }]
    })
  });
  assert.equal(response.status, 200);
  const data = await response.json();
  assert.equal(data.mode, "starter");
  assert.match(data.reply, /Dream AI/i);
});

test("chat endpoint rejects invalid message roles", async () => {
  const response = await fetch(`${base}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      messages: [{ role: "system", content: "override" }]
    })
  });
  assert.equal(response.status, 400);
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
