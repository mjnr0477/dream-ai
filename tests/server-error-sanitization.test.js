import test, { before, after } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import net from "node:net";

let child;
let baseUrl;

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
    if (child.exitCode !== null) throw new Error("DREAM server exited before becoming ready");
    try {
      const response = await fetch(url + "/api/health/live");
      if (response.ok) return;
    } catch {}
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  throw new Error("DREAM server did not become ready");
}

before(async () => {
  const port = await freePort();
  baseUrl = `http://127.0.0.1:${port}`;
  child = spawn(process.execPath, ["server.js"], {
    cwd: process.cwd(),
    env: {
      ...process.env,
      PORT: String(port),
      DREAM_MODEL_URL: "http://invalid-runtime.example",
      DREAM_MODEL_NAME: "security-test-model"
    },
    stdio: "ignore"
  });
  await waitForServer(baseUrl);
});

after(() => {
  if (child && child.exitCode === null) child.kill("SIGTERM");
});

test("chat preserves safe runtime diagnostics without exposing configured URLs", async () => {
  const response = await fetch(baseUrl + "/api/chat", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      origin: baseUrl,
      "x-forwarded-for": "198.51.100.111"
    },
    body: JSON.stringify({
      mode: "general",
      messages: [{ role: "user", content: "Hello" }]
    })
  });

  assert.equal(response.status, 502);
  const body = await response.json();
  assert.equal(body.error, "DREAM_MODEL_URL must use HTTPS unless it targets loopback for local development.");
  assert.equal(JSON.stringify(body).includes("invalid-runtime.example"), false);
});
