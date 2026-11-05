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
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error("DREAM server did not become ready");
}

before(async () => {
  const port = await freePort();
  baseUrl = `http://127.0.0.1:${port}`;
  child = spawn(process.execPath, ["server.js"], {
    cwd: process.cwd(),
    env: { ...process.env, PORT: String(port), DREAM_MODEL_URL: "", DREAM_MODEL_NAME: "" },
    stdio: "ignore"
  });
  await waitForServer(baseUrl);
});

after(() => {
  if (child && child.exitCode === null) child.kill("SIGTERM");
});

const validPayload = {
  mode: "general",
  messages: [{ role: "user", content: "Hello DREAM" }]
};

test("chat rejects missing or non-JSON content type", async () => {
  for (const contentType of [undefined, "text/plain"]) {
    const headers = contentType ? { "content-type": contentType } : {};
    const response = await fetch(baseUrl + "/api/chat", {
      method: "POST",
      headers,
      body: JSON.stringify(validPayload)
    });
    assert.equal(response.status, 415);
    assert.match((await response.json()).error, /application\/json/i);
  }
});

test("research rejects a hostile Origin before contacting public sources", async () => {
  const response = await fetch(baseUrl + "/api/research", {
    method: "POST",
    headers: { "content-type": "application/json", origin: "https://attacker.example" },
    body: JSON.stringify({ query: "DREAM AI" })
  });
  assert.equal(response.status, 403);
});

test("research requires JSON content type", async () => {
  const response = await fetch(baseUrl + "/api/research", {
    method: "POST",
    headers: { "content-type": "text/plain" },
    body: JSON.stringify({ query: "DREAM AI" })
  });
  assert.equal(response.status, 415);
});

test("chat rejects a hostile Origin header", async () => {
  const response = await fetch(baseUrl + "/api/chat", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      origin: "https://attacker.example"
    },
    body: JSON.stringify(validPayload)
  });
  assert.equal(response.status, 403);
});

test("chat rejects cross-site Fetch Metadata", async () => {
  const response = await fetch(baseUrl + "/api/chat", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "sec-fetch-site": "cross-site"
    },
    body: JSON.stringify(validPayload)
  });
  assert.equal(response.status, 403);
});

test("same-origin JSON chat remains available without a connected model", async () => {
  const response = await fetch(baseUrl + "/api/chat", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      origin: baseUrl,
      "sec-fetch-site": "same-origin"
    },
    body: JSON.stringify(validPayload)
  });
  assert.equal(response.status, 200);
  const body = await response.json();
  assert.equal(body.source, "dream-engine-unconfigured");
  assert.match(body.reply, /model runtime is not connected/i);
});

test("public health and capabilities hide sensitive model diagnostics", async () => {
  const healthResponse = await fetch(baseUrl + "/api/health");
  const health = await healthResponse.json();
  assert.equal(healthResponse.status, 200);
  assert.equal("checkpointSha256" in health, false);
  assert.equal("device" in health, false);
  assert.equal("parameters" in health, false);
  assert.equal("runtimeStatus" in health, false);

  const capabilitiesResponse = await fetch(baseUrl + "/api/capabilities");
  const capabilities = await capabilitiesResponse.json();
  assert.equal(capabilitiesResponse.status, 200);
  assert.equal("completedTrainingSteps" in capabilities.model, false);
  assert.equal("checkpointSha256" in capabilities.model, false);
});

test("static pages receive restrictive security headers", async () => {
  const response = await fetch(baseUrl + "/");
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("x-frame-options"), "DENY");
  assert.equal(response.headers.get("x-content-type-options"), "nosniff");
  assert.match(response.headers.get("content-security-policy") || "", /default-src 'self'/);
  assert.match(response.headers.get("content-security-policy") || "", /frame-ancestors 'none'/);
  assert.match(response.headers.get("content-security-policy") || "", /style-src-attr 'unsafe-inline'/);
  assert.equal(response.headers.get("permissions-policy"), "camera=(), microphone=(), geolocation=()");

  const apiResponse = await fetch(baseUrl + "/api/health");
  assert.equal(apiResponse.headers.get("x-frame-options"), "DENY");
  assert.equal(apiResponse.headers.get("cache-control"), "no-store");
});
