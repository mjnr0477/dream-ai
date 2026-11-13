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



test("chat rejects a same-host Origin with the wrong scheme", async () => {
  const response = await fetch(baseUrl + "/api/chat", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      origin: baseUrl.replace(/^http:/, "https:")
    },
    body: JSON.stringify(validPayload)
  });
  assert.equal(response.status, 403);
});

test("chat and research share a bounded per-client rate limit", async () => {
  const clientIp = "198.51.100.20";
  const headers = {
    "content-type": "application/json",
    origin: baseUrl,
    "cf-connecting-ip": clientIp,
    "x-forwarded-for": "203.0.113.250, 10.0.0.1"
  };
  for (let index = 0; index < 20; index += 1) {
    const response = await fetch(baseUrl + "/api/chat", {
      method: "POST",
      headers,
      body: JSON.stringify(validPayload)
    });
    assert.equal(response.status, 200);
  }

  const blockedResearch = await fetch(baseUrl + "/api/research", {
    method: "POST",
    headers,
    body: JSON.stringify({ query: "DREAM AI" })
  });
  assert.equal(blockedResearch.status, 429);
  assert.ok(Number(blockedResearch.headers.get("retry-after")) >= 1);

  const otherClient = await fetch(baseUrl + "/api/chat", {
    method: "POST",
    headers: { ...headers, "cf-connecting-ip": "198.51.100.21" },
    body: JSON.stringify(validPayload)
  });
  assert.equal(otherClient.status, 200);
});

test("API and error responses include baseline security headers", async () => {
  const health = await fetch(baseUrl + "/api/health");
  assert.equal(health.headers.get("x-frame-options"), "DENY");
  assert.equal(health.headers.get("x-content-type-options"), "nosniff");

  const missing = await fetch(baseUrl + "/does-not-exist");
  assert.equal(missing.status, 404);
  assert.equal(missing.headers.get("x-frame-options"), "DENY");

  const method = await fetch(baseUrl + "/api/health/live", { method: "PUT" });
  assert.equal(method.status, 405);
  assert.equal(method.headers.get("x-content-type-options"), "nosniff");
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
  assert.equal(body.source, "offline-status");
  assert.match(body.reply, /language-model worker is currently offline/i);
});

test("public health hides checkpoint and hardware diagnostics", async () => {
  const response = await fetch(baseUrl + "/api/health");
  const body = await response.json();
  assert.equal(response.status, 200);
  assert.equal("checkpointSha256" in body, false);
  assert.equal("device" in body, false);
  assert.equal("parameters" in body, false);
});

test("static pages receive restrictive security headers", async () => {
  const response = await fetch(baseUrl + "/");
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("x-frame-options"), "DENY");
  assert.equal(response.headers.get("x-content-type-options"), "nosniff");
  assert.match(response.headers.get("content-security-policy") || "", /default-src 'self'/);
  assert.match(response.headers.get("content-security-policy") || "", /frame-ancestors 'none'/);
  assert.match(response.headers.get("content-security-policy") || "", /style-src-attr 'unsafe-inline'/);
});
