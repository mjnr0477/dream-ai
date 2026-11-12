import test from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:net";
import { once } from "node:events";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath } from "node:url";

const SERVER_PATH = fileURLToPath(new URL("./server.js", import.meta.url));
const MODEL = "Qwen/Qwen3-1.7B";
const ADAPTER = "msjnr0477/dream-ai-adapter";

async function getFreePort() {
  const server = createServer();
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("Could not reserve a test port.");
  await new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  return address.port;
}

async function waitForHost(baseUrl, child) {
  for (let attempt = 0; attempt < 40; attempt += 1) {
    if (child.exitCode !== null) throw new Error("DREAM HOST exited before becoming healthy.");
    try {
      const response = await fetch(baseUrl + "/health", { signal: AbortSignal.timeout(1000) });
      if (response.ok) return response;
    } catch {}
    await delay(150);
  }
  throw new Error("DREAM HOST did not become healthy within six seconds.");
}

test("DREAM HOST validates workers and delivers a real queued chat completion", { timeout: 30000 }, async t => {
  const port = await getFreePort();
  const baseUrl = `http://127.0.0.1:${port}`;
  const workerToken = "test-worker-token-0123456789-abcdefghijklmnopqrstuvwxyz";
  const clientToken = "test-client-token-0123456789-abcdefghijklmnopqrstuvwxyz";
  const workerId = "worker-test-0001";

  const child = spawn(process.execPath, [SERVER_PATH], {
    env: {
      ...process.env,
      PORT: String(port),
      DREAM_HOST_WORKER_TOKEN: workerToken,
      DREAM_HOST_CLIENT_TOKEN: clientToken,
      DREAM_HOST_MODEL_NAME: MODEL,
      DREAM_HOST_ADAPTER_NAME: ADAPTER
    },
    stdio: "ignore"
  });

  t.after(async () => {
    if (child.exitCode === null) {
      child.kill("SIGTERM");
      await Promise.race([once(child, "exit"), delay(2000)]);
      if (child.exitCode === null) child.kill("SIGKILL");
    }
  });

  await waitForHost(baseUrl, child);

  const initialHealth = await fetch(baseUrl + "/health").then(response => response.json());
  assert.equal(initialHealth.name, "DREAM HOST");
  assert.equal(initialHealth.version, "0.2.0");
  assert.equal(initialHealth.modelConfigured, true);
  assert.equal(initialHealth.modelReady, false);
  assert.equal(initialHealth.persistence, "in-memory");

  const notReady = await fetch(baseUrl + "/health/ready");
  assert.equal(notReady.status, 503);

  const unauthorized = await fetch(baseUrl + "/api/v1/workers/register", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ workerId })
  });
  assert.equal(unauthorized.status, 401);

  const modelMismatch = await fetch(baseUrl + "/api/v1/workers/register", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: "Bearer " + workerToken
    },
    body: JSON.stringify({
      workerId,
      metadata: { model: "wrong-model", adapter: ADAPTER, device: "test-cpu" }
    })
  });
  assert.equal(modelMismatch.status, 200);

  const mismatchedHealth = await fetch(baseUrl + "/health").then(response => response.json());
  assert.equal(mismatchedHealth.modelReady, false);
  assert.equal(mismatchedHealth.workers.connected, 1);
  assert.equal(mismatchedHealth.workers.ready, 0);

  const mismatchedClaim = await fetch(baseUrl + "/api/v1/workers/jobs/claim", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: "Bearer " + workerToken
    },
    body: JSON.stringify({ workerId })
  });
  assert.equal(mismatchedClaim.status, 409);

  const registration = await fetch(baseUrl + "/api/v1/workers/register", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: "Bearer " + workerToken
    },
    body: JSON.stringify({
      workerId,
      metadata: { model: MODEL, adapter: ADAPTER, device: "Test GPU" }
    })
  });
  assert.equal(registration.status, 200);

  const ready = await fetch(baseUrl + "/health/ready");
  assert.equal(ready.status, 200);
  const readyBody = await ready.json();
  assert.equal(readyBody.modelReady, true);
  assert.equal(readyBody.workers.ready, 1);

  const wrongClientToken = await fetch(baseUrl + "/v1/models", {
    headers: { authorization: "Bearer invalid-client-token" }
  });
  assert.equal(wrongClientToken.status, 401);

  const models = await fetch(baseUrl + "/v1/models", {
    headers: { authorization: "Bearer " + clientToken }
  });
  assert.equal(models.status, 200);
  assert.equal((await models.json()).data[0].id, MODEL);

  const chatPromise = fetch(baseUrl + "/v1/chat/completions", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: "Bearer " + clientToken
    },
    body: JSON.stringify({
      model: MODEL,
      messages: [{ role: "user", content: "Say hello." }],
      max_tokens: 32
    })
  });

  let claimedJob = null;
  for (let attempt = 0; attempt < 40 && !claimedJob; attempt += 1) {
    const claim = await fetch(baseUrl + "/api/v1/workers/jobs/claim", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: "Bearer " + workerToken
      },
      body: JSON.stringify({ workerId })
    });
    if (claim.status === 200) claimedJob = (await claim.json()).job;
    else {
      assert.equal(claim.status, 204);
      await delay(50);
    }
  }
  assert.ok(claimedJob, "the registered GPU worker should be able to claim a chat job");

  const result = await fetch(baseUrl + "/api/v1/workers/jobs/" + claimedJob.id + "/result", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: "Bearer " + workerToken
    },
    body: JSON.stringify({
      workerId,
      content: "Hello from the DREAM HOST integration test.",
      promptTokens: 4,
      completionTokens: 9
    })
  });
  assert.equal(result.status, 200);

  const completionResponse = await chatPromise;
  assert.equal(completionResponse.status, 200);
  const completion = await completionResponse.json();
  assert.equal(completion.object, "chat.completion");
  assert.equal(completion.model, MODEL);
  assert.equal(
    completion.choices[0].message.content,
    "Hello from the DREAM HOST integration test."
  );
});
