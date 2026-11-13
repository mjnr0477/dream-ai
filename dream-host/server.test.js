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
  const notebookToken = "test-notebook-token-0123456789-abcdefghijklmnopqrstuvwxyz";
  const notebookWorkerToken = "test-notebook-worker-token-0123456789-abcdefghijklmnopqrstuvwxyz";
  const workerId = "worker-test-0001";

  const child = spawn(process.execPath, [SERVER_PATH], {
    env: {
      ...process.env,
      PORT: String(port),
      DREAM_HOST_WORKER_TOKEN: workerToken,
      DREAM_HOST_CLIENT_TOKEN: clientToken,
      DREAM_HOST_NOTEBOOK_TOKEN: notebookToken,
      DREAM_HOST_NOTEBOOK_WORKER_TOKEN: notebookWorkerToken,
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
  assert.equal(initialHealth.version, "0.3.0");
  assert.equal(initialHealth.modelConfigured, true);
  assert.equal(initialHealth.modelReady, false);
  assert.equal(initialHealth.persistence, "in-memory");

  const notebookPage = await fetch(baseUrl + "/notebook");
  assert.equal(notebookPage.status, 200);
  const notebookHtml = await notebookPage.text();
  assert.match(notebookHtml, /DREAM NOTEBOOK/);
  assert.match(notebookHtml, /Load native DREAM training/);
  assert.match(notebookHtml, /Persistent GPU setup guide/);
  assert.doesNotMatch(notebookHtml, /Kaggle|Qwen adapter|HF_TOKEN|notebook\/chat-worker/);
  assert.match(notebookPage.headers.get("content-security-policy"), /script-src 'self'/);

  const notebookScript = await fetch(baseUrl + "/notebook.js");
  assert.equal(notebookScript.status, 200);
  assert.match(notebookScript.headers.get("content-type"), /javascript/);
  const notebookScriptText = await notebookScript.text();
  assert.match(notebookScriptText, /indexedDB/);
  assert.match(notebookScriptText, /DREAM AI chat/);
  assert.doesNotMatch(notebookScriptText, /kaggle|HF_TOKEN|importAdapterWorkflow|adapter-template/i);

  const notebookStyle = await fetch(baseUrl + "/notebook.css");
  assert.equal(notebookStyle.status, 200);
  assert.match(notebookStyle.headers.get("content-type"), /css/);

  const trainingTemplate = await fetch(baseUrl + "/notebook/training-template.ipynb");
  assert.equal(trainingTemplate.status, 200);
  const trainingNotebook = await trainingTemplate.json();
  assert.equal(trainingNotebook.nbformat, 4);
  assert.ok(trainingNotebook.cells.length >= 10);
  const nativeTrainingSource = trainingNotebook.cells
    .map(cell => Array.isArray(cell.source) ? cell.source.join("") : String(cell.source || ""))
    .join("\n");
  assert.match(nativeTrainingSource, /native, from-scratch DREAM training workflow/i);
  assert.ok(nativeTrainingSource.includes("dream-native-pretrain-checkpoint.pt"));
  assert.equal(nativeTrainingSource.includes("PRODUCTION_ADAPTER_REPO"), false);
  assert.equal(nativeTrainingSource.includes("Qwen/Qwen3-1.7B"), false);

  const setupGuide = await fetch(baseUrl + "/notebook/permanent-gpu-setup.md");
  assert.equal(setupGuide.status, 200);
  assert.match(setupGuide.headers.get("content-type"), /text\/markdown/);
  const setupText = await setupGuide.text();
  assert.match(setupText, /Cloudflare R2/);
  assert.match(setupText, /dream-native-inference-worker/);

  for (const retiredPath of ["/notebook/adapter-template.ipynb", "/notebook/kaggle-worker.ipynb", "/notebook/chat-worker.ipynb"]) {
    const retired = await fetch(baseUrl + retiredPath);
    assert.equal(retired.status, 404);
  }
  const connectResponse = await fetch(baseUrl + "/api/v1/notebook/session", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: "Bearer " + notebookToken
    },
    body: JSON.stringify({ remember: true })
  });
  assert.equal(connectResponse.status, 200);
  const cookie = connectResponse.headers.get("set-cookie");
  assert.match(cookie, /dream_notebook_session=/);
  assert.match(cookie, /HttpOnly/i);
  assert.match(cookie, /Secure/i);
  assert.match(cookie, /SameSite=Strict/i);
  assert.match(cookie, /Max-Age=2592000/i);

  const cookieAuthorizedStatus = await fetch(baseUrl + "/api/v1/notebook/status", {
    headers: { cookie: cookie.split(";")[0] }
  });
  assert.equal(cookieAuthorizedStatus.status, 200);

  const disconnected = await fetch(baseUrl + "/api/v1/notebook/session", { method: "DELETE" });
  assert.equal(disconnected.status, 200);
  assert.match(disconnected.headers.get("set-cookie"), /Max-Age=0/i);
  const cookieAfterDisconnect = await fetch(baseUrl + "/api/v1/notebook/status", {
    headers: { cookie: "dream_notebook_session=" }
  });
  assert.equal(cookieAfterDisconnect.status, 401);

  const notebookUnauthorized = await fetch(baseUrl + "/api/v1/notebook/status");
  assert.equal(notebookUnauthorized.status, 401);

  const notebookStatusResponse = await fetch(baseUrl + "/api/v1/notebook/status", {
    headers: { authorization: "Bearer " + notebookToken }
  });
  assert.equal(notebookStatusResponse.status, 200);
  const notebookStatus = await notebookStatusResponse.json();
  assert.equal(notebookStatus.workers.connected, 0);
  assert.equal(notebookStatus.queue.pending, 0);

  const notebookJobResponse = await fetch(baseUrl + "/api/v1/notebook/jobs", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: "Bearer " + notebookToken
    },
    body: JSON.stringify({
      notebookId: "notebook-test-01",
      cellId: "cell-test-01",
      code: "print('notebook integration')"
    })
  });
  assert.equal(notebookJobResponse.status, 202);
  const notebookJob = (await notebookJobResponse.json()).job;
  assert.equal(notebookJob.status, "queued");

  const chatWorkerTokenRejected = await fetch(baseUrl + "/api/v1/notebook/workers/register", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: "Bearer " + workerToken
    },
    body: JSON.stringify({
      workerId: "notebook-worker-unauthorized",
      metadata: { role: "notebook", deviceType: "cpu" }
    })
  });
  assert.equal(chatWorkerTokenRejected.status, 401);

  const notebookWorkerRegistration = await fetch(baseUrl + "/api/v1/notebook/workers/register", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: "Bearer " + notebookWorkerToken
    },
    body: JSON.stringify({
      workerId: "notebook-worker-01",
      name: "Integration notebook worker",
      metadata: { role: "notebook", device: "Test CPU", deviceType: "cpu", python: "3.11" }
    })
  });
  assert.equal(notebookWorkerRegistration.status, 200);

  const notebookClaimResponse = await fetch(baseUrl + "/api/v1/notebook/workers/jobs/claim", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: "Bearer " + notebookWorkerToken
    },
    body: JSON.stringify({ workerId: "notebook-worker-01" })
  });
  assert.equal(notebookClaimResponse.status, 200);
  const claimedNotebookJob = (await notebookClaimResponse.json()).job;
  assert.equal(claimedNotebookJob.id, notebookJob.id);
  assert.equal(claimedNotebookJob.status, "running");
  assert.equal(claimedNotebookJob.code, "print('notebook integration')");

  const notebookResultResponse = await fetch(baseUrl + "/api/v1/notebook/workers/jobs/" + notebookJob.id + "/result", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: "Bearer " + notebookWorkerToken
    },
    body: JSON.stringify({
      workerId: "notebook-worker-01",
      stdout: "notebook integration\n",
      stderr: "",
      display: "None",
      executionCount: 1
    })
  });
  assert.equal(notebookResultResponse.status, 200);

  const notebookCompletedResponse = await fetch(baseUrl + "/api/v1/notebook/jobs/" + notebookJob.id, {
    headers: { authorization: "Bearer " + notebookToken }
  });
  assert.equal(notebookCompletedResponse.status, 200);
  const notebookCompleted = (await notebookCompletedResponse.json()).job;
  assert.equal(notebookCompleted.status, "completed");
  assert.equal(notebookCompleted.result.stdout, "notebook integration\n");

  // Notebook execution is training-only by default and must not queue the legacy Qwen/HF warm-up.
  const noWarmupClaim = await fetch(baseUrl + "/api/v1/notebook/workers/jobs/claim", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: "Bearer " + notebookWorkerToken
    },
    body: JSON.stringify({ workerId: "notebook-worker-01" })
  });
  assert.equal(noWarmupClaim.status, 204);

  const notReady = await fetch(baseUrl + "/health/ready");
  assert.equal(notReady.status, 503);
  const notebookReadyBody = await notReady.json();
  assert.equal(notebookReadyBody.modelReady, false);
  assert.equal(notebookReadyBody.chatAvailable, false);
  assert.equal(notebookReadyBody.modelRuntime, "offline");
  assert.equal(notebookReadyBody.notebookChatWarmupEnabled, false);
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
  assert.equal(mismatchedHealth.modelRuntime, "offline");
  assert.equal(mismatchedHealth.notebookChatWarmupEnabled, false);
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

  // A training-only notebook worker must not be asked to load the legacy Qwen/HF chat model.
  const trainingOnlyChatResponse = await fetch(baseUrl + "/v1/chat/completions", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: "Bearer " + clientToken
    },
    body: JSON.stringify({
      model: MODEL,
      messages: [{ role: "user", content: "Reply with a greeting." }],
      max_tokens: 32,
      temperature: 0
    })
  });
  assert.equal(trainingOnlyChatResponse.status, 503);
  const trainingOnlyChatError = await trainingOnlyChatResponse.json();
  assert.match(trainingOnlyChatError.error.message, /connected for training and notebook execution/i);

  const noLegacyChatJob = await fetch(baseUrl + "/api/v1/notebook/workers/jobs/claim", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: "Bearer " + notebookWorkerToken
    },
    body: JSON.stringify({ workerId: "notebook-worker-01" })
  });
  assert.equal(noLegacyChatJob.status, 204);

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

test("DREAM NOTEBOOK fails closed without a distinct access token while HOST still starts", { timeout: 15000 }, async t => {
  const port = await getFreePort();
  const baseUrl = `http://127.0.0.1:${port}`;
  const workerToken = "test-worker-token-0123456789-abcdefghijklmnopqrstuvwxyz";
  const clientToken = "test-client-token-0123456789-abcdefghijklmnopqrstuvwxyz";
  const child = spawn(process.execPath, [SERVER_PATH], {
    env: {
      ...process.env,
      PORT: String(port),
      DREAM_HOST_WORKER_TOKEN: workerToken,
      DREAM_HOST_CLIENT_TOKEN: clientToken,
      DREAM_HOST_NOTEBOOK_TOKEN: workerToken,
      DREAM_HOST_NOTEBOOK_WORKER_TOKEN: "test-notebook-worker-token-0123456789-abcdefghijklmnopqrstuvwxyz",
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
  const health = await fetch(baseUrl + "/health");
  assert.equal(health.status, 200);

  const disabled = await fetch(baseUrl + "/api/v1/notebook/status", {
    headers: { authorization: "Bearer " + workerToken }
  });
  assert.equal(disabled.status, 503);
  const body = await disabled.json();
  assert.match(body.error.message, /distinct private value/);
});

