import test from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:net";
import { once } from "node:events";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";

const SERVER_PATH = fileURLToPath(new URL("./server.js", import.meta.url));

async function freePort() {
  const server = createServer();
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const port = server.address().port;
  await new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  return port;
}

async function waitForHost(url, child) {
  for (let attempt = 0; attempt < 40; attempt += 1) {
    if (child.exitCode !== null) throw new Error("DREAM HOST exited before health check.");
    try {
      const response = await fetch(url + "/health", { signal: AbortSignal.timeout(1000) });
      if (response.ok) return;
    } catch {}
    await delay(150);
  }
  throw new Error("DREAM HOST did not become healthy.");
}

test("DREAM artifact vault authenticates uploads and serves content-addressed artifacts", { timeout: 15000 }, async t => {
  const port = await freePort();
  const root = await mkdtemp(path.join(tmpdir(), "dream-host-vault-"));
  const baseUrl = "http://127.0.0.1:" + port;
  const tokens = {
    worker: "test-worker-token-0123456789-abcdefghijklmnopqrstuvwxyz",
    client: "test-client-token-0123456789-abcdefghijklmnopqrstuvwxyz",
    notebook: "test-notebook-token-0123456789-abcdefghijklmnopqrstuvwxyz",
    notebookWorker: "test-notebook-worker-token-0123456789-abcdefghijklmnopqrstuvwxyz",
    artifact: "test-artifact-token-0123456789-abcdefghijklmnopqrstuvwxyz"
  };
  const child = spawn(process.execPath, [SERVER_PATH], {
    env: {
      ...process.env,
      PORT: String(port),
      DREAM_HOST_WORKER_TOKEN: tokens.worker,
      DREAM_HOST_CLIENT_TOKEN: tokens.client,
      DREAM_HOST_NOTEBOOK_TOKEN: tokens.notebook,
      DREAM_HOST_NOTEBOOK_WORKER_TOKEN: tokens.notebookWorker,
      DREAM_HOST_ARTIFACT_TOKEN: tokens.artifact,
      DREAM_HOST_ARTIFACTS_DIR: root,
      DREAM_HOST_MODEL_NAME: "Qwen/Qwen3-1.7B",
      DREAM_HOST_ADAPTER_NAME: "msjnr0477/dream-ai-adapter"
    },
    stdio: "ignore"
  });
  t.after(async () => {
    if (child.exitCode === null) {
      child.kill("SIGTERM");
      await Promise.race([once(child, "exit"), delay(1500)]);
      if (child.exitCode === null) child.kill("SIGKILL");
    }
    await rm(root, { recursive: true, force: true });
  });

  await waitForHost(baseUrl, child);
  const payload = Buffer.from("private DREAM model artifact payload");
  const denied = await fetch(baseUrl + "/api/v1/notebook/artifacts/dream-checkpoint.zip", {
    method: "PUT",
    body: payload
  });
  assert.equal(denied.status, 401);

  const upload = await fetch(baseUrl + "/api/v1/notebook/artifacts/dream-checkpoint.zip", {
    method: "PUT",
    headers: { authorization: "Bearer " + tokens.artifact, "content-type": "application/octet-stream" },
    body: payload
  });
  assert.equal(upload.status, 201);
  const artifact = (await upload.json()).artifact;
  assert.equal(artifact.size_bytes, payload.length);
  assert.match(artifact.sha256, /^[a-f0-9]{64}$/);

  const listing = await fetch(baseUrl + "/api/v1/notebook/artifacts", {
    headers: { authorization: "Bearer " + tokens.artifact }
  });
  assert.equal(listing.status, 200);
  assert.equal((await listing.json()).artifacts.length, 1);

  const download = await fetch(baseUrl + "/api/v1/notebook/artifacts/" + artifact.sha256, {
    headers: { authorization: "Bearer " + tokens.artifact }
  });
  assert.equal(download.status, 200);
  assert.equal(download.headers.get("x-artifact-sha256"), artifact.sha256);
  assert.deepEqual(Buffer.from(await download.arrayBuffer()), payload);
});
