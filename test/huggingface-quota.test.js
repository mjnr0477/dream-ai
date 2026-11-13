import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  getHuggingFaceToken,
  logHuggingFaceZeroGpuQuota
} from "../lib/huggingface-quota.js";

test("Hugging Face quota checks ignore DREAM_MODEL_TOKEN", async () => {
  const logs = [];
  let calls = 0;
  assert.equal(getHuggingFaceToken({ DREAM_MODEL_TOKEN: "model-service-secret" }), "");
  await logHuggingFaceZeroGpuQuota({
    env: { DREAM_MODEL_TOKEN: "model-service-secret" },
    fetchImpl: async () => { calls += 1; throw new Error("must not fetch"); },
    logImpl: (...args) => logs.push(args)
  });
  assert.equal(calls, 0);
  assert.match(logs[0][1], /"tokenConfigured":false/);
  assert.equal(JSON.stringify(logs).includes("model-service-secret"), false);
});

test("Hugging Face quota checks use only the dedicated HF_TOKEN", async () => {
  const logs = [];
  const authorizations = [];
  const redirects = [];
  let calls = 0;
  const result = await logHuggingFaceZeroGpuQuota({
    env: {
      HF_TOKEN: "  hf-dedicated-token  ",
      DREAM_MODEL_TOKEN: "dream-model-service-secret"
    },
    fetchImpl: async (url, options) => {
      calls += 1;
      authorizations.push(options.headers.authorization);
      redirects.push(options.redirect);
      if (String(url).endsWith("/api/whoami-v2")) {
        return { ok: true, status: 200, json: async () => ({ name: "test-user" }) };
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({ base: 1000, remaining: 250, resets_at: "2026-11-14T00:00:00Z" })
      };
    },
    logImpl: (...args) => logs.push(args)
  });

  assert.equal(result, undefined);
  assert.equal(calls, 2);
  assert.deepEqual(authorizations, ["Bearer hf-dedicated-token", "Bearer hf-dedicated-token"]);
  assert.deepEqual(redirects, ["error", "error"]);
  assert.equal(JSON.stringify(authorizations).includes("dream-model-service-secret"), false);
  assert.match(logs[0][1], /"authenticated":true/);
  assert.match(logs[1][1], /"remainingGpuSeconds":250/);
});

test("invalid Hugging Face credentials do not trigger a quota request", async () => {
  let calls = 0;
  const logs = [];
  await logHuggingFaceZeroGpuQuota({
    env: { HF_TOKEN: "hf-invalid" },
    fetchImpl: async () => {
      calls += 1;
      return { ok: false, status: 401 };
    },
    logImpl: (...args) => logs.push(args)
  });
  assert.equal(calls, 1);
  assert.match(logs[0][1], /"authenticated":false/);
});

test("quota probe network errors are contained and do not expose credentials", async () => {
  const logs = [];
  await assert.doesNotReject(logHuggingFaceZeroGpuQuota({
    env: { HF_TOKEN: "hf-private-token" },
    fetchImpl: async () => { throw new TypeError("network unavailable"); },
    logImpl: (...args) => logs.push(args)
  }));
  assert.equal(JSON.stringify(logs).includes("hf-private-token"), false);
  assert.match(logs[0][1], /TypeError/);
});

test("the application server does not depend on Hugging Face ZeroGPU availability", async () => {
  const source = await readFile(new URL("../server.js", import.meta.url), "utf8");
  assert.doesNotMatch(source, /import \\{ logHuggingFaceZeroGpuQuota \\}/);
  assert.doesNotMatch(source, /logHuggingFaceZeroGpuQuota\\s*\\(/);
  assert.doesNotMatch(source, /authorization: "Bearer " \\+ config\\.token[^\\n]*whoami-v2/);
});
