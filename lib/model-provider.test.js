import test from "node:test";
import assert from "node:assert/strict";
import { resolveModelConfig, requestChatCompletion } from "./model-provider.js";

const base = {
  DREAM_MODEL_NAME: "test-model",
  DREAM_MODEL_URL: "http://127.0.0.1:8080/v1"
};

test("native provider remains the default and resolves a local endpoint", () => {
  const config = resolveModelConfig(base);
  assert.equal(config.provider, "native");
  assert.equal(config.url, "http://127.0.0.1:8080/v1/chat/completions");
});

test("explicit compatible provider supports local and hosted HTTPS endpoints", () => {
  const local = resolveModelConfig({ ...base, DREAM_MODEL_PROVIDER: "openai-compatible" });
  const hosted = resolveModelConfig({
    DREAM_MODEL_PROVIDER: "openai-compatible",
    DREAM_MODEL_NAME: "hosted-model",
    DREAM_MODEL_URL: "https://models.example.test/v1",
    DREAM_MODEL_TOKEN: "server-only-token"
  });
  assert.equal(local.provider, "openai-compatible");
  assert.equal(hosted.url, "https://models.example.test/v1/chat/completions");
  assert.equal(hosted.token, "server-only-token");
});

test("rejects unsupported providers, credentials in URLs and insecure remote endpoints", () => {
  assert.throws(() => resolveModelConfig({ ...base, DREAM_MODEL_PROVIDER: "unknown" }), /must be native or openai-compatible/);
  assert.throws(() => resolveModelConfig({ ...base, DREAM_MODEL_URL: "https://user:pass@example.test/v1" }), /embedded credentials/);
  assert.throws(() => resolveModelConfig({ ...base, DREAM_MODEL_URL: "http://models.example.test/v1" }), /HTTPS/);
});

test("rejects missing model identity and invalid timeout configuration", () => {
  assert.throws(() => resolveModelConfig({ DREAM_MODEL_URL: base.DREAM_MODEL_URL }), /DREAM_MODEL_NAME/);
  assert.throws(() => resolveModelConfig({ ...base, DREAM_MODEL_TIMEOUT_MS: "999999" }), /TIMEOUT_MS/);
});

test("sends the configured model and returns a validated completion", async () => {
  const config = resolveModelConfig({ ...base, DREAM_MODEL_PROVIDER: "openai-compatible", DREAM_MODEL_TOKEN: "secret" });
  let request;
  const reply = await requestChatCompletion({
    config, messages: [{ role: "user", content: "hello" }], temperature: 0.2,
    fetchImpl: async (url, options) => {
      request = { url, options };
      return { ok: true, json: async () => ({ choices: [{ message: { content: "  hello back  " } }] }) };
    }
  });
  assert.equal(reply, "hello back");
  assert.equal(request.url, config.url);
  assert.equal(request.options.headers.authorization, "Bearer secret");
  assert.equal(JSON.parse(request.options.body).model, "test-model");
  assert.equal(request.options.redirect, "error");
});

test("surfaces provider HTTP failures and rejects empty completions", async () => {
  const config = resolveModelConfig({ ...base, DREAM_MODEL_PROVIDER: "openai-compatible" });
  await assert.rejects(requestChatCompletion({
    config, messages: [], temperature: 0.7,
    fetchImpl: async () => ({ ok: false, status: 503 })
  }), /HTTP 503/);
  await assert.rejects(requestChatCompletion({
    config, messages: [], temperature: 0.7,
    fetchImpl: async () => ({ ok: true, json: async () => ({ choices: [] }) })
  }), /empty response/);
});


test("serializes concurrent model requests for single-generation runtimes", async () => {
  const config = resolveModelConfig(base);
  let active = 0;
  let maximumActive = 0;
  let requestNumber = 0;
  const fetchImpl = async () => {
    active += 1;
    maximumActive = Math.max(maximumActive, active);
    const current = ++requestNumber;
    await new Promise(resolve => setTimeout(resolve, 20));
    active -= 1;
    return {
      ok: true,
      json: async () => ({ choices: [{ message: { content: `reply ${current}` } }] })
    };
  };
  const request = () => requestChatCompletion({
    config,
    messages: [{ role: "user", content: "hello" }],
    temperature: 0,
    fetchImpl
  });

  const replies = await Promise.all([request(), request()]);
  assert.equal(maximumActive, 1);
  assert.deepEqual(replies.sort(), ["reply 1", "reply 2"]);
});
