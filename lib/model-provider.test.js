import test from "node:test";
import assert from "node:assert/strict";
import { resolveModelConfig, requestChatCompletion, verifyGeminiModel } from "./model-provider.js";

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
  assert.throws(() => resolveModelConfig({ ...base, DREAM_MODEL_PROVIDER: "unknown" }), /must be native, openai-compatible or gemini/);
  assert.throws(() => resolveModelConfig({ ...base, DREAM_MODEL_URL: "https://user:pass@example.test/v1" }), /embedded credentials/);
  assert.throws(() => resolveModelConfig({ ...base, DREAM_MODEL_URL: "http://models.example.test/v1" }), /HTTPS/);
});

test("rejects missing model identity and invalid timeout configuration", () => {
  assert.throws(() => resolveModelConfig({ DREAM_MODEL_URL: base.DREAM_MODEL_URL }), /DREAM_MODEL_NAME/);
  assert.throws(() => resolveModelConfig({ ...base, DREAM_MODEL_TIMEOUT_MS: "999999" }), /TIMEOUT_MS/);
});

test("uses Gemini as a temporary fallback only when no model endpoint is configured", () => {
  const fallback = resolveModelConfig({ DREAM_GOOGLE_GEMINI_API_KEY: "server-only-key" });
  assert.equal(fallback.provider, "gemini");
  assert.equal(fallback.modelName, "gemini-3.8-flash");
  assert.equal(fallback.token, "server-only-key");

  const native = resolveModelConfig({
    ...base,
    DREAM_GOOGLE_GEMINI_API_KEY: "server-only-key"
  });
  assert.equal(native.provider, "native");

  assert.throws(() => resolveModelConfig({ DREAM_MODEL_PROVIDER: "gemini" }), /DREAM_GOOGLE_GEMINI_API_KEY is required/);
});

test("sends Gemini system instructions and conversation history securely", async () => {
  const config = resolveModelConfig({ DREAM_GOOGLE_GEMINI_API_KEY: "server-only-key" });
  let request;
  const reply = await requestChatCompletion({
    config,
    messages: [
      { role: "system", content: "You are DREAM AI." },
      { role: "user", content: "Explain photosynthesis." },
      { role: "assistant", content: "It is how plants make food." },
      { role: "user", content: "Give me an example." }
    ],
    temperature: 0.4,
    fetchImpl: async (url, options) => {
      request = { url: String(url), options };
      return {
        ok: true,
        json: async () => ({
          candidates: [{ content: { parts: [{ text: "  Green plants use light energy.  " }] } }]
        })
      };
    }
  });

  assert.equal(reply, "Green plants use light energy.");
  assert.equal(request.url, "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent");
  assert.equal(request.options.headers["x-goog-api-key"], "server-only-key");
  assert.equal(request.options.headers.authorization, undefined);
  const body = JSON.parse(request.options.body);
  assert.equal(body.systemInstruction.parts[0].text, "You are DREAM AI.");
  assert.deepEqual(body.contents.map(item => item.role), ["user", "model", "user"]);
  assert.equal(body.generationConfig.temperature, 0.4);
});

test("verifies that the selected Gemini model supports generateContent", async () => {
  const config = resolveModelConfig({ DREAM_GOOGLE_GEMINI_API_KEY: "server-only-key" });
  let request;
  const model = await verifyGeminiModel(config, async (url, options) => {
    request = { url: String(url), options };
    return {
      ok: true,
      json: async () => ({
        name: "models/gemini-3.8-flash",
        supportedGenerationMethods: ["generateContent", "countTokens"]
      })
    };
  });
  assert.equal(model.name, "models/gemini-3.8-flash");
  assert.equal(request.url, "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash");
  assert.equal(request.options.headers["x-goog-api-key"], "server-only-key");
  await assert.rejects(
    verifyGeminiModel(config, async () => ({
      ok: true,
      json: async () => ({ name: "models/gemini-3.8-flash", supportedGenerationMethods: ["countTokens"] })
    })),
    /does not support content generation/
  );
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
