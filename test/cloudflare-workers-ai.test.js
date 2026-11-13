import test from "node:test";
import assert from "node:assert/strict";
import { resolveModelConfig, requestChatCompletion } from "../lib/model-provider.js";
import { getDreamRuntimeStatus, isDreamEngineConfigured, generateDreamReply } from "../lib/dream-engine.js";

const accountId = "a".repeat(32);
const token = "cloudflare-test-token";

test("Cloudflare Workers AI configuration builds the account-scoped chat endpoint", () => {
  const config = resolveModelConfig({
    DREAM_MODEL_PROVIDER: "cloudflare-workers-ai",
    CLOUDFLARE_ACCOUNT_ID: accountId,
    CLOUDFLARE_API_TOKEN: token
  });

  assert.equal(config.provider, "cloudflare-workers-ai");
  assert.equal(config.modelName, "@cf/meta/llama-3.1-8b-instruct");
  assert.equal(
    config.url,
    `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/v1/chat/completions`
  );
  assert.equal(config.token, token);
});

test("Cloudflare Workers AI rejects missing or malformed credentials", () => {
  assert.throws(
    () => resolveModelConfig({ DREAM_MODEL_PROVIDER: "cloudflare-workers-ai", CLOUDFLARE_API_TOKEN: token }),
    /CLOUDFLARE_ACCOUNT_ID/
  );
  assert.throws(
    () => resolveModelConfig({ DREAM_MODEL_PROVIDER: "cloudflare-workers-ai", CLOUDFLARE_ACCOUNT_ID: "bad", CLOUDFLARE_API_TOKEN: token }),
    /32-character Cloudflare account ID/
  );
  assert.throws(
    () => resolveModelConfig({ DREAM_MODEL_PROVIDER: "cloudflare-workers-ai", CLOUDFLARE_ACCOUNT_ID: accountId }),
    /CLOUDFLARE_API_TOKEN/
  );
});

test("Cloudflare Workers AI sends authenticated OpenAI-compatible chat requests", async () => {
  const config = resolveModelConfig({
    DREAM_MODEL_PROVIDER: "cloudflare-workers-ai",
    DREAM_MODEL_NAME: "@cf/meta/llama-3.1-8b-instruct",
    CLOUDFLARE_ACCOUNT_ID: accountId,
    CLOUDFLARE_API_TOKEN: token
  });
  let request;
  const reply = await requestChatCompletion({
    config,
    messages: [{ role: "user", content: "Reply with DREAM AI connected." }],
    temperature: 0.2,
    fetchImpl: async (url, options) => {
      request = { url, options };
      return {
        ok: true,
        json: async () => ({ choices: [{ message: { content: "DREAM AI connected." } }] })
      };
    }
  });

  assert.equal(reply, "DREAM AI connected.");
  assert.equal(request.url, config.url);
  assert.equal(request.options.method, "POST");
  assert.equal(request.options.redirect, "error");
  assert.equal(request.options.headers.authorization, `Bearer ${token}`);
  const body = JSON.parse(request.options.body);
  assert.equal(body.model, config.modelName);
  assert.equal(body.messages[0].content, "Reply with DREAM AI connected.");
});

test("DREAM runtime reports Cloudflare as configured without pretending a live inference test ran", async () => {
  const env = {
    DREAM_MODEL_PROVIDER: "cloudflare-workers-ai",
    CLOUDFLARE_ACCOUNT_ID: accountId,
    CLOUDFLARE_API_TOKEN: token
  };
  assert.equal(isDreamEngineConfigured(env), true);
  const status = await getDreamRuntimeStatus(env);
  assert.equal(status.configured, true);
  assert.equal(status.ready, true);
  assert.equal(status.status, "configured");
  assert.equal(status.provider, "cloudflare-workers-ai");
});

test("DREAM chat uses Cloudflare Workers AI without calling a native runtime health endpoint", async () => {
  const originalFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async (url, options) => {
    calls += 1;
    assert.match(String(url), /api\.cloudflare\.com\/client\/v4\/accounts\//);
    assert.equal(options.method, "POST");
    return {
      ok: true,
      json: async () => ({ choices: [{ message: { content: "Cloudflare inference is working." } }] })
    };
  };

  try {
    const result = await generateDreamReply(
      [{ role: "user", content: "Test connection" }],
      "general",
      {
        DREAM_MODEL_PROVIDER: "cloudflare-workers-ai",
        CLOUDFLARE_ACCOUNT_ID: accountId,
        CLOUDFLARE_API_TOKEN: token
      }
    );
    assert.equal(result.reply, "Cloudflare inference is working.");
    assert.equal(result.source, "cloudflare-workers-ai");
    assert.equal(calls, 1);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
