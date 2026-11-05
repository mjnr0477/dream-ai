import test from "node:test";
import assert from "node:assert/strict";
import { generateDreamReply } from "../lib/dream-engine.js";

const messages = [{ role: "user", content: "Hello DREAM AI" }];

test("remote native model runtimes must use HTTPS", async () => {
  await assert.rejects(
    generateDreamReply(messages, "general", {
      DREAM_MODEL_URL: "http://model.example.com/v1",
      DREAM_MODEL_NAME: "dream-test-model"
    }),
    /DREAM_MODEL_URL must use HTTPS/
  );
});

test("native model runtime URLs reject embedded credentials", async () => {
  await assert.rejects(
    generateDreamReply(messages, "general", {
      DREAM_MODEL_URL: "https://user:secret@model.example.com/v1",
      DREAM_MODEL_NAME: "dream-test-model"
    }),
    /must not contain embedded credentials/
  );
});

test("native model runtime URLs reject non-HTTP protocols", async () => {
  await assert.rejects(
    generateDreamReply(messages, "general", {
      DREAM_MODEL_URL: "ftp://model.example.com/v1",
      DREAM_MODEL_NAME: "dream-test-model"
    }),
    /must use HTTPS/
  );
});

test("native runtime requests reject redirects instead of following them", async () => {
  const originalFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (_url, options) => {
    calls.push(options);
    if (options.method === "GET") {
      return {
        ok: true,
        json: async () => ({
          ok: true,
          engine: "dream-native-transformer",
          completed_training_steps: 1,
          checkpoint_sha256: "a".repeat(64),
          parameters: 100,
          model: "dream-test-model"
        })
      };
    }
    return {
      ok: true,
      json: async () => ({ choices: [{ message: { content: "DREAM test reply" } }] })
    };
  };

  try {
    const result = await generateDreamReply(messages, "general", {
      DREAM_MODEL_URL: "https://model.example.com/v1",
      DREAM_MODEL_NAME: "dream-test-model"
    });
    assert.equal(result.reply, "DREAM test reply");
    assert.equal(calls.length, 2);
    assert.ok(calls.every((options) => options.redirect === "error"));
  } finally {
    globalThis.fetch = originalFetch;
  }
});
