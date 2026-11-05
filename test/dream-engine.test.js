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
