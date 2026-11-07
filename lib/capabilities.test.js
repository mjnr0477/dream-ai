import test from "node:test";
import assert from "node:assert/strict";
import { getDreamCapabilities } from "./capabilities.js";

test("capabilities distinguish native DREAM and compatible upstream models", () => {
  const native = getDreamCapabilities({
    configured: true,
    ready: true,
    provider: "native",
    modelName: "dream-native-v1"
  });
  assert.equal(native.model.id, "dream-native-v1");
  assert.equal(native.model.type, "independent-native-transformer");
  assert.equal(native.model.status, "ready");

  const compatible = getDreamCapabilities({
    configured: true,
    ready: true,
    provider: "openai-compatible",
    modelName: "Qwen/Qwen3-1.7B"
  });
  assert.equal(compatible.model.id, "Qwen/Qwen3-1.7B");
  assert.equal(compatible.model.type, "compatible-upstream-model");
  assert.equal(compatible.model.status, "ready");
});

test("capabilities do not label an unverified or unconfigured runtime as native", () => {
  const unverified = getDreamCapabilities({
    configured: true,
    ready: false,
    status: "unavailable"
  });
  assert.equal(unverified.model.type, "unverified");
  assert.equal(unverified.model.status, "unavailable");

  const unconfigured = getDreamCapabilities({
    configured: false,
    ready: false
  });
  assert.equal(unconfigured.model.type, "not-configured");
  assert.equal(unconfigured.model.status, "not-configured");
});
