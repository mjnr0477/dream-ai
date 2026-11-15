import test from "node:test";
import assert from "node:assert/strict";
import { resolveRuntimeStatus } from "./runtime-status.js";

test("offline status never exposes a stale notebook GPU device", () => {
  assert.deepEqual(resolveRuntimeStatus({
    readyWorkerCount: 0,
    notebookComputeOnline: false,
    notebookChatWarmupEnabled: true,
    notebookChatModelReady: true,
    notebookChatModelDevice: "NVIDIA T4"
  }), {
    modelRuntime: "offline",
    modelReady: false,
    chatAvailable: false,
    device: null
  });
});

test("notebook device is reported only after a connected worker has a ready model", () => {
  const loading = resolveRuntimeStatus({
    notebookComputeOnline: true,
    notebookChatWarmupEnabled: true,
    notebookChatModelReady: false,
    notebookChatModelDevice: "NVIDIA T4"
  });
  assert.deepEqual(loading, {
    modelRuntime: "dream-notebook",
    modelReady: false,
    chatAvailable: true,
    device: null
  });

  const ready = resolveRuntimeStatus({
    notebookComputeOnline: true,
    notebookChatWarmupEnabled: true,
    notebookChatModelReady: true,
    notebookChatModelDevice: " NVIDIA T4 "
  });
  assert.deepEqual(ready, {
    modelRuntime: "dream-notebook",
    modelReady: true,
    chatAvailable: true,
    device: "NVIDIA T4"
  });
});

test("native GPU status uses the native worker device only", () => {
  assert.deepEqual(resolveRuntimeStatus({
    readyWorkerCount: 1,
    notebookComputeOnline: true,
    notebookChatWarmupEnabled: true,
    notebookChatModelReady: true,
    activeWorkerDevice: "NVIDIA L4",
    notebookChatModelDevice: "NVIDIA T4"
  }), {
    modelRuntime: "gpu-worker",
    modelReady: true,
    chatAvailable: true,
    device: "NVIDIA L4"
  });

  assert.equal(resolveRuntimeStatus({
    readyWorkerCount: 1,
    activeWorkerDevice: " "
  }).device, null);
});
