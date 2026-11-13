import test from "node:test";
import assert from "node:assert/strict";
import { summarizeRows } from "../scripts/summarize-evals.mjs";

const sample = [
  { id: "sw-1", category: "multilingual", language: "sw", humanScore: 4, latencyMs: 120 },
  { id: "sw-2", category: "multilingual", language: "sw", humanScore: 2, latencyMs: 180 },
  { id: "math-1", category: "reasoning", language: null, humanScore: null, latencyMs: 80 }
];

test("summarizes human scores by category and language without inventing missing scores", () => {
  const result = summarizeRows(sample);
  assert.deepEqual(result.overall, {
    cases: 3, scored: 2, unscored: 1, averageHumanScore: 3, averageLatencyMs: 127
  });
  assert.equal(result.byCategory.multilingual.averageHumanScore, 3);
  assert.equal(result.byLanguage.sw.scored, 2);
  assert.equal(result.byLanguage.sw.averageLatencyMs, 150);
  assert.equal(result.byCategory.reasoning.averageHumanScore, null);
});

test("rejects scores outside the documented 0-4 scale", () => {
  assert.throws(() => summarizeRows([{ ...sample[0], humanScore: 5 }]), /0 to 4/);
});

test("rejects duplicate case identifiers", () => {
  assert.throws(() => summarizeRows([sample[0], sample[0]]), /Duplicate evaluation id/);
});

test("rejects malformed language tags and latency", () => {
  assert.throws(() => summarizeRows([{ ...sample[0], language: "English" }]), /language tag/);
  assert.throws(() => summarizeRows([{ ...sample[0], latencyMs: -1 }]), /latencyMs/);
});

test("requires a non-empty result set", () => {
  assert.throws(() => summarizeRows([]), /at least one result/);
});


const applicationRun = sample.map(row => ({
  ...row,
  evaluationSource: "full-application-chat",
  runId: "chat-run-1",
  runStartedAt: "2026-11-13T14:00:00.000Z",
  evaluationSet: "core-v1",
  applicationOrigin: "https://dream-ai-live.onrender.com",
  applicationVersion: "0.3.0",
  runtimeProvider: "native"
}));

test("summarizes full-application chat results without requiring private checkpoint metadata", () => {
  const result = summarizeRows(applicationRun);
  assert.equal(result.run.evaluationSource, "full-application-chat");
  assert.equal(result.run.runId, "chat-run-1");
  assert.equal(result.run.applicationOrigin, "https://dream-ai-live.onrender.com");
  assert.equal(result.run.runtimeProvider, "native");
  assert.equal(result.overall.cases, 3);
  assert.equal(result.byLanguage.sw.averageHumanScore, 3);
});

test("rejects incomplete or mixed full-application evaluation provenance", () => {
  assert.throws(
    () => summarizeRows([{ ...applicationRun[0], runtimeProvider: undefined }]),
    /Incomplete application evaluation provenance/
  );
  assert.throws(
    () => summarizeRows([applicationRun[0], { ...applicationRun[1], runId: "different-run" }]),
    /Cannot combine different application evaluation runs/
  );
});

test("does not combine native checkpoint and public application evaluation rows", () => {
  const native = {
    ...sample[0],
    id: "native-sw",
    runId: "native-run",
    runStartedAt: "2026-11-13T14:00:00.000Z",
    evaluationSet: "core-v1",
    model: "dream-native",
    checkpointSha256: "a".repeat(64),
    completedTrainingSteps: 100,
    runtimeDevice: "cpu",
    runtimeParameters: 1000,
    tokenizer: "dream-bpe",
    runtimeArchitecture: "{}"
  };
  assert.throws(
    () => summarizeRows([native, applicationRun[0]]),
    /Cannot combine native-model and full-application/
  );
});
