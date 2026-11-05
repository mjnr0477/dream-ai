import test from "node:test";
import assert from "node:assert/strict";
import { summarizeRows } from "./summarize-evals.mjs";

function result(id, language, overrides = {}) {
  return {
    runId: "run-001",
    runStartedAt: "2026-11-05T00:00:00.000Z",
    evaluationSet: "core-v1",
    model: "dream-native",
    checkpointSha256: "a".repeat(64),
    completedTrainingSteps: 1200,
    runtimeDevice: "cpu",
    runtimeParameters: 250000,
    tokenizer: "UTF8ByteTokenizer",
    runtimeArchitecture: "{\"d_model\":16}",
    id,
    category: "multilingual",
    language,
    latencyMs: 120,
    humanScore: 3,
    ...overrides
  };
}

test("summarizes one checkpoint run with language breakdown and provenance", () => {
  const summary = summarizeRows([
    result("english-001", "en"),
    result("swahili-001", "sw")
  ]);
  assert.equal(summary.run.runId, "run-001");
  assert.equal(summary.run.checkpointSha256, "a".repeat(64));
  assert.equal(summary.byLanguage.en.averageHumanScore, 3);
  assert.equal(summary.byLanguage.sw.averageHumanScore, 3);
});

test("rejects rows from different runs or checkpoints", () => {
  assert.throws(
    () => summarizeRows([
      result("case-1", "en"),
      result("case-2", "sw", { checkpointSha256: "b".repeat(64) })
    ]),
    /different benchmark runs or checkpoints/
  );
  assert.throws(
    () => summarizeRows([
      result("case-1", "en"),
      result("case-2", "sw", { runId: "run-002" })
    ]),
    /different benchmark runs or checkpoints/
  );
});

test("retains compatibility with older human-scored result files", () => {
  const summary = summarizeRows([{
    id: "legacy-001",
    category: "reasoning",
    language: null,
    latencyMs: 50,
    humanScore: 4
  }]);
  assert.equal(summary.run, null);
  assert.equal(summary.overall.averageHumanScore, 4);
});
