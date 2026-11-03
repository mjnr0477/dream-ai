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
