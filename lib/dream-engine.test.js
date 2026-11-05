import test from "node:test";
import assert from "node:assert/strict";
import { buildSystemPrompt, getDreamGenerationOptions } from "./dream-engine.js";

test("coding mode gives actionable implementation guidance", () => {
  const prompt = buildSystemPrompt("coding");
  for (const requirement of [
    "complete, runnable files",
    "Match the requested language",
    "complete files labeled with repository paths",
    "do not imply that you inspected the repository",
    "Match the requested language",
    "complete files labeled with repository paths",
    "do not imply that you inspected the repository",
    "inspect the relevant code",
    "root cause",
    "regression",
    "parameterize database queries",
    "untrusted data",
    "Do not invent libraries",
    "Never claim a patch was applied"
  ]) {
    assert.ok(prompt.includes(requirement), `Missing coding instruction: ${requirement}`);
  }
});

test("unknown modes safely fall back to general behavior", () => {
  const prompt = buildSystemPrompt("not-a-real-mode");
  assert.ok(prompt.includes("Answer clearly and helpfully"));
  assert.ok(!prompt.includes("Act as a senior software engineer"));
});

test("general and learning modes remain distinct", () => {
  assert.ok(buildSystemPrompt("general").includes("distinguish verified facts"));
  assert.ok(buildSystemPrompt("learning").includes("patient expert tutor"));
});

test("coding mode uses lower-variance sampling without changing other modes", () => {
  assert.deepEqual(getDreamGenerationOptions("coding"), { temperature: 0.2 });
  assert.deepEqual(getDreamGenerationOptions("general"), { temperature: 0.7 });
  assert.deepEqual(getDreamGenerationOptions("learning"), { temperature: 0.7 });
  assert.deepEqual(getDreamGenerationOptions("unknown"), { temperature: 0.7 });
});
