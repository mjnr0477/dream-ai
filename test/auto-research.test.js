import test from "node:test";
import assert from "node:assert/strict";
import { buildResearchContext, shouldAutoResearch } from "../lib/auto-research.js";

test("automatically researches stable factual questions in normal chat", () => {
  assert.equal(shouldAutoResearch("Tell me about Manchester City"), true);
  assert.equal(shouldAutoResearch("Who is Ada Lovelace?"), true);
  assert.equal(shouldAutoResearch("What is photosynthesis?"), true);
  assert.equal(shouldAutoResearch("Manchester City team"), true);
});

test("does not use static research sources for live or time-sensitive questions", () => {
  assert.equal(shouldAutoResearch("What are the latest Manchester City results?"), false);
  assert.equal(shouldAutoResearch("What is the score today?"), false);
  assert.equal(shouldAutoResearch("Who is leading the league right now?"), false);
});

test("does not research creative, personal, short or specialist-mode requests", () => {
  assert.equal(shouldAutoResearch("Hi Dream AI"), false);
  assert.equal(shouldAutoResearch("Write me a short story about a moon robot"), false);
  assert.equal(shouldAutoResearch("What do you think about my plan?"), false);
  assert.equal(shouldAutoResearch("Who is Ada Lovelace?", "coding"), false);
  assert.equal(shouldAutoResearch("What is photosynthesis?", "learning"), false);
});

test("research context accepts only bounded encyclopedia source URLs and text", () => {
  const prepared = buildResearchContext({ results: [
    { title: "Manchester City F.C.", url: "https://en.wikipedia.org/wiki/Manchester_City_F.C.", snippet: "A football club based in Manchester." },
    { title: "Unsafe destination", url: "https://attacker.example/redirect", extract: "Must be excluded." },
    { title: "Unsafe scheme", url: "javascript:alert(1)", extract: "Must be excluded." }
  ] });
  assert.equal(prepared.sources.length, 1);
  assert.equal(prepared.sources[0].title, "Manchester City F.C.");
  assert.match(prepared.context, /A football club based in Manchester/);
  assert.doesNotMatch(prepared.context, /attacker|javascript:/);
});

test("research context enforces a strict evidence size limit", () => {
  const prepared = buildResearchContext({ results: Array.from({ length: 5 }, (_, index) => ({
    title: "Source " + index,
    url: "https://en.wikipedia.org/wiki/Source_" + index,
    extract: "A".repeat(620)
  })) });
  assert.ok(prepared.context.length <= 4000);
  assert.ok(prepared.sources.length <= 5);
});
