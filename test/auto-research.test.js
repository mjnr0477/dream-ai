import test from "node:test";
import assert from "node:assert/strict";
import { buildResearchContext, shouldAutoResearch } from "../lib/auto-research.js";

test("automatically researches stable factual questions in normal chat", () => {
  assert.equal(shouldAutoResearch("Tell me about Manchester City"), true);
  assert.equal(shouldAutoResearch("Who is Ada Lovelace?"), true);
  assert.equal(shouldAutoResearch("What is photosynthesis?"), true);
  assert.equal(shouldAutoResearch("Manchester City team"), true);
});

test("routes current and time-sensitive questions to fresh public-source research", () => {
  assert.equal(shouldAutoResearch("What are the latest Manchester City results?"), true);
  assert.equal(shouldAutoResearch("What is the score today?"), true);
  assert.equal(shouldAutoResearch("Who is leading the league right now?"), true);
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


test("Stack Exchange evidence is accepted only from the Stack Overflow host", () => {
  const prepared = buildResearchContext({ platformResearch: { results: [
    {
      provider: "Stack Exchange",
      title: "A public programming answer",
      url: "https://stackoverflow.com/questions/123456/example",
      snippet: "A concise answer excerpt."
    },
    {
      provider: "Stack Exchange",
      title: "Untrusted host",
      url: "https://attacker.example/questions/123",
      snippet: "Must not be included."
    }
  ] } });
  assert.equal(prepared.sources.length, 1);
  assert.equal(prepared.sources[0].provider, "Stack Exchange");
  assert.match(prepared.context, /A concise answer excerpt/);
  assert.doesNotMatch(prepared.context, /attacker\.example/);
});
