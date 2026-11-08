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

test("Google grounding retains secure publisher URLs returned by Google Search", () => {
  const prepared = buildResearchContext({
    query: "latest technology news",
    platformResearch: { results: [
      { provider: "Google Search", title: "Publisher report", url: "https://www.bbc.com/news/technology-123", snippet: "A sourced current report." },
      { provider: "Google Search", title: "Insecure link", url: "http://www.example.com/report", snippet: "Must be excluded." },
      { provider: "Google Search", title: "Credential link", url: "https://user:pass@example.com/report", snippet: "Must be excluded." }
    ] }
  });
  assert.equal(prepared.sources.length, 1);
  assert.equal(prepared.sources[0].provider, "Google Search");
  assert.equal(prepared.sources[0].url, "https://www.bbc.com/news/technology-123");
  assert.match(prepared.context, /A sourced current report/);
  assert.doesNotMatch(prepared.context, /Must be excluded/);
});

test("time-sensitive research context balances providers instead of letting Wikipedia crowd out current sources", () => {
  const prepared = buildResearchContext({
    query: "What are the latest developments today?",
    results: Array.from({ length: 5 }, (_, index) => ({
      title: "Encyclopedia source " + index,
      url: "https://en.wikipedia.org/wiki/Encyclopedia_" + index,
      extract: "A".repeat(500)
    })),
    globalResearch: {
      webResults: [{
        provider: "Brave Search LLM Context",
        title: "Independent web source",
        url: "https://www.example.org/current-report",
        snippet: "A relevant source from the web index."
      }],
      news: [{
        provider: "GDELT",
        title: "Recent publisher report",
        url: "https://www.bbc.com/news/current-report",
        domain: "bbc.com",
        seenAt: "2026-11-08T10:00:00Z"
      }]
    },
    platformResearch: { results: [{
      provider: "Google Search",
      title: "Google-grounded publisher",
      url: "https://www.reuters.com/world/current-report",
      snippet: "Google-grounded current reporting."
    }] }
  });
  const providers = new Set(prepared.sources.map(source => source.provider));
  assert.ok(providers.has("Google Search"));
  assert.ok(providers.has("Brave Search LLM Context"));
  assert.ok(providers.has("GDELT"));
  assert.ok(providers.has("Wikipedia"));
  assert.ok(prepared.sources.length <= 14);
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
