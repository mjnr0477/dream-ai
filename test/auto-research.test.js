import test from "node:test";
import assert from "node:assert/strict";
import { buildLiveNewsContext, buildResearchContext, isTimeSensitiveResearchIntent, isUnsupportedLiveDataIntent, shouldAutoResearch } from "../lib/auto-research.js";

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


test("routes recent news and sports questions to the recent-news search, not static references", () => {
  assert.equal(isTimeSensitiveResearchIntent("What is the latest news on Manchester City's performance?"), true);
  assert.equal(isTimeSensitiveResearchIntent("What is the live score?"), false);
  assert.equal(isUnsupportedLiveDataIntent("What is the live score?"), true);
  assert.equal(isUnsupportedLiveDataIntent("What is the current weather?"), true);
  assert.equal(isUnsupportedLiveDataIntent("What is the USD exchange rate right now?"), true);
  assert.equal(isUnsupportedLiveDataIntent("Write me a story about the live score"), false);
  assert.equal(isUnsupportedLiveDataIntent("What was Manchester City's latest result?"), false);
  assert.equal(isTimeSensitiveResearchIntent("Who won yesterday?"), true);
  assert.equal(isTimeSensitiveResearchIntent("What is the weather today?"), false);
  assert.equal(isTimeSensitiveResearchIntent("What is the exchange rate right now?"), false);
  assert.equal(isTimeSensitiveResearchIntent("Write me a story about the latest match"), false);
  assert.equal(isTimeSensitiveResearchIntent("What is the latest news?", "learning"), false);
  assert.equal(shouldAutoResearch("What is the latest news on Manchester City?"), false);
});

test("builds a bounded recent-news context with publisher, timestamps and safe citations", () => {
  const now = new Date("2026-11-08T12:00:00Z");
  const prepared = buildLiveNewsContext({
    retrievedAt: "2026-11-08T11:59:00Z",
    results: [
      {
        title: "Manchester City lose 3-1 away",
        publisher: "Example Sport",
        publishedAt: "2026-11-08T10:30:00Z",
        url: "https://news.google.com/rss/articles/CBMiExample",
        excerpt: "A recent report about the match."
      },
      {
        title: "Unsafe link",
        publisher: "Unknown",
        publishedAt: "2026-11-08T10:00:00Z",
        url: "https://attacker.example/story",
        excerpt: "Must not be included."
      },
      {
        title: "Old report",
        publisher: "Old News",
        publishedAt: "2026-10-01T10:00:00Z",
        url: "https://news.google.com/rss/articles/CBMiOld"
      }
    ]
  }, now);
  assert.equal(prepared.sources.length, 1);
  assert.equal(prepared.sources[0].kind, "news");
  assert.equal(prepared.sources[0].publisher, "Example Sport");
  assert.match(prepared.context, /retrieved at 2026-11-08T11:59:00.000Z UTC/);
  assert.match(prepared.context, /PUBLISHED AT \(UTC\): 2026-11-08T10:30:00.000Z/);
  assert.match(prepared.context, /not a guaranteed live news feed, live match clock, or official scoreboard/);
  assert.doesNotMatch(prepared.context, /attacker\.example|Old report/);
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


test("automatic research context combines encyclopedia excerpts with safe scholarly metadata", () => {
  const prepared = buildResearchContext({
    results: [{
      title: "Climate adaptation",
      url: "https://en.wikipedia.org/wiki/Climate_adaptation",
      extract: "Climate adaptation reduces vulnerability to climate change."
    }],
    scholarlyResearch: {
      results: [
        {
          provider: "Crossref",
          title: "Climate adaptation evidence review",
          authors: ["A. Researcher", "B. Scientist"],
          publicationDate: "2025-02-01",
          url: "https://doi.org/10.1234/climate"
        },
        {
          provider: "OpenAlex",
          title: "Unsafe publication link",
          authors: [],
          publicationDate: "2024",
          url: "https://attacker.example/paper"
        },
        {
          provider: "UnknownProvider",
          title: "Unknown provider is not trusted",
          authors: [],
          publicationDate: "2024",
          url: "https://openalex.org/W123"
        }
      ]
    }
  });

  assert.equal(prepared.sources.length, 2);
  assert.deepEqual(prepared.sources.map(source => source.kind), [
    "encyclopedia", "scholarly-metadata"
  ]);
  assert.equal(prepared.sources[1].provider, "Crossref");
  assert.match(prepared.context, /Climate adaptation reduces vulnerability/);
  assert.match(prepared.context, /SCHOLARLY PUBLICATION METADATA ONLY/);
  assert.match(prepared.context, /AUTHORS: A\. Researcher; B\. Scientist/);
  assert.match(prepared.context, /PUBLICATION DATE: 2025-02-01/);
  assert.match(prepared.context, /do not establish a paper's methods, results, or conclusions/);
  assert.doesNotMatch(prepared.context, /attacker\.example|Unknown provider/);
});

test("scholarly-only results can support a research context without encyclopedia results", () => {
  const prepared = buildResearchContext({
    scholarlyResearch: {
      results: [{
        provider: "OpenAlex",
        title: "Swahili language technology survey",
        authors: [],
        publicationDate: "2023",
        url: "https://openalex.org/W123"
      }]
    }
  });
  assert.equal(prepared.sources.length, 1);
  assert.equal(prepared.sources[0].kind, "scholarly-metadata");
  assert.match(prepared.context, /Swahili language technology survey/);
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
