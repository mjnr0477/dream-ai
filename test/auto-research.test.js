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

test("researches factual questions in general, coding and learning modes", () => {
  for (const mode of ["general", "coding", "learning"]) {
    assert.equal(shouldAutoResearch("Who is Ada Lovelace?", mode), true, mode);
    assert.equal(shouldAutoResearch("How do JavaScript promises work?", mode), true, mode);
    assert.equal(shouldAutoResearch("What are the latest Python releases today?", mode), true, mode);
  }
});

test("does not research creative, personal, greeting or unsupported-mode requests", () => {
  assert.equal(shouldAutoResearch("Hi Dream AI"), false);
  assert.equal(shouldAutoResearch("Write me a short story about a moon robot"), false);
  assert.equal(shouldAutoResearch("What do you think about my plan?"), false);
  assert.equal(shouldAutoResearch("What is photosynthesis?", "unknown"), false);
});

test("Google's aggregate grounding synthesis is not attributed to an individual publisher", () => {
  const answer = "A grounded summary that draws on multiple search results.";
  const prepared = buildResearchContext({
    query: "latest technology update",
    platformResearch: {
      googleGroundedAnswer: answer,
      results: [
        { provider: "Google Search", title: "Publisher one", url: "https://www.bbc.com/news/one", snippet: "" },
        { provider: "Google Search", title: "Publisher two", url: "https://www.reuters.com/world/two", snippet: "" }
      ]
    }
  });
  assert.equal(prepared.context.split(answer).length - 1, 1);
  assert.match(prepared.context, /not a verbatim excerpt or a claim attributed to any single listed publisher/);
  assert.match(prepared.context, /https:\/\/www\.bbc\.com\/news\/one/);
  assert.match(prepared.context, /https:\/\/www\.reuters\.com\/world\/two/);
  assert.ok(prepared.sources.every(source => source.provider === "Google Search"));
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

test("PubMed abstracts are accepted from the official PubMed host and cited as abstracts", () => {
  const prepared = buildResearchContext({
    query: "What does recent PubMed research say about malaria prevention?",
    platformResearch: {
      results: [
        {
          provider: "PubMed",
          title: "Malaria prevention study",
          url: "https://pubmed.ncbi.nlm.nih.gov/12345678/",
          snippet: "The abstract reports a reduction in infection in the study population.",
          sourceType: "pubmed-abstract",
          publishedAt: "2025"
        },
        {
          provider: "PubMed",
          title: "Untrusted PubMed lookalike",
          url: "https://pubmed.ncbi.nlm.nih.gov.attacker.example/123/",
          snippet: "Must be excluded."
        }
      ]
    }
  });
  assert.equal(prepared.sources.length, 1);
  assert.equal(prepared.sources[0].provider, "PubMed");
  assert.match(prepared.context, /abstract reports a reduction/);
  assert.match(prepared.context, /PubMed abstracts are summaries, not full-text articles/);
  assert.doesNotMatch(prepared.context, /attacker\\.example|Must be excluded/);
});


test("arXiv and Semantic Scholar abstracts enter context only through their trusted source hosts", () => {
  const prepared = buildResearchContext({
    scholarlyResearch: { results: [
      {
        provider: "arXiv",
        title: "A preprint about reliable AI",
        url: "https://arxiv.org/abs/2601.01234v1",
        publicationDate: "2026-01-12",
        abstract: "An abstract retrieved from the arXiv API."
      },
      {
        provider: "Semantic Scholar",
        title: "A peer-reviewed paper about verification",
        url: "https://www.semanticscholar.org/paper/Verification/0123456789abcdef",
        publicationDate: "2025",
        abstract: "An abstract retrieved from Semantic Scholar."
      },
      {
        provider: "arXiv",
        title: "Untrusted destination",
        url: "https://attacker.example/abs/2601.99999",
        abstract: "This must not enter context."
      }
    ] }
  });
  assert.equal(prepared.sources.length, 2);
  assert.ok(prepared.sources.some(source => source.provider === "arXiv"));
  assert.ok(prepared.sources.some(source => source.provider === "Semantic Scholar"));
  assert.match(prepared.context, /An abstract retrieved from the arXiv API/);
  assert.match(prepared.context, /An abstract retrieved from Semantic Scholar/);
  assert.doesNotMatch(prepared.context, /attacker\.example|This must not enter context/);
});


test("GitHub repository metadata enters coding research only from github.com", () => {
  const prepared = buildResearchContext({
    query: "Find a Python CSV library",
    platformResearch: { results: [
      {
        provider: "GitHub Public Repositories",
        title: "example/csv-parser",
        url: "https://github.com/example/csv-parser",
        snippet: "A small Python CSV parser.",
        sourceType: "public-repository-metadata"
      },
      {
        provider: "GitHub Public Repositories",
        title: "unsafe/repository",
        url: "https://attacker.example/unsafe",
        snippet: "Untrusted URL."
      }
    ] }
  });
  assert.equal(prepared.sources.length, 1);
  assert.equal(prepared.sources[0].provider, "GitHub Public Repositories");
  assert.match(prepared.context, /A small Python CSV parser/);
  assert.doesNotMatch(prepared.context, /attacker\.example|Untrusted URL/);
});
