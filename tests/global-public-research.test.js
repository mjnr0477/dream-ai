import test from "node:test";
import assert from "node:assert/strict";
import { searchGlobalPublicSources } from "../lib/research-global.js";
import { shouldAutoResearch, buildResearchContext } from "../lib/auto-research.js";
import { getDreamCapabilities } from "../lib/capabilities.js";
import { createDreamToolRegistry, runPublicResearch } from "../lib/dream-tools.js";

function response(payload) {
  return { ok: true, json: async () => payload };
}

function makeFetch(fixtures, calls = []) {
  return async (url, options = {}) => {
    const parsed = new URL(url);
    calls.push({ url: parsed, options });
    const key = parsed.hostname;
    if (!(key in fixtures)) return { ok: false, status: 503, json: async () => ({}) };
    return response(fixtures[key]);
  };
}

test("global public research discovers entities and recent headlines", async () => {
  const calls = [];
  const fetchImpl = makeFetch({
    "www.wikidata.org": {
      search: [{
        id: "Q42",
        label: "Douglas Adams",
        description: "English writer and humorist"
      }]
    },
    "api.gdeltproject.org": {
      articles: [{
        title: "A recent public news report",
        url: "https://news.example.org/story",
        seendate: "20261108T093000Z",
        language: "English",
        sourcecountry: "Kenya"
      }]
    }
  }, calls);

  const result = await searchGlobalPublicSources("Douglas Adams", fetchImpl, {});
  assert.equal(result.entities.length, 1);
  assert.equal(result.entities[0].entityId, "Q42");
  assert.equal(result.news.length, 1);
  assert.equal(result.news[0].seenAt, "2026-11-08T09:30:00.000Z");
  assert.equal(result.webResults.length, 0);
  assert.equal(result.providers.find(p => p.provider === "Brave Search").status, "not-configured");
  assert.equal(calls.length, 2);
  assert.ok(calls.every(call => call.options.redirect === "error"));
});

test("optional Brave web search is used only with a server-side key", async () => {
  const calls = [];
  const fetchImpl = makeFetch({
    "www.wikidata.org": { search: [] },
    "api.gdeltproject.org": { articles: [] },
    "api.search.brave.com": {
      grounding: { generic: [{
        title: "Useful public page",
        snippets: ["A first extracted page passage.", "A second relevant passage."],
        url: "https://example.org/reference"
      }] },
      sources: {
        "https://example.org/reference": {
          hostname: "example.org",
          age: ["Sunday, November 8, 2026", "2026-11-08", "today", "2026-11-08T10:00:00Z"],
          fetched_content_timestamp: 1794132000
        }
      }
    }
  }, calls);
  const result = await searchGlobalPublicSources(
    "public information",
    fetchImpl,
    { DREAM_BRAVE_SEARCH_API_KEY: "test-only-secret" }
  );
  assert.equal(result.webResults.length, 1);
  const braveCall = calls.find(call => call.url.hostname === "api.search.brave.com");
  assert.equal(braveCall.options.headers["X-Subscription-Token"], "test-only-secret");
  assert.equal(braveCall.url.pathname, "/res/v1/llm/context");
  assert.ok(result.webResults[0].snippet.includes("A first extracted page passage."));
  assert.equal(result.webResults[0].sourceType, "extracted-web-page-context");
  assert.equal(result.providers.find(p => p.provider === "Brave Search").status, "ok");
});

test("unsafe source links are discarded and provider failure is isolated", async () => {
  const fetchImpl = async url => {
    const host = new URL(url).hostname;
    if (host === "www.wikidata.org") {
      return response({ search: [{
        id: "Q5",
        label: "Human",
        description: "A living person",
        concepturi: "javascript:alert(1)"
      }] });
    }
    if (host === "api.gdeltproject.org") {
      return response({ articles: [{
        title: "Unsafe",
        url: "javascript:alert(1)",
        seendate: "not-a-date"
      }] });
    }
    return { ok: false, status: 500, json: async () => ({}) };
  };
  const result = await searchGlobalPublicSources("human", fetchImpl, {});
  assert.equal(result.entities.length, 1);
  assert.equal(result.entities[0].url, "https://www.wikidata.org/wiki/Q5");
  assert.equal(result.news.length, 0);
});

test("global public search rejects empty and overlong queries", async () => {
  await assert.rejects(() => searchGlobalPublicSources("  ", async () => response({})), /1 to 200/);
  await assert.rejects(() => searchGlobalPublicSources("x".repeat(201), async () => response({})), /1 to 200/);
});

test("automatic research includes current and factual questions but skips creative prompts", () => {
  assert.equal(shouldAutoResearch("What is the latest news about Manchester City?"), true);
  assert.equal(shouldAutoResearch("Who is Wangari Maathai and what did she do?"), true);
  assert.equal(shouldAutoResearch("Taylor Swift"), true);
  assert.equal(shouldAutoResearch("Rihanna"), true);
  assert.equal(shouldAutoResearch("Write me a short story about a footballer."), false);
  assert.equal(shouldAutoResearch("What is a hash map?", "coding"), true);
  assert.equal(shouldAutoResearch("Explain photosynthesis step by step.", "learning"), true);
});

test("research context includes source links from all public provider types", () => {
  const prepared = buildResearchContext({
    results: [{
      title: "Wikipedia item",
      snippet: "An encyclopedia snippet",
      url: "https://en.wikipedia.org/wiki/Example"
    }],
    globalResearch: {
      webResults: [{
        provider: "Brave Search LLM Context",
        title: "Web result",
        snippet: "A public web excerpt",
        url: "https://example.org/page"
      }],
      news: [{
        provider: "GDELT",
        title: "Recent headline",
        domain: "example.net",
        seenAt: "2026-11-08T09:30:00.000Z",
        url: "https://example.net/news"
      }],
      entities: [{
        provider: "Wikidata",
        title: "Public figure",
        description: "A public entity entry",
        url: "https://www.wikidata.org/wiki/Q42"
      }]
    },
    scholarlyResearch: {
      results: [{
        provider: "Crossref",
        title: "A research paper",
        authors: ["A. Author"],
        publicationDate: "2025-01-01",
        url: "https://doi.org/10.0000/example"
      }]
    }
  });
  assert.equal(prepared.sources.length, 5);
  assert.ok(prepared.context.includes("Recent headline"));
  assert.ok(prepared.context.includes("Public figure"));
  assert.ok(prepared.context.includes("A research paper"));
  assert.ok(prepared.sources.every(source => source.url.startsWith("https://")));
});

test("research context includes public video and social-platform sources", () => {
  const prepared = buildResearchContext({
    platformResearch: {
      results: [
        {
          provider: "YouTube Data API",
          title: "Public video",
          snippet: "Public video description",
          url: "https://www.youtube.com/watch?v=abc123",
          publishedAt: "2026-11-08T10:00:00Z"
        },
        {
          provider: "Bluesky",
          title: "Public author · public post",
          snippet: "Public post text",
          url: "https://bsky.app/profile/public.example/post/post123"
        }
      ]
    }
  });
  assert.equal(prepared.sources.length, 2);
  assert.ok(prepared.context.includes("Public video description"));
  assert.ok(prepared.context.includes("Public post text"));
  assert.ok(prepared.sources.every(source => source.url.startsWith("https://")));
});

test("capabilities disclose general web-index availability truthfully", () => {
  const withoutKey = getDreamCapabilities({}, {});
  const webWithoutKey = withoutKey.tools.find(tool => tool.id === "web_research");
  assert.equal(webWithoutKey.status, "not-configured");
  const withKey = getDreamCapabilities({}, { DREAM_BRAVE_SEARCH_API_KEY: "configured" });
  const webWithKey = withKey.tools.find(tool => tool.id === "web_research");
  assert.equal(webWithKey.status, "available");
  assert.deepEqual(withKey.application.researchScope, [
    "multi-provider-public-source-discovery",
    "brave-web-context"
  ]);
});

test("research orchestration combines encyclopedia, global and scholarly providers", async () => {
  const fetchImpl = async url => {
    const parsed = new URL(url);
    if (parsed.hostname === "en.wikipedia.org") {
      if (parsed.searchParams.get("list") === "search") {
        return response({ query: { search: [{
          pageid: 42,
          title: "Douglas Adams",
          snippet: "Writer and humorist"
        }] } });
      }
      return response({ query: { pages: {
        "42": {
          pageid: 42,
          extract: "Douglas Adams was an English writer and humorist.",
          touched: "2026-10-01T10:00:00Z",
          fullurl: "https://en.wikipedia.org/wiki/Douglas_Adams"
        }
      } } });
    }
    if (parsed.hostname === "www.wikidata.org") {
      return response({ search: [{
        id: "Q42",
        label: "Douglas Adams",
        description: "English writer and humorist"
      }] });
    }
    if (parsed.hostname === "api.gdeltproject.org") {
      return response({ articles: [{
        title: "Recent report",
        url: "https://news.example.com/report",
        seendate: "20261108T080000Z"
      }] });
    }
    if (parsed.hostname === "api.crossref.org") {
      return response({ message: { items: [{
        DOI: "10.1234/dream-test",
        title: ["A bibliographic test paper"],
        author: [{ given: "Ada", family: "Lovelace" }],
        published: { "date-parts": [[2026, 1, 2]] }
      }] } });
    }
    if (parsed.hostname === "api.openalex.org") {
      return response({ results: [{
        id: "https://openalex.org/W123",
        doi: "https://doi.org/10.1234/dream-test",
        title: "A bibliographic test paper",
        publication_date: "2026-01-02",
        authorships: [],
        primary_location: { landing_page_url: "https://doi.org/10.1234/dream-test" }
      }] });
    }
    return { ok: false, status: 503, json: async () => ({}) };
  };
  const registry = createDreamToolRegistry({ fetchImpl, env: {}, audit: () => {} });
  const result = await runPublicResearch("Douglas Adams", registry);
  assert.equal(result.results.length, 1);
  assert.equal(result.globalResearch.entities[0].entityId, "Q42");
  assert.equal(result.globalResearch.news[0].provider, "GDELT");
  assert.equal(result.scholarlyResearch.results.length, 1);
  assert.equal(result.providerAvailability.encyclopedia, "available");
  assert.equal(result.providerAvailability.globalPublicSources, "available");
  assert.equal(result.providerAvailability.scholarly, "available");
});
