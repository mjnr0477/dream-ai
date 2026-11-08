import assert from "node:assert/strict";
import test from "node:test";
import { createDreamToolRegistry, runPublicKnowledgeSearch, runPublicResearch } from "../lib/dream-tools.js";

const principal = {
  id: "test-public-research",
  permissions: ["research:public"]
};

function responseWith(entries) {
  return new Response(JSON.stringify({ query: { search: entries } }), {
    status: 200,
    headers: { "content-type": "application/json" }
  });
}

function responseWithPages(pages) {
  return new Response(JSON.stringify({ query: { pages } }), {
    status: 200,
    headers: { "content-type": "application/json" }
  });
}

test("public knowledge search is registered as a bounded read-only tool", () => {
  const registry = createDreamToolRegistry({ fetchImpl: async () => responseWith([]) });
  const [tool] = registry.list();
  assert.equal(tool.id, "public_knowledge_search");
  assert.equal(tool.risk, "read");
  assert.equal(tool.permission, "research:public");
  assert.equal(tool.timeoutMs, 8000);
  assert.match(tool.description, /Wikipedia/);
});

test("registered research tool returns cleaned source-linked results without requiring approval", async () => {
  const requestedUrls = [];
  const requestOptions = [];
  const auditEvents = [];
  let calls = 0;
  const registry = createDreamToolRegistry({
    audit: event => auditEvents.push(event),
    fetchImpl: async (url, options) => {
      requestedUrls.push(new URL(url));
      requestOptions.push(options);
      calls += 1;
      if (calls === 1) {
        return responseWith([
          { pageid: 1, title: "DREAM AI", snippet: "<b>Dream</b> AI &amp; public research" },
          { pageid: 2, title: "Second source", snippet: "A second public result." }
        ]);
      }
      return responseWithPages({
        "1": {
          pageid: 1,
          extract: "Dream AI is a public artificial intelligence project.",
          touched: "2026-11-01T12:00:00Z",
          fullurl: "https://en.wikipedia.org/wiki/DREAM_AI"
        },
        "2": {
          pageid: 2,
          extract: "A second public encyclopedia introduction.",
          touched: "2026-11-02T12:00:00Z",
          fullurl: "https://en.wikipedia.org/wiki/Second_source"
        }
      });
    }
  });

  const execution = await registry.execute({
    toolId: "public_knowledge_search",
    input: { query: "DREAM AI" },
    principal
  });

  assert.equal(calls, 2);
  assert.equal(requestedUrls[0].hostname, "en.wikipedia.org");
  assert.equal(requestedUrls[0].pathname, "/w/api.php");
  assert.equal(requestedUrls[0].searchParams.get("list"), "search");
  assert.equal(requestedUrls[1].searchParams.get("prop"), "extracts|info");
  assert.equal(requestOptions.every(options => options.redirect === "error"), true);
  assert.equal(execution.toolId, "public_knowledge_search");
  assert.equal(execution.result.provider, "Wikipedia");
  assert.equal(execution.result.evidenceLevel, "encyclopedia-introduction");
  assert.equal(execution.result.results.length, 2);
  assert.equal(execution.result.results[0].snippet, "Dream AI & public research");
  assert.equal(execution.result.results[0].extract, "Dream AI is a public artificial intelligence project.");
  assert.equal(execution.result.results[0].url, "https://en.wikipedia.org/wiki/DREAM_AI");
  assert.equal(auditEvents.length, 1);
  assert.equal(auditEvents[0].status, "succeeded");
  assert.equal(Object.hasOwn(auditEvents[0], "query"), false);
  assert.equal(Object.hasOwn(auditEvents[0], "input"), false);
});

test("public research tool denies principals without the server permission", async () => {
  let called = false;
  const auditEvents = [];
  const registry = createDreamToolRegistry({
    audit: event => auditEvents.push(event),
    fetchImpl: async () => {
      called = true;
      return responseWith([]);
    }
  });

  await assert.rejects(
    registry.execute({
      toolId: "public_knowledge_search",
      input: { query: "DREAM AI" },
      principal: { id: "untrusted", permissions: [] }
    }),
    /not authorized/
  );
  assert.equal(called, false);
  assert.equal(auditEvents[0].status, "denied");
  assert.equal(auditEvents[0].reason, "permission-denied");
});

test("public research tool rejects oversized or unexpected input before upstream access", async () => {
  let called = false;
  const registry = createDreamToolRegistry({
    fetchImpl: async () => {
      called = true;
      return responseWith([]);
    }
  });

  await assert.rejects(
    registry.execute({ toolId: "public_knowledge_search", input: { query: "x".repeat(201) }, principal }),
    /too long/
  );
  await assert.rejects(
    registry.execute({ toolId: "public_knowledge_search", input: { query: "DREAM AI", url: "https://example.com" }, principal }),
    /unexpected property/
  );
  assert.equal(called, false);
});


test("research orchestration combines encyclopedia excerpts with scholarly metadata for reasoning", async () => {
  const registry = createDreamToolRegistry({
    audit: () => {},
    fetchImpl: async url => {
      const parsed = new URL(url);
      if (parsed.hostname === "en.wikipedia.org" && parsed.searchParams.get("list") === "search") {
        return responseWith([{ pageid: 12, title: "Climate adaptation", snippet: "A public overview." }]);
      }
      if (parsed.hostname === "en.wikipedia.org") {
        return responseWithPages({ "12": {
          pageid: 12, extract: "A short encyclopedia introduction.",
          touched: "2026-10-01T00:00:00Z",
          fullurl: "https://en.wikipedia.org/wiki/Climate_adaptation"
        }});
      }
      if (parsed.hostname === "api.crossref.org") {
        return new Response(JSON.stringify({ message: { items: [{
          DOI: "10.1234/climate", title: ["Climate adaptation paper"],
          author: [{ given: "A", family: "Scientist" }],
          published: { "date-parts": [[2025, 2, 1]] }
        }]}}), { status: 200, headers: { "content-type": "application/json" }});
      }
      return new Response(JSON.stringify({ results: [{
        id: "https://openalex.org/W12", doi: "https://doi.org/10.1234/climate",
        title: "Climate adaptation paper", publication_date: "2025-02-01",
        authorships: [], primary_location: { landing_page_url: "https://doi.org/10.1234/climate" }
      }] }), { status: 200, headers: { "content-type": "application/json" }});
    }
  });
  const result = await runPublicResearch("climate adaptation", registry);
  assert.equal(result.provider, "Wikipedia");
  assert.equal(result.results.length, 1);
  assert.equal(result.scholarlyResearch.results.length, 1);
  assert.equal(result.scholarlyResearch.results[0].provider, "Crossref");
  assert.deepEqual(result.providerAvailability, {
    encyclopedia: "available",
    globalPublicSources: "available",
    scholarly: "available",
    platforms: "unavailable"
  });
});

test("automatic chat research can use the audited encyclopedia tool without scholarly provider calls", async () => {
  let calls = 0;
  const registry = createDreamToolRegistry({
    audit: () => {},
    fetchImpl: async url => {
      calls += 1;
      const parsed = new URL(url);
      if (parsed.searchParams.get("list") === "search") {
        return responseWith([{ pageid: 7, title: "Manchester City F.C.", snippet: "An English football club." }]);
      }
      return responseWithPages({ "7": {
        pageid: 7, extract: "Manchester City F.C. is a football club in Manchester.",
        fullurl: "https://en.wikipedia.org/wiki/Manchester_City_F.C."
      }});
    }
  });
  const result = await runPublicKnowledgeSearch("Manchester City", registry);
  assert.equal(calls, 2);
  assert.equal(result.provider, "Wikipedia");
  assert.equal(result.results.length, 1);
  assert.equal(result.results[0].title, "Manchester City F.C.");
});
