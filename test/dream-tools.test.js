import assert from "node:assert/strict";
import test from "node:test";
import { createDreamToolRegistry } from "../lib/dream-tools.js";

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
