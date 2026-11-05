import test from "node:test";
import assert from "node:assert/strict";
import { searchMultiSourceResearch } from "../lib/research-multisource.js";

const now = new Date("2026-11-05T00:00:00Z");
function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json" } });
}
function crossref(items) { return { message: { items } }; }
function openalex(results) { return { results }; }

test("queries Crossref and OpenAlex and returns source-attributed scholarly metadata", async () => {
  const calls = [];
  const result = await searchMultiSourceResearch("climate adaptation", async url => {
    const parsed = new URL(url); calls.push(parsed);
    if (parsed.hostname === "api.crossref.org") return jsonResponse(crossref([{
      DOI: "10.1234/example", title: ["Climate adaptation evidence"],
      author: [{ given: "Ada", family: "Researcher" }],
      published: { "date-parts": [[2025, 4, 2]] }
    }]));
    return jsonResponse(openalex([{
      id: "https://openalex.org/W123", doi: "https://doi.org/10.1234/example",
      title: "Climate adaptation evidence", publication_date: "2025-04-02",
      authorships: [{ author: { display_name: "Ada Researcher" } }],
      primary_location: { landing_page_url: "https://doi.org/10.1234/example" }
    }, {
      id: "https://openalex.org/W456", doi: null, title: "A separate paper",
      publication_date: "2018-01-01", authorships: [], primary_location: {}
    }]));
  }, now);
  assert.deepEqual(calls.map(x => x.hostname).sort(), ["api.crossref.org", "api.openalex.org"]);
  assert.equal(result.providers.length, 2);
  assert.equal(result.providerCountSucceeded, 2);
  assert.equal(result.results.length, 2);
  assert.equal(result.results[0].doi, "10.1234/example");
  assert.equal(result.results[0].freshness, "recent");
  assert.equal(result.freshnessSummary.older, 1);
  assert.ok(result.results.every(r => r.provider && r.url && r.sourceType === "scholarly-metadata"));
  assert.match(result.limitations[0], /not proof/);
});

test("preserves partial results when one scholarly provider is unavailable", async () => {
  const result = await searchMultiSourceResearch("public health", async url => {
    if (new URL(url).hostname === "api.crossref.org") return jsonResponse(crossref([]), 503);
    return jsonResponse(openalex([{
      id: "https://openalex.org/W1", title: "Public health study",
      publication_date: "2020-03-01", authorships: [], primary_location: {}
    }]));
  }, now);
  assert.equal(result.results.length, 1);
  assert.equal(result.providers.find(p => p.provider === "Crossref").status, "unavailable");
  assert.equal(result.providers.find(p => p.provider === "OpenAlex").status, "ok");
});

test("fails transparently when both providers fail or return malformed payloads", async () => {
  await assert.rejects(searchMultiSourceResearch("science", async () => jsonResponse({}, 503)), /No scholarly research provider/);
  await assert.rejects(searchMultiSourceResearch("science", async () => jsonResponse({})), /No scholarly research provider/);
});

test("rejects invalid queries before making network requests", async () => {
  let called = false;
  const fetchImpl = async () => { called = true; };
  await assert.rejects(searchMultiSourceResearch("  ", fetchImpl), /1 to 200 characters/);
  await assert.rejects(searchMultiSourceResearch("x".repeat(201), fetchImpl), /1 to 200 characters/);
  assert.equal(called, false);
});

test("rejects unsafe provider URLs and classifies undated and future records conservatively", async () => {
  const result = await searchMultiSourceResearch("research integrity", async url => {
    if (new URL(url).hostname === "api.crossref.org") return jsonResponse(crossref([{
      DOI: "10.1111/unsafe", title: ["Undated work"], published: { "date-parts": [] }
    }]));
    return jsonResponse(openalex([{
      id: "http://openalex.org/W2", title: "Future work", publication_date: "2028-01-01",
      primary_location: { landing_page_url: "javascript:alert(1)" }
    }]));
  }, now);
  assert.equal(result.results.find(r => r.title === "Undated work").freshness, "unknown");
  const future = result.results.find(r => r.title === "Future work");
  assert.equal(future.freshness, "future-dated");
  assert.equal(future.url, null);
});
