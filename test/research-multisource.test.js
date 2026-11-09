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
  }, now, { arxivMinIntervalMs: 0 });
  assert.deepEqual(calls.map(x => x.hostname).sort(), ["api.crossref.org", "api.openalex.org", "api.semanticscholar.org", "export.arxiv.org"]);
  assert.equal(result.providers.length, 4);
  assert.equal(result.providers.find(p => p.provider === "arXiv").status, "unavailable");
  assert.equal(result.providers.find(p => p.provider === "Semantic Scholar").status, "unavailable");
  assert.equal(result.providerCountSucceeded, 2);
  assert.equal(result.results.length, 2);
  assert.equal(result.results[0].doi, "10.1234/example");
  assert.equal(result.results[0].freshness, "recent");
  assert.equal(result.freshnessSummary.older, 1);
  assert.ok(result.results.every(r => r.provider && r.url && r.sourceType === "scholarly-metadata"));
  assert.match(result.limitations[0], /not proof/);
});

test("keeps distinct non-Latin scholarly titles when DOI metadata is absent", async () => {
  const result = await searchMultiSourceResearch("African language research", async url => {
    if (new URL(url).hostname === "api.crossref.org") return jsonResponse(crossref([{
      title: ["የአየር ሁኔታ ጥናት"], author: [], published: { "date-parts": [[2024]] }
    }]));
    return jsonResponse(openalex([{
      id: "https://openalex.org/W999", doi: null, title: "الصحة العامة والبحث",
      publication_date: "2023-05-01", authorships: [], primary_location: {}
    }]));
  }, now, { arxivMinIntervalMs: 0 });
  assert.equal(result.results.length, 2);
  assert.deepEqual(result.results.map(item => item.title).sort(), ["الصحة العامة والبحث", "የአየር ሁኔታ ጥናት"].sort());
});

test("preserves partial results when one scholarly provider is unavailable", async () => {
  const result = await searchMultiSourceResearch("public health", async url => {
    if (new URL(url).hostname === "api.crossref.org") return jsonResponse(crossref([]), 503);
    return jsonResponse(openalex([{
      id: "https://openalex.org/W1", title: "Public health study",
      publication_date: "2020-03-01", authorships: [], primary_location: {}
    }]));
  }, now, { arxivMinIntervalMs: 0 });
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
  }, now, { arxivMinIntervalMs: 0 });
  assert.equal(result.results.find(r => r.title === "Undated work").freshness, "unknown");
  const future = result.results.find(r => r.title === "Future work");
  assert.equal(future.freshness, "future-dated");
  assert.equal(future.url, null);
});


test("searches arXiv without credentials and returns an attributed preprint abstract", async () => {
  const requests = [];
  const xml = `<?xml version="1.0" encoding="UTF-8"?>
  <feed xmlns="http://www.w3.org/2005/Atom">
    <entry>
      <id>http://arxiv.org/abs/2601.01234v1</id>
      <title>  Reliable AI for public research  </title>
      <published>2026-01-12T10:00:00Z</published>
      <summary>We study reliable public research. &amp; evidence verification.</summary>
      <author><name>Jane Researcher</name></author>
      <author><name>John Scientist</name></author>
    </entry>
  </feed>`;
  const result = await searchMultiSourceResearch("reliable AI", async url => {
    const parsed = new URL(url);
    requests.push(parsed);
    if (parsed.hostname === "export.arxiv.org") {
      return new Response(xml, { status: 200, headers: { "content-type": "application/atom+xml" } });
    }
    if (parsed.hostname === "api.crossref.org") return jsonResponse(crossref([]));
    return jsonResponse(openalex([]));
  }, now, { arxivMinIntervalMs: 0 });
  const record = result.results.find(item => item.provider === "arXiv");
  assert.ok(requests.some(url => url.hostname === "export.arxiv.org"));
  assert.equal(record.title, "Reliable AI for public research");
  assert.equal(record.providerRecordId, "2601.01234v1");
  assert.deepEqual(record.authors, ["Jane Researcher", "John Scientist"]);
  assert.equal(record.abstract, "We study reliable public research. & evidence verification.");
  assert.equal(record.url, "https://arxiv.org/abs/2601.01234v1");
  assert.equal(record.freshness, "recent");
  assert.equal(result.providers.find(p => p.provider === "arXiv").status, "ok");
});


test("searches Semantic Scholar's anonymous paper endpoint and returns abstract metadata", async () => {
  const requests = [];
  const result = await searchMultiSourceResearch("reliable public AI", async url => {
    const parsed = new URL(url);
    requests.push(parsed);
    if (parsed.hostname === "api.semanticscholar.org") {
      return jsonResponse({ data: [{
        paperId: "0123456789abcdef0123456789abcdef01234567",
        title: "Reliable public AI systems",
        abstract: "A study of reliability, transparency and evidence.",
        year: 2026,
        authors: [{ name: "A. Researcher" }],
        url: "https://www.semanticscholar.org/paper/Reliable-public-AI-systems/0123456789abcdef0123456789abcdef01234567",
        externalIds: { DOI: "10.1234/reliable-ai" },
        isOpenAccess: true
      }] });
    }
    if (parsed.hostname === "export.arxiv.org") {
      return new Response("<feed></feed>", { status: 200, headers: { "content-type": "application/atom+xml" } });
    }
    if (parsed.hostname === "api.crossref.org") return jsonResponse(crossref([]));
    return jsonResponse(openalex([]));
  }, now, { arxivMinIntervalMs: 0 });
  const record = result.results.find(item => item.provider === "Semantic Scholar");
  assert.ok(requests.some(url => url.hostname === "api.semanticscholar.org"));
  assert.equal(record.title, "Reliable public AI systems");
  assert.equal(record.abstract, "A study of reliability, transparency and evidence.");
  assert.equal(record.publicationDate, "2026");
  assert.equal(record.doi, "10.1234/reliable-ai");
  assert.equal(record.openAccess, true);
  assert.equal(result.providers.find(p => p.provider === "Semantic Scholar").status, "ok");
});
