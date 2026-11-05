import test from "node:test";
import assert from "node:assert/strict";
import { searchPublicKnowledge } from "../lib/research.js";

test("searches the fixed public knowledge endpoint and returns safe source links", async () => {
  let requestedUrl;
  const result = await searchPublicKnowledge("Kenya education", async url => {
    requestedUrl = new URL(url);
    return {
      ok: true,
      json: async () => ({
        query: {
          search: [{
            title: "Education in Kenya",
            snippet: "Education <span class=\"searchmatch\">in Kenya</span> &amp; schools"
          }]
        }
      })
    };
  });

  assert.equal(requestedUrl.origin, "https://en.wikipedia.org");
  assert.equal(requestedUrl.searchParams.get("srsearch"), "Kenya education");
  assert.equal(result.provider, "Wikipedia");
  assert.equal(result.results.length, 1);
  assert.equal(result.results[0].title, "Education in Kenya");
  assert.equal(result.results[0].snippet, "Education in Kenya & schools");
  assert.equal(
    result.results[0].url,
    "https://en.wikipedia.org/wiki/Education_in_Kenya"
  );
});

test("rejects empty and oversized research queries before making a request", async () => {
  let called = false;
  const fetchImpl = async () => { called = true; };
  await assert.rejects(searchPublicKnowledge("  ", fetchImpl), /1 to 200 characters/);
  await assert.rejects(searchPublicKnowledge("x".repeat(201), fetchImpl), /1 to 200 characters/);
  assert.equal(called, false);
});

test("rejects malformed upstream responses", async () => {
  await assert.rejects(
    searchPublicKnowledge("science", async () => ({
      ok: true,
      json: async () => ({})
    })),
    /invalid response/
  );
});

test("reports upstream failures without returning fabricated results", async () => {
  await assert.rejects(
    searchPublicKnowledge("science", async () => ({ ok: false, json: async () => ({}) })),
    /public research source returned an error/
  );
});
