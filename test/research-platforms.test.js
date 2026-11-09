import assert from "node:assert/strict";
import test from "node:test";
import { searchPubMed } from "../lib/research-platforms.js";

function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json" }
  });
}

const sampleXml = `<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">12345678</PMID>
      <Article>
        <ArticleTitle>Learning &amp; educational outcomes in Kenya</ArticleTitle>
        <AuthorList>
          <Author><ForeName>Jane</ForeName><LastName>Researcher</LastName></Author>
          <Author><CollectiveName>Kenya Learning Group</CollectiveName></Author>
        </AuthorList>
        <Journal><Title>Journal of Education Research</Title></Journal>
        <PubDate><Year>2025</Year></PubDate>
        <Abstract>
          <AbstractText Label="BACKGROUND">Learning access remains uneven.</AbstractText>
          <AbstractText Label="RESULTS">Community programmes improved access &amp; attendance.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>`;

test("searches PubMed without a user API key and returns source-linked abstracts", async () => {
  const calls = [];
  const results = await searchPubMed("education access Kenya", async (url, options) => {
    const parsed = new URL(url);
    calls.push({ parsed, options });
    if (parsed.pathname.endsWith("/esearch.fcgi")) {
      return jsonResponse({ esearchresult: { idlist: ["12345678"] } });
    }
    if (parsed.pathname.endsWith("/efetch.fcgi")) {
      return new Response(sampleXml, { status: 200, headers: { "content-type": "application/xml" } });
    }
    throw new Error("Unexpected PubMed endpoint");
  });

  assert.equal(calls.length, 2);
  assert.equal(calls[0].parsed.origin, "https://eutils.ncbi.nlm.nih.gov");
  assert.equal(calls[0].parsed.searchParams.get("db"), "pubmed");
  assert.equal(calls[0].parsed.searchParams.get("term"), "education access Kenya");
  assert.equal(calls[0].parsed.searchParams.get("retmode"), "json");
  assert.equal(calls[0].parsed.searchParams.has("api_key"), false);
  assert.equal(calls[1].parsed.searchParams.get("id"), "12345678");
  assert.equal(calls[1].parsed.searchParams.get("retmode"), "xml");
  assert.equal(calls.every(call => call.options.redirect === "error"), true);
  assert.equal(results.length, 1);
  assert.equal(results[0].provider, "PubMed");
  assert.equal(results[0].title, "Learning & educational outcomes in Kenya");
  assert.equal(results[0].snippet, "BACKGROUND: Learning access remains uneven. RESULTS: Community programmes improved access & attendance.");
  assert.deepEqual(results[0].authors, ["Jane Researcher", "Kenya Learning Group"]);
  assert.equal(results[0].journal, "Journal of Education Research");
  assert.equal(results[0].publishedAt, "2025");
  assert.equal(results[0].url, "https://pubmed.ncbi.nlm.nih.gov/12345678/");
  assert.equal(results[0].sourceType, "pubmed-abstract");
  assert.equal(results[0].abstractAvailable, true);
});

test("returns an empty result without fetching article records when PubMed finds no IDs", async () => {
  let calls = 0;
  const results = await searchPubMed("query without matches", async () => {
    calls += 1;
    return jsonResponse({ esearchresult: { idlist: [] } });
  });
  assert.deepEqual(results, []);
  assert.equal(calls, 1);
});

test("rejects invalid PubMed queries before making requests", async () => {
  let called = false;
  const fetchImpl = async () => {
    called = true;
    return jsonResponse({ esearchresult: { idlist: [] } });
  };
  await assert.rejects(searchPubMed(" "), /1 to 200 characters/);
  await assert.rejects(searchPubMed("x".repeat(201)), /1 to 200 characters/);
  assert.equal(called, false);
});

test("rejects malformed PubMed identifiers and XML without fabricating citations", async () => {
  await assert.rejects(searchPubMed("climate research", async () =>
    jsonResponse({ esearchresult: { idlist: ["javascript:alert(1)"] } })
  ), /invalid-pubmed-search-response/);

  const results = await searchPubMed("climate research", async url => {
    const parsed = new URL(url);
    if (parsed.pathname.endsWith("/esearch.fcgi")) {
      return jsonResponse({ esearchresult: { idlist: ["98765"] } });
    }
    return new Response("<html>not XML</html>", { status: 200 });
  });
  assert.fail("Expected malformed XML to reject, got " + JSON.stringify(results));
});
