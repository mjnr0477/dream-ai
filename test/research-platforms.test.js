import assert from "node:assert/strict";
import test from "node:test";
import { searchPubMed, searchPlatformSources } from "../lib/research-platforms.js";

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

  await assert.rejects(searchPubMed("climate research", async url => {
    const parsed = new URL(url);
    if (parsed.pathname.endsWith("/esearch.fcgi")) {
      return jsonResponse({ esearchresult: { idlist: ["98765"] } });
    }
    return new Response("<html>not XML</html>", { status: 200 });
  }), /invalid-pubmed-response/);
});

test("GitHub public repository search runs for coding intent without credentials", async () => {
  const requests = [];
  const result = await searchPlatformSources("Python code example for a CSV parser", async (url, options) => {
    const parsed = new URL(url);
    requests.push({ parsed, options });

    if (parsed.hostname === "api.github.com") {
      return jsonResponse({ items: [{
        full_name: "example/csv-parser",
        html_url: "https://github.com/example/csv-parser",
        description: "A small Python CSV parser.",
        language: "Python",
        stargazers_count: 123,
        updated_at: "2026-11-01T12:00:00Z",
        topics: ["python", "csv"],
        license: { spdx_id: "MIT" }
      }, {
        full_name: "attacker/unsafe",
        html_url: "https://attacker.example/unsafe",
        description: "This URL must be excluded."
      }] });
    }
    if (parsed.hostname === "public.api.bsky.app") return jsonResponse({ posts: [] });
    if (parsed.hostname === "api.stackexchange.com") return jsonResponse({ items: [] });
    if (parsed.hostname === "www.ebi.ac.uk") return jsonResponse({ resultList: { result: [] } });
    if (parsed.hostname === "news.google.com") return new Response("<rss><channel></channel></rss>", {
      status: 200, headers: { "content-type": "application/rss+xml" }
    });
    throw new Error("Unexpected external provider: " + parsed.hostname);
  }, {});

  const github = result.results.find(item => item.provider === "GitHub Public Repositories");
  assert.ok(github);
  assert.equal(github.title, "example/csv-parser");
  assert.equal(github.url, "https://github.com/example/csv-parser");
  assert.equal(github.language, "Python");
  assert.equal(github.stars, 123);
  assert.equal(github.license, "MIT");
  assert.doesNotMatch(JSON.stringify(result.results), /attacker\.example/);

  const request = requests.find(item => item.parsed.hostname === "api.github.com");
  assert.equal(request.parsed.pathname, "/search/repositories");
  assert.equal(request.parsed.searchParams.get("per_page"), "5");
  assert.equal(request.parsed.searchParams.get("sort"), "stars");
  assert.equal(request.options.method, "GET");
  assert.equal(request.options.headers["x-github-api-version"], "2026-03-10");
  assert.equal(request.options.headers.authorization, undefined);
  assert.equal(result.providers.find(item => item.provider === "GitHub Public Repositories").status, "ok");
});

test("GitHub search is skipped for non-coding questions", async () => {
  const hosts = [];
  const result = await searchPlatformSources("What is the history of the Great Wall of China?", async (url) => {
    const parsed = new URL(url);
    hosts.push(parsed.hostname);
    if (parsed.hostname === "public.api.bsky.app") return jsonResponse({ posts: [] });
    if (parsed.hostname === "api.stackexchange.com") return jsonResponse({ items: [] });
    if (parsed.hostname === "www.ebi.ac.uk") return jsonResponse({ resultList: { result: [] } });
    if (parsed.hostname === "news.google.com") return new Response("<rss><channel></channel></rss>", {
      status: 200, headers: { "content-type": "application/rss+xml" }
    });
    throw new Error("Unexpected external provider: " + parsed.hostname);
  }, {});

  assert.equal(hosts.includes("api.github.com"), false);
  assert.equal(result.providers.find(item => item.provider === "GitHub Public Repositories").status, "not-applicable");
});

test("GitHub anonymous quota errors are reported and never retried", async () => {
  let githubCalls = 0;
  const result = await searchPlatformSources("JavaScript library for CSV parsing", async url => {
    const parsed = new URL(url);
    if (parsed.hostname === "api.github.com") {
      githubCalls += 1;
      return jsonResponse({ message: "API rate limit exceeded" }, 403);
    }
    if (parsed.hostname === "public.api.bsky.app") return jsonResponse({ posts: [] });
    if (parsed.hostname === "api.stackexchange.com") return jsonResponse({ items: [] });
    if (parsed.hostname === "www.ebi.ac.uk") return jsonResponse({ resultList: { result: [] } });
    if (parsed.hostname === "news.google.com") return new Response("<rss><channel></channel></rss>", {
      status: 200, headers: { "content-type": "application/rss+xml" }
    });
    throw new Error("Unexpected external provider: " + parsed.hostname);
  }, {});

  assert.equal(githubCalls, 1);
  assert.equal(result.providers.find(item => item.provider === "GitHub Public Repositories").status, "unavailable");
  assert.equal(result.results.some(item => item.provider === "GitHub Public Repositories"), false);
});

test("automatic public research uses the freshness-filtered regional Google News feed", async () => {
  const calls = [];
  const publishedAt = new Date(Date.now() - 20 * 60 * 1000);
  const xml = [
    "<rss><channel><item>",
    "<title><![CDATA[Manchester City announce a squad update]]></title>",
    "<link>https://news.google.com/rss/articles/CBMiRecentStory</link>",
    "<pubDate>" + publishedAt.toUTCString() + "</pubDate>",
    '<source url="https://www.bbc.com/">BBC Sport</source>',
    "<description><![CDATA[The report includes a dated squad update.]]></description>",
    "</item></channel></rss>"
  ].join("");

  const research = await searchPlatformSources(
    "latest Manchester City news today",
    async (url, options = {}) => {
      const parsed = new URL(url);
      calls.push({ parsed, options });
      if (parsed.hostname === "news.google.com") {
        return new Response(xml, { status: 200, headers: { "content-type": "application/rss+xml" } });
      }
      if (parsed.hostname === "public.api.bsky.app") return jsonResponse({ posts: [] });
      if (parsed.hostname === "api.stackexchange.com") return jsonResponse({ items: [] });
      if (parsed.hostname === "www.ebi.ac.uk") return jsonResponse({ resultList: { result: [] } });
      if (parsed.hostname === "eutils.ncbi.nlm.nih.gov") return jsonResponse({ esearchresult: { idlist: [] } });
      throw new Error("Unexpected external provider: " + parsed.hostname);
    },
    {}
  );

  const news = research.results.find(item => item.provider === "Google News RSS");
  assert.ok(news);
  assert.equal(news.title, "Manchester City announce a squad update");
  assert.equal(news.publisher, "BBC Sport");
  assert.equal(news.publishedAt, publishedAt.toISOString());
  assert.match(news.snippet, /dated squad update/);
  assert.equal(news.sourceType, "recent-news-headline");
  assert.equal(new URL(news.url).hostname, "news.google.com");

  const feedCall = calls.find(call => call.parsed.hostname === "news.google.com");
  assert.ok(feedCall);
  assert.equal(feedCall.parsed.searchParams.get("q"), "latest Manchester City news today when:1d");
  assert.equal(feedCall.options.redirect, "error");
  assert.equal(research.providers.find(item => item.provider === "Google News RSS").status, "ok");
});
