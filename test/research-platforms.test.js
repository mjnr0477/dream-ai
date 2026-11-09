import test from "node:test";
import assert from "node:assert/strict";
import { searchPlatformSources } from "../lib/research-platforms.js";

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json" }
  });
}

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
