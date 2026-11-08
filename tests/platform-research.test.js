import test from "node:test";
import assert from "node:assert/strict";
import { searchPlatformSources } from "../lib/research-platforms.js";
import { getDreamCapabilities } from "../lib/capabilities.js";

function jsonResponse(payload) {
  return { ok: true, json: async () => payload };
}
function textResponse(value) {
  return { ok: true, text: async () => value };
}
function captureFetch(handler) {
  const requests = [];
  const fetchImpl = async (url, options) => {
    const request = { url: new URL(url), options };
    requests.push(request);
    return handler(request);
  };
  return { fetchImpl, requests };
}

test("Google Search grounding returns cited public web sources without replacing DREAM", async () => {
  const { fetchImpl, requests } = captureFetch(() => jsonResponse({
    candidates: [{
      content: { parts: [{ text: "A current public fact with citations." }] },
      groundingMetadata: {
        webSearchQueries: ["current public fact"],
        groundingChunks: [
          { web: { title: "Official source", uri: "https://vertexaisearch.cloud.google.com/grounding-api-redirect/example" } }
        ]
      }
    }]
  }));
  const result = await searchPlatformSources("current public fact", fetchImpl, {
    DREAM_GOOGLE_GEMINI_API_KEY: "server-only-test-key"
  });
  const call = requests.find(item => item.url.hostname === "generativelanguage.googleapis.com");
  assert.ok(call);
  assert.equal(call.url.searchParams.has("key"), false);
  assert.equal(call.options.headers["x-goog-api-key"], "server-only-test-key");
  assert.equal(call.options.method, "POST");
  assert.deepEqual(JSON.parse(call.options.body).tools, [{ google_search: {} }]);
  assert.equal(result.googleGroundedAnswer, "A current public fact with citations.");
  assert.equal(result.results.find(item => item.provider === "Google Search").title, "Official source");
});

test("YouTube search returns public video metadata and uses the server API key", async () => {
  const { fetchImpl, requests } = captureFetch(() => jsonResponse({ items: [{
    id: { videoId: "abc123" },
    snippet: {
      title: "Public video",
      description: "Public description",
      channelTitle: "Official channel",
      publishedAt: "2026-11-08T10:00:00Z"
    }
  }] }));
  const result = await searchPlatformSources("public video", fetchImpl, {
    DREAM_YOUTUBE_API_KEY: "youtube-test-key"
  });
  const call = requests.find(item => item.url.hostname === "www.googleapis.com");
  assert.ok(call);
  assert.equal(call.url.pathname, "/youtube/v3/search");
  assert.equal(call.url.searchParams.has("key"), false);
  assert.equal(call.options.headers["x-goog-api-key"], "youtube-test-key");
  assert.equal(result.results.find(item => item.provider === "YouTube Data API").url, "https://www.youtube.com/watch?v=abc123");
  assert.equal(result.results.find(item => item.provider === "YouTube Data API").channel, "Official channel");
});

test("Bluesky public search works without credentials", async () => {
  const { fetchImpl, requests } = captureFetch(() => jsonResponse({ posts: [{
    uri: "at://did:plc:abc/app.bsky.feed.post/post123",
    indexedAt: "2026-11-08T09:00:00Z",
    author: { handle: "public.example", displayName: "Public Author" },
    record: { text: "A public post about the topic." }
  }] }));
  const result = await searchPlatformSources("topic", fetchImpl, {});
  assert.ok(requests.some(item => item.url.hostname === "public.api.bsky.app"));
  const post = result.results.find(item => item.provider === "Bluesky");
  assert.equal(post.url, "https://bsky.app/profile/public.example/post/post123");
  assert.equal(post.snippet, "A public post about the topic.");
});

test("X public search activates only with a server-side bearer token", async () => {
  const { fetchImpl, requests } = captureFetch(() => jsonResponse({
    data: [{ id: "12345", text: "A public X post", author_id: "u1", created_at: "2026-11-08T10:00:00Z" }],
    includes: { users: [{ id: "u1", username: "publicuser" }] }
  }));
  const result = await searchPlatformSources("public post", fetchImpl, {
    DREAM_X_BEARER_TOKEN: "x-test-token"
  });
  const call = requests.find(item => item.url.hostname === "api.x.com");
  assert.ok(call);
  assert.equal(call.options.headers.authorization, "Bearer x-test-token");
  assert.equal(result.results.find(item => item.provider === "X API").url, "https://x.com/publicuser/status/12345");
});

test("Europe PMC returns biomedical abstracts with publication provenance", async () => {
  const { fetchImpl, requests } = captureFetch(() => jsonResponse({
    resultList: {
      result: [{
        id: "123456",
        pmid: "123456",
        pmcid: "PMC12345",
        title: "A biomedical research paper",
        authorString: "A. Researcher",
        abstractText: "A structured abstract for public research discovery.",
        firstPublicationDate: "2026-05-01",
        journalInfo: { journal: { title: "Open Research Journal" } },
        isOpenAccess: "Y"
      }]
    }
  }));
  const result = await searchPlatformSources("biomedical research", fetchImpl, {});
  assert.ok(requests.some(item => item.url.hostname === "www.ebi.ac.uk"));
  const paper = result.results.find(item => item.provider === "Europe PMC");
  assert.equal(paper.title, "A biomedical research paper");
  assert.equal(paper.snippet, "A structured abstract for public research discovery.");
  assert.equal(paper.url, "https://europepmc.org/articles/PMC12345");
  assert.equal(paper.openAccess, true);
});

test("Google News RSS returns headline metadata with publisher links", async () => {
  const xml = '<rss><channel><item><title>Kenya technology update</title><link>https://publisher.example/news/1</link><pubDate>Sun, 08 Nov 2026 10:00:00 GMT</pubDate></item></channel></rss>';
  const { fetchImpl, requests } = captureFetch(() => textResponse(xml));
  const result = await searchPlatformSources("Kenya technology", fetchImpl, {});
  assert.ok(requests.some(item => item.url.hostname === "news.google.com"));
  const news = result.results.find(item => item.provider === "Google News RSS");
  assert.equal(news.title, "Kenya technology update");
  assert.equal(news.url, "https://publisher.example/news/1");
});

test("unconfigured providers are reported honestly and invalid queries are rejected", async () => {
  const fetchImpl = async url => {
    if (new URL(url).hostname === "public.api.bsky.app") return jsonResponse({ posts: [] });
    if (new URL(url).hostname === "news.google.com") return textResponse("<rss></rss>");
    throw new Error("unexpected provider call");
  };
  const result = await searchPlatformSources("valid query", fetchImpl, {});
  assert.equal(result.providers.find(p => p.provider === "Google Search").status, "not-configured");
  assert.equal(result.providers.find(p => p.provider === "YouTube Data API").status, "not-configured");
  assert.equal(result.providers.find(p => p.provider === "X API").status, "not-configured");
  await assert.rejects(() => searchPlatformSources(""), /1 to 200/);
});

test("capabilities expose Google, YouTube and X integrations only when credentials exist", () => {
  const withoutKeys = getDreamCapabilities({}, {});
  assert.equal(withoutKeys.tools.find(item => item.id === "google_search_grounding").status, "not-configured");
  assert.equal(withoutKeys.tools.find(item => item.id === "youtube_public_search").status, "not-configured");
  assert.equal(withoutKeys.tools.find(item => item.id === "x_public_search").status, "not-configured");
  const configured = getDreamCapabilities({}, {
    DREAM_GOOGLE_GEMINI_API_KEY: "google",
    DREAM_YOUTUBE_API_KEY: "youtube",
    DREAM_X_BEARER_TOKEN: "x"
  });
  assert.equal(configured.tools.find(item => item.id === "google_search_grounding").status, "available");
  assert.equal(configured.tools.find(item => item.id === "youtube_public_search").status, "available");
  assert.equal(configured.tools.find(item => item.id === "x_public_search").status, "available");
  assert.ok(configured.application.researchScope.includes("google-search-grounding"));
});
