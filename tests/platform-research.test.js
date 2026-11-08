import test from "node:test";
import assert from "node:assert/strict";
import { searchPlatformSources } from "../lib/research-platforms.js";

function jsonResponse(payload) {
  return { ok: true, json: async () => payload };
}
function textResponse(value) {
  return { ok: true, text: async () => value };
}

test("Google Search grounding returns cited public web sources without replacing DREAM", async () => {
  let request;
  const fetchImpl = async (url, options) => {
    request = { url: new URL(url), options };
    return jsonResponse({
      candidates: [{
        content: { parts: [{ text: "A current public fact with citations." }] },
        groundingMetadata: {
          webSearchQueries: ["current public fact"],
          groundingChunks: [
            { web: { title: "Official source", uri: "https://vertexaisearch.cloud.google.com/grounding-api-redirect/example" } }
          ]
        }
      }]
    });
  };
  const result = await searchPlatformSources("current public fact", fetchImpl, {
    DREAM_GOOGLE_GEMINI_API_KEY: "server-only-test-key"
  });
  assert.equal(request.url.hostname, "generativelanguage.googleapis.com");
  assert.equal(request.url.searchParams.get("key"), "server-only-test-key");
  assert.equal(request.options.method, "POST");
  assert.deepEqual(JSON.parse(request.options.body).tools, [{ google_search: {} }]);
  assert.equal(result.googleGroundedAnswer, "A current public fact with citations.");
  assert.equal(result.results[0].provider, "Google Search");
});

test("YouTube search returns public video metadata and uses the server API key", async () => {
  let request;
  const fetchImpl = async (url, options) => {
    request = { url: new URL(url), options };
    return jsonResponse({ items: [{
      id: { videoId: "abc123" },
      snippet: {
        title: "Public video",
        description: "Public description",
        channelTitle: "Official channel",
        publishedAt: "2026-11-08T10:00:00Z"
      }
    }] });
  };
  const result = await searchPlatformSources("public video", fetchImpl, {
    DREAM_YOUTUBE_API_KEY: "youtube-test-key"
  });
  assert.equal(request.url.hostname, "www.googleapis.com");
  assert.equal(request.url.pathname, "/youtube/v3/search");
  assert.equal(request.url.searchParams.get("key"), "youtube-test-key");
  assert.equal(result.results[0].url, "https://www.youtube.com/watch?v=abc123");
  assert.equal(result.results[0].channel, "Official channel");
});

test("Bluesky public search works without credentials and never exposes private account data", async () => {
  let request;
  const fetchImpl = async (url, options) => {
    request = { url: new URL(url), options };
    return jsonResponse({ posts: [{
      uri: "at://did:plc:abc/app.bsky.feed.post/post123",
      indexedAt: "2026-11-08T09:00:00Z",
      author: { handle: "public.example", displayName: "Public Author" },
      record: { text: "A public post about the topic." }
    }] });
  };
  const result = await searchPlatformSources("topic", fetchImpl, {});
  assert.equal(request.url.hostname, "public.api.bsky.app");
  assert.equal(result.results[0].provider, "Bluesky");
  assert.equal(result.results[0].url, "https://bsky.app/profile/public.example/post/post123");
  assert.equal(result.providers.find(p => p.provider === "Bluesky").status, "ok");
});

test("X public search activates only with a server-side bearer token", async () => {
  let request;
  const fetchImpl = async (url, options) => {
    request = { url: new URL(url), options };
    return jsonResponse({
      data: [{ id: "12345", text: "A public X post", author_id: "u1", created_at: "2026-11-08T10:00:00Z" }],
      includes: { users: [{ id: "u1", username: "publicuser" }] }
    });
  };
  const result = await searchPlatformSources("public post", fetchImpl, {
    DREAM_X_BEARER_TOKEN: "x-test-token"
  });
  assert.equal(request.url.hostname, "api.x.com");
  assert.equal(request.options.headers.authorization, "Bearer x-test-token");
  assert.equal(result.results[0].url, "https://x.com/publicuser/status/12345");
});

test("Google News RSS returns headline metadata with publisher links", async () => {
  let request;
  const xml = '<rss><channel><item><title>Kenya technology update</title><link>https://publisher.example/news/1</link><pubDate>Sun, 08 Nov 2026 10:00:00 GMT</pubDate></item></channel></rss>';
  const fetchImpl = async (url, options) => {
    request = { url: new URL(url), options };
    return textResponse(xml);
  };
  const result = await searchPlatformSources("Kenya technology", fetchImpl, {});
  assert.equal(request.url.hostname, "news.google.com");
  assert.equal(result.results[0].provider, "Google News RSS");
  assert.equal(result.results[0].title, "Kenya technology update");
  assert.equal(result.results[0].url, "https://publisher.example/news/1");
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
