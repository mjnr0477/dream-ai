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
          { web: { title: "Official source", uri: "https://vertexaisearch.cloud.google.com/grounding-api-redirect/example" } },
          { web: { title: "Independent publisher", uri: "https://www.bbc.com/news/example" } }
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
  const googleSources = result.results.filter(item => item.provider === "Google Search");
  assert.equal(googleSources.length, 2);
  assert.equal(googleSources[0].title, "Official source");
  assert.equal(googleSources[0].snippet, "");
  assert.equal(googleSources[1].url, "https://www.bbc.com/news/example");
  assert.equal(googleSources[1].snippet, "");
  assert.ok(googleSources.every(source => source.snippet === ""));
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

test("World Bank Indicators returns country statistics without an API key and preserves observation years", async () => {
  const { fetchImpl, requests } = captureFetch(request => {
    if (request.url.hostname === "api.worldbank.org") {
      if (request.url.pathname === "/v2/country") {
        return jsonResponse([{
          page: 1, pages: 1, per_page: "400", total: 3
        }, [
          { id: "KEN", name: "Kenya", region: { value: "Sub-Saharan Africa" } },
          { id: "UGA", name: "Uganda", region: { value: "Sub-Saharan Africa" } },
          { id: "WLD", name: "World", region: { value: "Aggregates" } }
        ]]);
      }
      if (request.url.pathname === "/v2/country/KEN/indicator/SP.POP.TOTL") {
        return jsonResponse([{
          page: 1, pages: 1, per_page: "25", total: 1, lastupdated: "2026-10-01"
        }, [{
          country: { id: "KEN", value: "Kenya" },
          countryiso3code: "KEN",
          indicator: { id: "SP.POP.TOTL", value: "Population, total" },
          date: "2024",
          value: 56432900,
          unit: "",
          obs_status: "",
          decimal: 0
        }]]);
      }
    }
    if (request.url.hostname === "public.api.bsky.app") return jsonResponse({ posts: [] });
    if (request.url.hostname === "news.google.com") return textResponse("<rss></rss>");
    if (request.url.hostname === "api.stackexchange.com") return jsonResponse({ items: [], quota_remaining: 9000 });
    if (request.url.hostname === "www.ebi.ac.uk") return jsonResponse({ resultList: { result: [] } });
    throw new Error("unexpected provider request");
  });

  const result = await searchPlatformSources("What is the population of Kenya?", fetchImpl, {});
  const countryCall = requests.find(item =>
    item.url.hostname === "api.worldbank.org" && item.url.pathname === "/v2/country"
  );
  const dataCall = requests.find(item =>
    item.url.hostname === "api.worldbank.org" && item.url.pathname.includes("/indicator/")
  );
  assert.ok(countryCall);
  assert.ok(dataCall);
  assert.equal(countryCall.url.searchParams.get("format"), "json");
  assert.equal(dataCall.url.pathname, "/v2/country/KEN/indicator/SP.POP.TOTL");
  assert.equal(dataCall.url.searchParams.get("mrnev"), "5");
  assert.equal(dataCall.options.headers.authorization, undefined);

  const provider = result.providers.find(item => item.provider === "World Bank Indicators");
  assert.equal(provider.status, "ok");
  const observation = result.results.find(item => item.provider === "World Bank Indicators");
  assert.equal(observation.country, "Kenya");
  assert.equal(observation.indicatorCode, "SP.POP.TOTL");
  assert.equal(observation.observationYear, "2024");
  assert.match(observation.snippet, /Observation year: 2024/);
  assert.match(observation.snippet, /not a live measurement/);
  assert.equal(observation.url, "https://api.worldbank.org/v2/country/KEN/indicator/SP.POP.TOTL?format=json&mrnev=5");
});

test("World Bank statistics honour an explicitly requested historical year", async () => {
  const { fetchImpl, requests } = captureFetch(request => {
    if (request.url.hostname === "api.worldbank.org") {
      if (request.url.pathname === "/v2/country") {
        return jsonResponse([{}, [
          { id: "KEN", name: "Kenya", region: { value: "Sub-Saharan Africa" } }
        ]]);
      }
      if (request.url.pathname === "/v2/country/KEN/indicator/NY.GDP.PCAP.CD") {
        return jsonResponse([{}, [{
          country: { id: "KEN", value: "Kenya" },
          countryiso3code: "KEN",
          indicator: { id: "NY.GDP.PCAP.CD", value: "GDP per capita (current US$)" },
          date: "2020",
          value: 1966.2,
          unit: ""
        }]]);
      }
    }
    if (request.url.hostname === "public.api.bsky.app") return jsonResponse({ posts: [] });
    if (request.url.hostname === "news.google.com") return textResponse("<rss></rss>");
    if (request.url.hostname === "api.stackexchange.com") return jsonResponse({ items: [], quota_remaining: 9000 });
    if (request.url.hostname === "www.ebi.ac.uk") return jsonResponse({ resultList: { result: [] } });
    throw new Error("unexpected provider request");
  });

  const result = await searchPlatformSources("What was GDP per capita in Kenya in 2020?", fetchImpl, {});
  const call = requests.find(item =>
    item.url.hostname === "api.worldbank.org" && item.url.pathname.includes("/indicator/")
  );
  assert.ok(call);
  assert.equal(call.url.pathname, "/v2/country/KEN/indicator/NY.GDP.PCAP.CD", JSON.stringify(requests.map(item => item.url.toString())));
  assert.equal(call.url.searchParams.get("date"), "2020");
  assert.equal(call.url.searchParams.has("mrnev"), false);
  const provider = result.providers.find(item => item.provider === "World Bank Indicators");
  assert.equal(provider.status, "ok", JSON.stringify(requests.map(item => item.url.toString())));
  const value = result.results.find(item => item.provider === "World Bank Indicators");
  assert.equal(value.indicatorCode, "NY.GDP.PCAP.CD");
  assert.equal(value.observationYear, "2020");
  assert.equal(new URL(value.url).searchParams.get("date"), "2020");
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
  assert.equal(result.providers.find(p => p.provider === "World Bank Indicators").status, "not-applicable");
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


test("Stack Exchange public search works without credentials", async () => {
  const { fetchImpl, requests } = captureFetch(request => {
    if (request.url.hostname === "api.stackexchange.com") {
      return jsonResponse({
        quota_remaining: 9975,
        items: [{
          title: "How do I safely parse JSON in JavaScript?",
          link: "https://stackoverflow.com/questions/123456/safe-json-parse",
          body: "<p>Use JSON.parse inside a try/catch and validate the resulting shape.</p>",
          tags: ["javascript", "json"],
          score: 42,
          answer_count: 3,
          is_answered: true,
          creation_date: 1794144000
        }]
      });
    }
    if (request.url.hostname === "public.api.bsky.app") return jsonResponse({ posts: [] });
    if (request.url.hostname === "news.google.com") return textResponse("<rss></rss>");
    throw new Error("unexpected provider call");
  });
  const result = await searchPlatformSources("parse JSON safely", fetchImpl, {});
  const call = requests.find(item => item.url.hostname === "api.stackexchange.com");
  assert.ok(call);
  assert.equal(call.url.pathname, "/2.3/search/advanced");
  assert.equal(call.url.searchParams.get("site"), "stackoverflow");
  assert.equal(call.url.searchParams.get("q"), "parse JSON safely");
  assert.equal(call.options.headers.authorization, undefined);
  const question = result.results.find(item => item.provider === "Stack Exchange");
  assert.equal(question.title, "How do I safely parse JSON in JavaScript?");
  assert.equal(question.url, "https://stackoverflow.com/questions/123456/safe-json-parse");
  assert.match(question.snippet, /JSON.parse/);
  assert.equal(question.score, 42);
  assert.deepEqual(question.tags, ["javascript", "json"]);
  assert.equal(result.providers.find(item => item.provider === "Stack Exchange").status, "ok");
});
