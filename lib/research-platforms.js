const MAX_QUERY_CHARS = 200;
const TIMEOUT_MS = 9000;
const USER_AGENT = "DREAM-AI/0.5 (public source discovery)";

function clean(value, limit = 700) {
  return typeof value === "string"
    ? value.replace(/<[^>]*>/g, " ").replace(/&amp;/g, "&")
      .replace(/&quot;/g, '"').replace(/&#39;/g, "'")
      .replace(/&lt;/g, "<").replace(/&gt;/g, ">")
      .replace(/[\u0000-\u001f\u007f]/g, " ").replace(/\s+/g, " ").trim().slice(0, limit)
    : "";
}

function safeUrl(value) {
  try {
    const url = new URL(value);
    if (url.protocol !== "https:" || url.username || url.password || url.port) return null;
    return url.href;
  } catch {
    return null;
  }
}

async function request(url, fetchImpl, headers = {}, accept = "application/json") {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const response = await fetchImpl(url, {
      method: "GET",
      headers: { accept, "user-agent": USER_AGENT, ...headers },
      redirect: "error",
      signal: controller.signal
    });
    if (!response.ok) throw new Error("upstream-error");
    return accept.includes("json") ? response.json() : response.text();
  } finally {
    clearTimeout(timer);
  }
}

function result(provider, title, url, snippet, extra = {}) {
  const safe = safeUrl(url);
  const cleanTitle = clean(title, 300);
  if (!safe || !cleanTitle) return null;
  return {
    provider,
    title: cleanTitle,
    url: safe,
    snippet: clean(snippet, 1200),
    sourceType: "public-platform-record",
    ...extra
  };
}

async function runGoogleGrounding(query, fetchImpl, apiKey) {
  const endpoint = new URL("https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent");
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  let payload;
  try {
    const response = await fetchImpl(endpoint, {
      method: "POST",
      headers: { "content-type": "application/json", "user-agent": USER_AGENT, "x-goog-api-key": apiKey },
      redirect: "error",
      signal: controller.signal,
      body: JSON.stringify({
        contents: [{ parts: [{ text: query }] }],
        tools: [{ google_search: {} }]
      })
    });
    if (!response.ok) throw new Error("upstream-error");
    payload = await response.json();
  } finally {
    clearTimeout(timer);
  }
  const candidate = payload?.candidates?.[0];
  const answer = clean(candidate?.content?.parts?.map(part => part.text || "").join("\n"), 4200);
  const chunks = candidate?.groundingMetadata?.groundingChunks;
  if (!Array.isArray(chunks)) throw new Error("google-grounding-metadata-missing");
  const sources = chunks.slice(0, 8).flatMap(chunk => {
    const web = chunk?.web;
    const item = result("Google Search", web?.title || "Google Search source", web?.uri, answer);
    return item ? [item] : [];
  });
  return { answer, queries: (candidate?.groundingMetadata?.webSearchQueries || []).slice(0, 6).map(value => clean(value, 200)), sources };
}

async function searchYouTube(query, fetchImpl, apiKey) {
  const endpoint = new URL("https://www.googleapis.com/youtube/v3/search");
  endpoint.search = new URLSearchParams({
    part: "snippet",
    q: query,
    type: "video",
    maxResults: "8",
    order: "relevance",
    key: apiKey
  }).toString();
  const payload = await request(endpoint, fetchImpl);
  if (!Array.isArray(payload?.items)) throw new Error("invalid-youtube-response");
  return payload.items.slice(0, 8).flatMap(item => {
    const id = clean(item?.id?.videoId, 32);
    const snippet = item?.snippet || {};
    const record = result(
      "YouTube Data API",
      snippet.title,
      id ? "https://www.youtube.com/watch?v=" + encodeURIComponent(id) : "",
      snippet.description,
      { channel: clean(snippet.channelTitle, 160), publishedAt: clean(snippet.publishedAt, 40) }
    );
    return record ? [record] : [];
  });
}

async function searchBluesky(query, fetchImpl) {
  const endpoint = new URL("https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts");
  endpoint.search = new URLSearchParams({ q: query, limit: "10", sort: "latest" }).toString();
  const payload = await request(endpoint, fetchImpl);
  if (!Array.isArray(payload?.posts)) throw new Error("invalid-bluesky-response");
  return payload.posts.slice(0, 10).flatMap(post => {
    const author = post?.author || {};
    const uri = clean(post?.uri, 500);
    const key = uri.match(/\/app\.bsky\.feed\.post\/([A-Za-z0-9]+)/)?.[1];
    const handle = clean(author.handle, 200);
    const url = key && handle ? "https://bsky.app/profile/" + encodeURIComponent(handle) + "/post/" + key : "";
    const record = result(
      "Bluesky",
      clean(author.displayName || handle, 160) + " · public post",
      url,
      post?.record?.text,
      { handle, indexedAt: clean(post?.indexedAt, 40) }
    );
    return record ? [record] : [];
  });
}

async function searchX(query, fetchImpl, bearerToken) {
  const endpoint = new URL("https://api.x.com/2/tweets/search/recent");
  endpoint.search = new URLSearchParams({
    query,
    max_results: "10",
    "tweet.fields": "created_at,lang,public_metrics,author_id",
    expansions: "author_id",
    "user.fields": "name,username,verified"
  }).toString();
  const payload = await request(endpoint, fetchImpl, { authorization: "Bearer " + bearerToken });
  if (payload?.data !== undefined && !Array.isArray(payload.data)) throw new Error("invalid-x-response");
  const users = new Map((payload.includes?.users || []).map(user => [user.id, user]));
  return (payload.data || []).slice(0, 10).flatMap(post => {
    const author = users.get(post.author_id) || {};
    const handle = clean(author.username, 100);
    const record = result(
      "X API",
      (handle ? "@" + handle : "Public post") + " · X post",
      "https://x.com/" + (handle || "i") + "/status/" + encodeURIComponent(post.id),
      post.text,
      { handle, publishedAt: clean(post.created_at, 40), language: clean(post.lang, 20), publicMetrics: post.public_metrics || {} }
    );
    return record ? [record] : [];
  });
}

function xmlDecode(value) {
  return clean(value, 900).replace(/&apos;/g, "'").replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(Number(n)));
}

async function searchGoogleNewsRss(query, fetchImpl) {
  const endpoint = new URL("https://news.google.com/rss/search");
  endpoint.search = new URLSearchParams({ q: query, hl: "en-KE", gl: "KE", ceid: "KE:en" }).toString();
  const xml = await request(endpoint, fetchImpl, {}, "application/rss+xml,text/xml");
  if (typeof xml !== "string") throw new Error("invalid-google-news-rss");
  const items = [...xml.matchAll(/<item>([\s\S]*?)<\/item>/gi)].slice(0, 8);
  return items.flatMap(([_, block]) => {
    const patterns = {
      title: /<title>([\s\S]*?)<\/title>/i,
      link: /<link>([\s\S]*?)<\/link>/i,
      pubDate: /<pubDate>([\s\S]*?)<\/pubDate>/i
    };
    const tag = name => block.match(patterns[name])?.[1] || "";
    const title = xmlDecode(tag("title"));
    const link = xmlDecode(tag("link"));
    const publishedAt = xmlDecode(tag("pubDate"));
    const record = result("Google News RSS", title, link, "", { publishedAt });
    return record ? [record] : [];
  });
}

export async function searchPlatformSources(query, fetchImpl = fetch, env = process.env) {
  if (typeof query !== "string" || !query.trim() || query.trim().length > MAX_QUERY_CHARS) {
    throw new RangeError("Research query must contain 1 to 200 characters.");
  }
  const normalized = query.trim();
  const tasks = await Promise.all([
    (env.DREAM_GOOGLE_GEMINI_API_KEY
      ? runGoogleGrounding(normalized, fetchImpl, env.DREAM_GOOGLE_GEMINI_API_KEY)
        .then(data => ({ provider: "Google Search", status: data.sources.length ? "ok" : "no-results", results: data.sources, answer: data.answer, queries: data.queries }))
        .catch(error => ({ provider: "Google Search", status: error.name === "AbortError" ? "timeout" : "unavailable", results: [] }))
      : Promise.resolve({ provider: "Google Search", status: "not-configured", results: [] })),
    (env.DREAM_YOUTUBE_API_KEY
      ? searchYouTube(normalized, fetchImpl, env.DREAM_YOUTUBE_API_KEY)
        .then(results => ({ provider: "YouTube Data API", status: results.length ? "ok" : "no-results", results }))
        .catch(error => ({ provider: "YouTube Data API", status: error.name === "AbortError" ? "timeout" : "unavailable", results: [] }))
      : Promise.resolve({ provider: "YouTube Data API", status: "not-configured", results: [] })),
    searchBluesky(normalized, fetchImpl)
      .then(results => ({ provider: "Bluesky", status: results.length ? "ok" : "no-results", results }))
      .catch(error => ({ provider: "Bluesky", status: error.name === "AbortError" ? "timeout" : "unavailable", results: [] })),
    (env.DREAM_X_BEARER_TOKEN
      ? searchX(normalized, fetchImpl, env.DREAM_X_BEARER_TOKEN)
        .then(results => ({ provider: "X API", status: results.length ? "ok" : "no-results", results }))
        .catch(error => ({ provider: "X API", status: error.name === "AbortError" ? "timeout" : "unavailable", results: [] }))
      : Promise.resolve({ provider: "X API", status: "not-configured", results: [] })),
    searchGoogleNewsRss(normalized, fetchImpl)
      .then(results => ({ provider: "Google News RSS", status: results.length ? "ok" : "no-results", results }))
      .catch(error => ({ provider: "Google News RSS", status: error.name === "AbortError" ? "timeout" : "unavailable", results: [] }))
  ]);
  const successful = tasks.filter(item => item.status === "ok" || item.status === "no-results");
  if (!successful.length) throw new Error("All platform source providers are unavailable.");
  return {
    query: normalized,
    providers: tasks.map(({ provider, status }) => ({ provider, status })),
    results: tasks.flatMap(item => item.results || []),
    googleGroundedAnswer: tasks.find(item => item.provider === "Google Search")?.answer || "",
    googleSearchQueries: tasks.find(item => item.provider === "Google Search")?.queries || [],
    limitations: [
      "Google Search grounding is an externally generated, cited research result; it does not change or train the DREAM model.",
      "YouTube results are public video metadata and descriptions, not transcripts or private account data.",
      "Bluesky and X results are public posts subject to each platform's access and use rules.",
      "Google News RSS provides headlines and publisher links, not complete article text.",
      "Unavailable or unconfigured providers are reported rather than silently represented as searched."
    ]
  };
}
