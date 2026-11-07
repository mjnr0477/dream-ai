const MAX_QUERY_CHARS = 200;
const RESULT_LIMIT = 6;
const REQUEST_TIMEOUT_MS = 7000;
const USER_AGENT = "DREAM-AI/0.4 (public source research)";

function safeText(value, limit = 600) {
  return typeof value === "string"
    ? value.replace(/<[^>]*>/g, " ")
      .replace(/&amp;/g, "&").replace(/&quot;/g, '"')
      .replace(/&#39;/g, "'").replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">").replace(/[\u0000-\u001f\u007f]/g, " ")
      .replace(/\s+/g, " ").trim().slice(0, limit)
    : "";
}

function safeHttpsUrl(value) {
  try {
    const url = new URL(value);
    if (url.protocol !== "https:" || url.username || url.password || url.port) return null;
    return url.href;
  } catch {
    return null;
  }
}

async function requestJson(url, fetchImpl, headers = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetchImpl(url, {
      method: "GET",
      headers: {
        accept: "application/json",
        "user-agent": USER_AGENT,
        ...headers
      },
      redirect: "error",
      signal: controller.signal
    });
    if (!response.ok) throw new Error("upstream-error");
    return await response.json();
  } finally {
    clearTimeout(timeout);
  }
}

async function searchWikidata(query, fetchImpl) {
  const endpoint = new URL("https://www.wikidata.org/w/api.php");
  endpoint.search = new URLSearchParams({
    action: "wbsearchentities",
    search: query,
    language: "en",
    uselang: "en",
    type: "item",
    limit: String(RESULT_LIMIT),
    format: "json"
  }).toString();
  const payload = await requestJson(endpoint, fetchImpl);
  if (!Array.isArray(payload?.search)) throw new Error("invalid-wikidata-response");
  return payload.search.slice(0, RESULT_LIMIT).flatMap(item => {
    const id = typeof item?.id === "string" && /^Q[1-9]\d*$/.test(item.id) ? item.id : null;
    const title = safeText(item?.label, 200);
    const description = safeText(item?.description, 500);
    if (!id || !title) return [];
    return [{
      provider: "Wikidata",
      title,
      description,
      entityId: id,
      url: "https://www.wikidata.org/wiki/" + id,
      sourceType: "public-entity-record"
    }];
  });
}

function parseGdeltDate(value) {
  if (typeof value !== "string" || !/^\d{8}T\d{6}Z$/.test(value)) return null;
  const date = new Date(
    value.slice(0, 4) + "-" + value.slice(4, 6) + "-" + value.slice(6, 8) +
    "T" + value.slice(9, 11) + ":" + value.slice(11, 13) + ":" + value.slice(13, 15) + "Z"
  );
  return Number.isNaN(date.getTime()) ? null : date.toISOString();
}

async function searchRecentNews(query, fetchImpl) {
  const endpoint = new URL("https://api.gdeltproject.org/api/v2/doc/doc");
  endpoint.search = new URLSearchParams({
    query,
    mode: "ArtList",
    format: "json",
    maxrecords: String(RESULT_LIMIT),
    sort: "DateDesc",
    timespan: "7d"
  }).toString();
  const payload = await requestJson(endpoint, fetchImpl);
  if (payload?.articles !== undefined && !Array.isArray(payload.articles)) {
    throw new Error("invalid-gdelt-response");
  }
  return (payload.articles || []).slice(0, RESULT_LIMIT).flatMap(article => {
    const title = safeText(article?.title, 300);
    const url = safeHttpsUrl(article?.url);
    if (!title || !url) return [];
    let domain = "";
    try { domain = new URL(url).hostname; } catch {}
    return [{
      provider: "GDELT",
      title,
      domain,
      url,
      publishedAt: parseGdeltDate(article?.seendate),
      language: safeText(article?.language, 40),
      sourceCountry: safeText(article?.sourcecountry, 80),
      sourceType: "recent-news-headline"
    }];
  });
}

async function searchBraveWeb(query, fetchImpl, apiKey) {
  const endpoint = new URL("https://api.search.brave.com/res/v1/web/search");
  endpoint.search = new URLSearchParams({
    q: query,
    count: String(RESULT_LIMIT),
    safesearch: "moderate"
  }).toString();
  const payload = await requestJson(endpoint, fetchImpl, {
    "X-Subscription-Token": apiKey
  });
  const webResults = payload?.web?.results;
  if (!Array.isArray(webResults)) throw new Error("invalid-brave-response");
  return webResults.slice(0, RESULT_LIMIT).flatMap(item => {
    const title = safeText(item?.title, 300);
    const description = safeText(item?.description, 700);
    const url = safeHttpsUrl(item?.url);
    if (!title || !description || !url) return [];
    return [{
      provider: "Brave Search",
      title,
      snippet: description,
      url,
      sourceType: "web-search-result"
    }];
  });
}

export async function searchGlobalPublicSources(
  query,
  fetchImpl = fetch,
  env = process.env
) {
  if (typeof query !== "string") throw new TypeError("Research query must be text.");
  const normalized = query.trim();
  if (!normalized || normalized.length > MAX_QUERY_CHARS) {
    throw new RangeError("Research query must contain 1 to 200 characters.");
  }

  const providers = await Promise.all([
    searchWikidata(normalized, fetchImpl)
      .then(results => ({ name: "Wikidata", status: results.length ? "ok" : "no-results", results }))
      .catch(error => ({ name: "Wikidata", status: error.name === "AbortError" ? "timeout" : "unavailable", results: [] })),
    searchRecentNews(normalized, fetchImpl)
      .then(results => ({ name: "GDELT", status: results.length ? "ok" : "no-results", results }))
      .catch(error => ({ name: "GDELT", status: error.name === "AbortError" ? "timeout" : "unavailable", results: [] })),
    env.DREAM_BRAVE_SEARCH_API_KEY
      ? searchBraveWeb(normalized, fetchImpl, env.DREAM_BRAVE_SEARCH_API_KEY)
        .then(results => ({ name: "Brave Search", status: results.length ? "ok" : "no-results", results }))
        .catch(error => ({ name: "Brave Search", status: error.name === "AbortError" ? "timeout" : "unavailable", results: [] }))
      : Promise.resolve({ name: "Brave Search", status: "not-configured", results: [] })
  ]);

  const available = providers.some(provider => provider.status === "ok" || provider.status === "no-results");
  if (!available) throw new Error("All global public research providers are unavailable.");
  return {
    query: normalized,
    researchType: "global-public-source-discovery",
    providers: providers.map(({ name, status }) => ({ provider: name, status })),
    providerCountSucceeded: providers.filter(provider => provider.status === "ok" || provider.status === "no-results").length,
    entities: providers.find(provider => provider.name === "Wikidata")?.results || [],
    news: providers.find(provider => provider.name === "GDELT")?.results || [],
    webResults: providers.find(provider => provider.name === "Brave Search")?.results || [],
    limitations: [
      "Wikidata results are public structured entity records, not a complete dossier about a person.",
      "GDELT results are recent news headlines from the preceding seven days; they are not a live event feed or a complete news index.",
      "General web-index results are enabled only when DREAM_BRAVE_SEARCH_API_KEY is configured.",
      "Search results are leads for verification, not proof that every statement in a source is true."
    ]
  };
}
