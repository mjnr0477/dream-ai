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
    // Google returns one grounded synthesis for the full source bundle, not
    // a separate excerpt for each URL. Keep individual source records as links
    // only so the synthesis cannot be misattributed to one publisher.
    const item = result(
      "Google Search",
      web?.title || "Google Search source",
      web?.uri,
      ""
    );
    return item ? [item] : [];
  });
  return {
    answer,
    queries: (candidate?.groundingMetadata?.webSearchQueries || []).slice(0, 6).map(value => clean(value, 200)),
    sources
  };
}

async function searchYouTube(query, fetchImpl, apiKey) {
  const endpoint = new URL("https://www.googleapis.com/youtube/v3/search");
  endpoint.search = new URLSearchParams({
    part: "snippet",
    q: query,
    type: "video",
    maxResults: "8",
    order: "relevance"
  }).toString();
  const payload = await request(endpoint, fetchImpl, { "x-goog-api-key": apiKey });
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


async function searchStackExchange(query, fetchImpl) {
  const endpoint = new URL("https://api.stackexchange.com/2.3/search/advanced");
  endpoint.search = new URLSearchParams({
    order: "desc",
    sort: "relevance",
    q: query,
    site: "stackoverflow",
    pagesize: "8",
    filter: "withbody"
  }).toString();
  const payload = await request(endpoint, fetchImpl);
  if (!Array.isArray(payload?.items)) throw new Error("invalid-stackexchange-response");
  const results = payload.items.slice(0, 8).flatMap(item => {
    const url = safeUrl(item?.link);
    const title = clean(item?.title, 300);
    if (!url || !title || !url.startsWith("https://stackoverflow.com/questions/")) return [];
    const createdAt = Number.isSafeInteger(item?.creation_date)
      ? new Date(item.creation_date * 1000).toISOString()
      : null;
    const tags = Array.isArray(item?.tags)
      ? item.tags.slice(0, 8).map(tag => clean(tag, 60)).filter(Boolean)
      : [];
    const recordItem = result(
      "Stack Exchange",
      title,
      url,
      item?.body || "",
      {
        site: "Stack Overflow",
        tags,
        score: Number.isSafeInteger(item?.score) ? item.score : null,
        answerCount: Number.isSafeInteger(item?.answer_count) ? item.answer_count : null,
        answered: item?.is_answered === true,
        ...(createdAt ? { publishedAt: createdAt } : {})
      }
    );
    return recordItem ? [recordItem] : [];
  });
  return {
    results,
    quotaRemaining: Number.isSafeInteger(payload?.quota_remaining) ? payload.quota_remaining : null,
    backoffSeconds: Number.isSafeInteger(payload?.backoff) ? payload.backoff : null
  };
}

function xmlDecode(value) {
  return clean(value, 900).replace(/&apos;/g, "'").replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(Number(n)));
}

async function searchEuropePmc(query, fetchImpl) {
  const endpoint = new URL("https://www.ebi.ac.uk/europepmc/webservices/rest/search");
  endpoint.search = new URLSearchParams({
    query,
    format: "json",
    resultType: "core",
    pageSize: "6"
  }).toString();
  const payload = await request(endpoint, fetchImpl);
  const records = payload?.resultList?.result;
  if (!Array.isArray(records)) throw new Error("invalid-europe-pmc-response");
  return records.slice(0, 6).flatMap(item => {
    const id = clean(item?.pmid || item?.id, 40);
    const url = item?.pmcid
      ? "https://europepmc.org/articles/" + encodeURIComponent(item.pmcid)
      : id ? "https://europepmc.org/article/MED/" + encodeURIComponent(id) : "";
    const record = result(
      "Europe PMC",
      item?.title,
      url,
      item?.abstractText || item?.authorString || "",
      {
        authors: clean(item?.authorString, 400),
        journal: clean(item?.journalInfo?.journal?.title || item?.journalTitle, 180),
        publishedAt: clean(item?.firstPublicationDate || item?.pubYear, 40),
        pmid: id,
        openAccess: item?.isOpenAccess === "Y"
      }
    );
    return record ? [record] : [];
  });
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


const PUBMED_REQUEST_INTERVAL_MS = 450;
let pubmedRequestQueue = Promise.resolve();
let nextPubmedRequestAt = 0;

async function requestPubMed(url, fetchImpl, accept = "application/json") {
  const task = pubmedRequestQueue.then(async () => {
    const waitMs = Math.max(0, nextPubmedRequestAt - Date.now());
    if (waitMs > 0) await new Promise(resolve => setTimeout(resolve, waitMs));
    nextPubmedRequestAt = Date.now() + PUBMED_REQUEST_INTERVAL_MS;
    return request(url, fetchImpl, {}, accept);
  });
  pubmedRequestQueue = task.catch(() => {});
  return task;
}

function decodePubMedXmlText(value, maxLength = 1200) {
  const stripped = clean(value, maxLength)
    .replace(/&#x([0-9a-f]+);/gi, (_, hex) => {
      const code = Number.parseInt(hex, 16);
      return Number.isInteger(code) && code > 0 && code <= 0x10ffff &&
        !(code >= 0xd800 && code <= 0xdfff)
        ? String.fromCodePoint(code) : " ";
    })
    .replace(/&#(\d+);/g, (_, decimal) => {
      const code = Number(decimal);
      return Number.isInteger(code) && code > 0 && code <= 0x10ffff &&
        !(code >= 0xd800 && code <= 0xdfff)
        ? String.fromCodePoint(code) : " ";
    });
  return clean(stripped, maxLength);
}

function pubmedElement(block, tag) {
  const match = block.match(new RegExp("<" + tag + "\\b[^>]*>([\\\\s\\\\S]*?)</" + tag + ">", "i"));
  return match ? decodePubMedXmlText(match[1], 1200) : "";
}

function parsePubMedArticles(xml) {
  if (typeof xml !== "string" || !/<PubmedArticleSet\b/i.test(xml)) {
    throw new Error("invalid-pubmed-response");
  }
  const articles = [...xml.matchAll(/<PubmedArticle\b[^>]*>([\s\S]*?)<\/PubmedArticle>/gi)];
  return articles.slice(0, 5).flatMap(match => {
    const block = match[1];
    const pmid = pubmedElement(block, "PMID");
    const title = pubmedElement(block, "ArticleTitle");
    if (!/^\d{1,12}$/.test(pmid) || !title) return [];

    const abstractParts = [...block.matchAll(/<AbstractText\b([^>]*)>([\s\S]*?)<\/AbstractText>/gi)]
      .map(part => {
        const labelMatch = part[1].match(/\b(?:Label|NlmCategory)="([^"]+)"/i);
        const label = labelMatch ? decodePubMedXmlText(labelMatch[1], 80) : "";
        const text = decodePubMedXmlText(part[2], 1000);
        return text ? (label ? label + ": " : "") + text : "";
      })
      .filter(Boolean);
    const abstract = abstractParts.join(" ").slice(0, 1100);

    const authors = [...block.matchAll(/<Author\b[^>]*>([\s\S]*?)<\/Author>/gi)]
      .slice(0, 8)
      .map(authorMatch => {
        const author = authorMatch[1];
        const collective = pubmedElement(author, "CollectiveName");
        if (collective) return collective;
        const given = pubmedElement(author, "ForeName") || pubmedElement(author, "Initials");
        const family = pubmedElement(author, "LastName");
        return [given, family].filter(Boolean).join(" ");
      })
      .filter(Boolean);

    const journalBlock = block.match(/<Journal\b[^>]*>([\s\S]*?)<\/Journal>/i)?.[1] || "";
    const journal = pubmedElement(journalBlock, "Title");
    const pubDateBlock = block.match(/<PubDate\b[^>]*>([\s\S]*?)<\/PubDate>/i)?.[1] || "";
    const publicationYear = pubmedElement(pubDateBlock, "Year") ||
      pubmedElement(pubDateBlock, "MedlineDate").match(/\b(?:19|20)\d{2}\b/)?.[0] || "";

    const recordItem = result(
      "PubMed",
      title,
      "https://pubmed.ncbi.nlm.nih.gov/" + pmid + "/",
      abstract,
      {
        sourceType: abstract ? "pubmed-abstract" : "pubmed-bibliographic-record",
        pmid,
        authors,
        ...(journal ? { journal } : {}),
        ...(publicationYear ? { publishedAt: publicationYear } : {}),
        abstractAvailable: Boolean(abstract)
      }
    );
    return recordItem ? [recordItem] : [];
  });
}

export async function searchPubMed(query, fetchImpl = fetch) {
  if (typeof query !== "string" || !query.trim() || query.trim().length > MAX_QUERY_CHARS) {
    throw new RangeError("PubMed query must contain 1 to 200 characters.");
  }
  const normalized = query.trim();
  const searchUrl = new URL("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi");
  searchUrl.search = new URLSearchParams({
    db: "pubmed",
    term: normalized,
    retmode: "json",
    retmax: "5",
    sort: "relevance",
    tool: "DREAM_AI"
  }).toString();

  const searchPayload = await requestPubMed(searchUrl, fetchImpl);
  const ids = searchPayload?.esearchresult?.idlist;
  if (!Array.isArray(ids) || ids.some(id => typeof id !== "string" || !/^\d{1,12}$/.test(id))) {
    throw new Error("invalid-pubmed-search-response");
  }
  const limitedIds = ids.slice(0, 5);
  if (!limitedIds.length) return [];

  const fetchUrl = new URL("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi");
  fetchUrl.search = new URLSearchParams({
    db: "pubmed",
    id: limitedIds.join(","),
    rettype: "abstract",
    retmode: "xml",
    tool: "DREAM_AI"
  }).toString();
  const xml = await requestPubMed(fetchUrl, fetchImpl, "application/xml,text/xml");
  return parsePubMedArticles(xml);
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
      .catch(error => ({ provider: "Google News RSS", status: error.name === "AbortError" ? "timeout" : "unavailable", results: [] })),
    searchStackExchange(normalized, fetchImpl)
      .then(data => ({ provider: "Stack Exchange", status: data.results.length ? "ok" : "no-results", results: data.results, quotaRemaining: data.quotaRemaining, backoffSeconds: data.backoffSeconds }))
      .catch(error => ({ provider: "Stack Exchange", status: error.name === "AbortError" ? "timeout" : "unavailable", results: [] })),
    searchEuropePmc(normalized, fetchImpl)
      .then(results => ({ provider: "Europe PMC", status: results.length ? "ok" : "no-results", results }))
      .catch(error => ({ provider: "Europe PMC", status: error.name === "AbortError" ? "timeout" : "unavailable", results: [] })),
    searchPubMed(normalized, fetchImpl)
      .then(results => ({ provider: "PubMed", status: results.length ? "ok" : "no-results", results }))
      .catch(error => ({ provider: "PubMed", status: error.name === "AbortError" ? "timeout" : "unavailable", results: [] }))
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
      "Europe PMC provides publication metadata and abstracts where supplied; it does not mean all linked full texts are open access.",
      "PubMed provides citation metadata and abstracts where supplied; an abstract is not the full article and article-level rights still apply.",
      "Stack Exchange results are public Q&A excerpts under site-specific attribution and license terms; they are for retrieval and are not automatically approved training data.",
      "Unavailable or unconfigured providers are reported rather than silently represented as searched."
    ]
  };
}
