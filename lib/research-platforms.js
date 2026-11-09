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

const WORLD_BANK_INDICATORS = [
  { code: "NY.GDP.PCAP.CD", name: "GDP per capita (current US$)", terms: ["gdp per capita", "income per person"] },
  { code: "NY.GDP.MKTP.KD.ZG", name: "GDP growth (annual %)", terms: ["gdp growth", "economic growth"] },
  { code: "SP.POP.TOTL", name: "Population, total", terms: ["population", "number of people"] },
  { code: "NY.GDP.MKTP.CD", name: "GDP (current US$)", terms: ["gross domestic product", "gdp", "economy"] },
  { code: "FP.CPI.TOTL.ZG", name: "Inflation, consumer prices (annual %)", terms: ["inflation", "consumer prices"] },
  { code: "SL.UEM.TOTL.ZS", name: "Unemployment, total (% of labour force)", terms: ["unemployment", "jobless rate"] },
  { code: "SP.DYN.LE00.IN", name: "Life expectancy at birth, total (years)", terms: ["life expectancy"] },
  { code: "SE.ADT.LITR.ZS", name: "Literacy rate, adult total (% of people ages 15 and above)", terms: ["literacy rate", "literacy"] },
  { code: "SE.PRM.ENRR", name: "School enrollment, primary (% gross)", terms: ["school enrollment", "school enrolment", "primary enrollment", "primary enrolment"] },
  { code: "EG.ELC.ACCS.ZS", name: "Access to electricity (% of population)", terms: ["electricity access", "access to electricity"] },
  { code: "IT.NET.USER.ZS", name: "Individuals using the Internet (% of population)", terms: ["internet users", "internet access", "internet penetration"] },
  { code: "SI.POV.NAHC", name: "Poverty headcount ratio at national poverty lines (% of population)", terms: ["poverty rate", "poverty headcount", "poverty"] },
  { code: "EN.ATM.CO2E.PC", name: "CO2 emissions (metric tons per capita)", terms: ["co2 emissions", "carbon emissions"] },
  { code: "SH.DYN.MORT", name: "Mortality rate, under-5 (per 1,000 live births)", terms: ["child mortality", "under five mortality", "under 5 mortality"] },
  { code: "SH.STA.MMRT", name: "Maternal mortality ratio (per 100,000 live births)", terms: ["maternal mortality"] },
  { code: "SL.AGR.EMPL.ZS", name: "Employment in agriculture (% of total employment)", terms: ["agricultural employment", "agriculture employment"] },
  { code: "TX.VAL.MRCH.CD.WT", name: "Merchandise exports (current US$)", terms: ["merchandise exports", "exports"] },
  { code: "TM.VAL.MRCH.CD.WT", name: "Merchandise imports (current US$)", terms: ["merchandise imports", "imports"] }
];

const WORLD_BANK_COUNTRY_CACHE = new WeakMap();
const WORLD_BANK_COUNTRY_CACHE_TTL = 24 * 60 * 60 * 1000;

function normalizeWorldBankText(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

function containsWorldBankPhrase(text, phrase) {
  const haystack = " " + normalizeWorldBankText(text) + " ";
  const needle = " " + normalizeWorldBankText(phrase) + " ";
  return haystack.includes(needle);
}

function findWorldBankIndicator(query) {
  return WORLD_BANK_INDICATORS
    .map((indicator, index) => ({
      indicator,
      index,
      matchedLength: Math.max(
        0,
        ...indicator.terms
          .filter(term => containsWorldBankPhrase(query, term))
          .map(term => normalizeWorldBankText(term).length)
      )
    }))
    .filter(match => match.matchedLength > 0)
    .sort((a, b) => b.matchedLength - a.matchedLength || a.index - b.index)[0]
    ?.indicator || null;
}

async function loadWorldBankCountries(fetchImpl) {
  const cached = WORLD_BANK_COUNTRY_CACHE.get(fetchImpl);
  if (cached && cached.expiresAt > Date.now()) return cached.promise;

  const endpoint = new URL("https://api.worldbank.org/v2/country");
  endpoint.search = new URLSearchParams({ format: "json", per_page: "400" }).toString();
  const promise = request(endpoint, fetchImpl).then(payload => {
    const records = payload?.[1];
    if (!Array.isArray(records)) throw new Error("invalid-world-bank-country-response");
    return records.filter(country =>
      /^[A-Z]{3}$/.test(country?.id || "") &&
      country?.name &&
      country?.region?.value &&
      country.region.value !== "Aggregates"
    );
  }).catch(error => {
    WORLD_BANK_COUNTRY_CACHE.delete(fetchImpl);
    throw error;
  });
  WORLD_BANK_COUNTRY_CACHE.set(fetchImpl, {
    expiresAt: Date.now() + WORLD_BANK_COUNTRY_CACHE_TTL,
    promise
  });
  return promise;
}

function worldBankCountryCodes(query, countries) {
  const aliases = [
    { code: "USA", names: ["united states", "usa"] },
    { code: "GBR", names: ["united kingdom", "great britain", "britain", "uk"] },
    { code: "ARE", names: ["united arab emirates", "uae"] },
    { code: "COD", names: ["democratic republic of the congo", "dr congo", "drc"] },
    { code: "COG", names: ["republic of the congo"] },
    { code: "KOR", names: ["south korea", "republic of korea"] },
    { code: "PRK", names: ["north korea"] },
    { code: "RUS", names: ["russia"] },
    { code: "TUR", names: ["turkey"] },
    { code: "VNM", names: ["vietnam"] }
  ];
  if (containsWorldBankPhrase(query, "world") || containsWorldBankPhrase(query, "worldwide") ||
      containsWorldBankPhrase(query, "global")) {
    return [{ id: "WLD", name: "World", length: 5 }];
  }
  if (containsWorldBankPhrase(query, "sub saharan africa")) {
    return [{ id: "SSF", name: "Sub-Saharan Africa", length: 18 }];
  }

  const matched = countries.flatMap(country => {
    const name = normalizeWorldBankText(country.name);
    return containsWorldBankPhrase(query, name)
      ? [{ id: country.id, name: country.name, length: name.length }]
      : [];
  });
  for (const alias of aliases) {
    if (alias.names.some(name => containsWorldBankPhrase(query, name))) {
      const country = countries.find(item => item.id === alias.code);
      if (country) matched.push({
        id: country.id,
        name: country.name,
        length: Math.max(...alias.names.map(name => name.length))
      });
    }
  }
  const unique = [...new Map(matched.map(item => [item.id, item])).values()]
    .sort((a, b) => b.length - a.length)
    .slice(0, 4);
  return unique.length ? unique : [{ id: "WLD", name: "World", length: 5 }];
}

async function searchWorldBankIndicators(query, fetchImpl) {
  const indicator = findWorldBankIndicator(query);
  if (!indicator) return { applicable: false, results: [] };

  const countries = await loadWorldBankCountries(fetchImpl);
  const selected = worldBankCountryCodes(query, countries);
  const countryPath = selected.map(item => item.id).join(";");
  const endpoint = new URL(
    "https://api.worldbank.org/v2/country/" + countryPath + "/indicator/" + indicator.code
  );
  const requestedYears = [...query.matchAll(/\b(?:19|20)\d{2}\b/g)].map(match => match[0]);
  const dateRange = requestedYears.length > 1
    ? Math.min(...requestedYears.map(Number)) + ":" + Math.max(...requestedYears.map(Number))
    : requestedYears[0] || "";
  const dataParams = { format: "json", per_page: "25" };
  if (dateRange) dataParams.date = dateRange;
  else dataParams.mrnev = "5";
  endpoint.search = new URLSearchParams(dataParams).toString();
  const payload = await request(endpoint, fetchImpl);
  const rows = payload?.[1];
  if (!Array.isArray(rows)) return { applicable: true, results: [] };

  const selectedNames = new Map(selected.map(item => [item.id, item.name]));
  const records = rows.filter(row =>
    typeof row?.date === "string" &&
    /^\d{4}$/.test(row.date) &&
    row.value !== null &&
    row.value !== "" &&
    Number.isFinite(Number(row.value))
  ).slice(0, 12).flatMap(row => {
    const code = row?.country?.id || row?.countryiso3code;
    if (!/^[A-Z]{3}$/.test(code || "")) return [];
    const country = selectedNames.get(code) || clean(row?.country?.value, 120);
    if (!country) return [];
    const numeric = Number(row.value);
    const formatted = new Intl.NumberFormat("en", { maximumFractionDigits: 3 }).format(numeric);
    const unit = clean(row?.unit, 80);
    const snippet = [
      indicator.name + " for " + country + ": " + formatted + (unit ? " " + unit : ""),
      "Observation year: " + row.date,
      "This is the latest non-empty reported observation returned by the World Bank API, not a live measurement."
    ].join(". ");
    const sourceUrl = new URL(
      "https://api.worldbank.org/v2/country/" + code + "/indicator/" + indicator.code
    );
    const sourceParams = { format: "json" };
    if (dateRange) sourceParams.date = dateRange;
    else sourceParams.mrnev = "5";
    sourceUrl.search = new URLSearchParams(sourceParams).toString();
    const item = result(
      "World Bank Indicators",
      indicator.name + " — " + country,
      sourceUrl.href,
      snippet,
      {
        country,
        countryCode: code,
        indicatorCode: indicator.code,
        observationYear: row.date,
        value: numeric,
        unit,
        sourceType: "official development indicator data",
        ...(payload?.[0]?.lastupdated ? { indexedAt: clean(payload[0].lastupdated, 40) } : {})
      }
    );
    return item ? [item] : [];
  });
  return { applicable: true, results: records };
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
      .catch(error => ({ provider: "Europe PMC", status: error.name === "AbortError" ? "timeout" : "unavailable", results: [] }))
    ,
    searchWorldBankIndicators(normalized, fetchImpl)
      .then(data => ({
        provider: "World Bank Indicators",
        status: !data.applicable ? "not-applicable" : data.results.length ? "ok" : "no-results",
        results: data.results
      }))
      .catch(error => ({ provider: "World Bank Indicators", status: error.name === "AbortError" ? "timeout" : "unavailable", results: [] }))
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
      "Stack Exchange results are public Q&A excerpts under site-specific attribution and license terms; they are for retrieval and are not automatically approved training data.",
      "World Bank indicator values are official statistical observations for the reported year; they are not live measurements, and each dataset's license exceptions still apply.",
      "Unavailable or unconfigured providers are reported rather than silently represented as searched."
    ]
  };
}
