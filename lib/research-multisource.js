const MAX_QUERY_CHARS = 200;
const RESULT_LIMIT = 5;
const REQUEST_TIMEOUT_MS = 7000;
const USER_AGENT = "DREAM-AI/0.4 (multi-source public research)";

let arxivRequestQueue = Promise.resolve();
let arxivLastRequestAt = 0;

function safeText(value, limit = 500) {
  return typeof value === "string"
    ? value.replace(/<[^>]*>/g, " ").replace(/&amp;/g, "&")
      .replace(/&quot;/g, '"').replace(/&#39;/g, "'")
      .replace(/&lt;/g, "<").replace(/&gt;/g, ">")
      .replace(/\s+/g, " ").trim().slice(0, limit)
    : "";
}

function safeHttpsUrl(value, allowedHosts) {
  try {
    const url = new URL(value);
    return url.protocol === "https:" && !url.username && !url.password && !url.port && allowedHosts.includes(url.hostname) ? url.href : null;
  } catch {
    return null;
  }
}

function freshness(date, now) {
  if (!date || !/^\d{4}(-\d{2}(-\d{2})?)?$/.test(date)) return "unknown";
  const year = Number(date.slice(0, 4));
  const age = now.getUTCFullYear() - year;
  if (age < 0) return "future-dated";
  return age <= 2 ? "recent" : "older";
}

async function requestJson(url, fetchImpl) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetchImpl(url, {
      method: "GET",
      headers: { accept: "application/json", "user-agent": USER_AGENT },
      redirect: "error",
      signal: controller.signal
    });
    if (!response.ok) throw new Error("upstream-error");
    return await response.json();
  } finally {
    clearTimeout(timeout);
  }
}

function normalizeCrossref(payload, now) {
  const items = payload?.message?.items;
  if (!Array.isArray(items)) throw new Error("invalid-crossref-response");
  return items.slice(0, RESULT_LIMIT).flatMap(item => {
    const title = safeText(item?.title?.[0], 300);
    if (!title) return [];
    const doi = typeof item.DOI === "string" ? item.DOI.toLowerCase() : null;
    const dateParts = item.published?.["date-parts"]?.[0] ?? item.created?.["date-parts"]?.[0];
    const publicationDate = Array.isArray(dateParts) && Number.isInteger(dateParts[0])
      ? dateParts.map((part, index) => index === 0 ? String(part) : String(part).padStart(2, "0")).join("-")
      : null;
    const url = doi ? safeHttpsUrl("https://doi.org/" + encodeURIComponent(item.DOI), ["doi.org"]) : null;
    return [{
      provider: "Crossref",
      providerRecordId: doi,
      title,
      authors: Array.isArray(item.author) ? item.author.slice(0, 8).map(a => safeText([a.given, a.family].filter(Boolean).join(" "), 120)).filter(Boolean) : [],
      publicationDate,
      doi,
      url,
      sourceType: "scholarly-metadata",
      abstract: null,
      freshness: freshness(publicationDate, now)
    }];
  });
}

function normalizeOpenAlex(payload, now) {
  const items = payload?.results;
  if (!Array.isArray(items)) throw new Error("invalid-openalex-response");
  return items.slice(0, RESULT_LIMIT).flatMap(item => {
    const title = safeText(item?.title, 300);
    if (!title) return [];
    const doiRaw = typeof item.doi === "string" ? item.doi : null;
    let doi = null;
    if (doiRaw) {
      try { doi = new URL(doiRaw).hostname === "doi.org" ? decodeURIComponent(new URL(doiRaw).pathname.slice(1)).toLowerCase() : null; } catch {}
    }
    const landing = safeHttpsUrl(item.primary_location?.landing_page_url, ["doi.org", "www.ncbi.nlm.nih.gov", "arxiv.org", "openalex.org"]);
    const idUrl = safeHttpsUrl(item.id, ["openalex.org"]);
    const publicationDate = typeof item.publication_date === "string" ? item.publication_date.slice(0, 10) : null;
    const authors = Array.isArray(item.authorships) ? item.authorships.slice(0, 8).map(a => safeText(a.author?.display_name, 120)).filter(Boolean) : [];
    return [{
      provider: "OpenAlex",
      providerRecordId: idUrl,
      title,
      authors,
      publicationDate,
      doi,
      url: landing || (doi ? "https://doi.org/" + encodeURIComponent(doi) : idUrl),
      sourceType: "scholarly-metadata",
      abstract: null,
      freshness: freshness(publicationDate, now)
    }];
  });
}


function decodeXml(value) {
  return value
    .replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, "$1")
    .replace(/&amp;/g, "&").replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'").replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&#(\d+);/g, (_, code) => {
      const point = Number(code);
      return point > 0 && point <= 0x10ffff ? String.fromCodePoint(point) : "";
    })
    .replace(/&#x([0-9a-f]+);/gi, (_, code) => {
      const point = Number.parseInt(code, 16);
      return point > 0 && point <= 0x10ffff ? String.fromCodePoint(point) : "";
    });
}

function atomText(block, tag) {
  const match = block.match(new RegExp("<" + tag + "(?:\\s[^>]*)?>([\\s\\S]*?)</" + tag + ">", "i"));
  return match ? safeText(decodeXml(match[1].replace(/<[^>]*>/g, " ")), 3000) : "";
}

async function waitForArxivSlot(minIntervalMs) {
  const previous = arxivRequestQueue;
  let release;
  arxivRequestQueue = new Promise(resolve => { release = resolve; });
  await previous;
  try {
    const remaining = minIntervalMs - (Date.now() - arxivLastRequestAt);
    if (remaining > 0) await new Promise(resolve => setTimeout(resolve, remaining));
    arxivLastRequestAt = Date.now();
  } finally {
    release();
  }
}


async function searchSemanticScholar(query, fetchImpl, now) {
  const endpoint = new URL("https://api.semanticscholar.org/graph/v1/paper/search");
  endpoint.search = new URLSearchParams({
    query,
    limit: String(RESULT_LIMIT),
    fields: "paperId,title,abstract,year,authors,url,externalIds,isOpenAccess"
  }).toString();
  const payload = await requestJson(endpoint, fetchImpl);
  if (!Array.isArray(payload?.data)) throw new Error("invalid-semantic-scholar-response");
  return payload.data.slice(0, RESULT_LIMIT).flatMap(item => {
    const title = safeText(item?.title, 300);
    const paperId = typeof item?.paperId === "string" ? safeText(item.paperId, 100) : "";
    const url = safeHttpsUrl(item?.url, ["www.semanticscholar.org", "semanticscholar.org"]);
    if (!title || !paperId || !url) return [];
    const year = Number.isInteger(item?.year) && item.year >= 1800 && item.year <= 2200
      ? String(item.year) : null;
    const doiValue = typeof item?.externalIds?.DOI === "string"
      ? item.externalIds.DOI.trim().toLowerCase() : "";
    const doi = /^10\.\d{4,9}\/[^\s]+$/.test(doiValue) ? doiValue : null;
    return [{
      provider: "Semantic Scholar",
      providerRecordId: paperId,
      title,
      authors: Array.isArray(item?.authors)
        ? item.authors.slice(0, 8).map(author => safeText(author?.name, 120)).filter(Boolean)
        : [],
      publicationDate: year,
      doi,
      url,
      sourceType: "scholarly-metadata",
      abstract: safeText(item?.abstract, 2400),
      freshness: freshness(year, now),
      ...(typeof item?.isOpenAccess === "boolean" ? { openAccess: item.isOpenAccess } : {})
    }];
  });
}

async function searchArxiv(query, fetchImpl, minIntervalMs, now) {
  await waitForArxivSlot(minIntervalMs);
  const endpoint = new URL("https://export.arxiv.org/api/query");
  endpoint.search = new URLSearchParams({
    search_query: "all:" + query,
    start: "0",
    max_results: String(RESULT_LIMIT),
    sortBy: "relevance",
    sortOrder: "descending"
  }).toString();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetchImpl(endpoint, {
      method: "GET",
      headers: { accept: "application/atom+xml", "user-agent": USER_AGENT },
      redirect: "error",
      signal: controller.signal
    });
    if (!response.ok) throw new Error("upstream-error");
    const xml = await response.text();
    if (typeof xml !== "string" || !/<feed\b/i.test(xml)) {
      throw new Error("invalid-arxiv-response");
    }
    const entries = [...xml.matchAll(/<entry\b[^>]*>([\s\S]*?)<\/entry>/gi)];
    return entries.slice(0, RESULT_LIMIT).flatMap(([_, entry]) => {
      const title = atomText(entry, "title");
      const idValue = atomText(entry, "id");
      const published = atomText(entry, "published").slice(0, 10);
      const abstract = atomText(entry, "summary").slice(0, 2400);
      let arxivId = "";
      try {
        const idUrl = new URL(idValue);
        if (idUrl.hostname !== "arxiv.org" || !idUrl.pathname.startsWith("/abs/")) return [];
        arxivId = decodeURIComponent(idUrl.pathname.slice("/abs/".length));
      } catch {
        return [];
      }
      if (!title || !/^[A-Za-z0-9][A-Za-z0-9._/-]{0,80}$/.test(arxivId)) return [];
      const authors = [...entry.matchAll(/<author\b[^>]*>([\s\S]*?)<\/author>/gi)]
        .slice(0, 8)
        .map(([__, author]) => atomText(author, "name"))
        .filter(Boolean);
      const url = "https://arxiv.org/abs/" + arxivId.split("/").map(encodeURIComponent).join("/");
      return [{
        provider: "arXiv",
        providerRecordId: arxivId,
        title,
        authors,
        publicationDate: /^\d{4}-\d{2}-\d{2}$/.test(published) ? published : null,
        doi: null,
        url,
        sourceType: "scholarly-preprint-metadata",
        abstract,
        freshness: freshness(published, now)
      }];
    });
  } finally {
    clearTimeout(timeout);
  }
}

function deduplicateRecords(records) {
  const seen = new Set();
  return records.filter(record => {
    const normalizedTitle = record.title.normalize("NFKC").toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();
    const fallbackId = record.providerRecordId || record.provider || "unknown";
    const key = record.doi ? "doi:" + record.doi : normalizedTitle
      ? "title:" + normalizedTitle
      : "record:" + fallbackId;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export async function searchMultiSourceResearch(query, fetchImpl = fetch, now = new Date(), options = {}) {
  if (typeof query !== "string") throw new TypeError("Research query must be text.");
  const normalized = query.trim();
  if (!normalized || normalized.length > MAX_QUERY_CHARS) {
    throw new RangeError("Research query must contain 1 to 200 characters.");
  }

  const crossrefUrl = new URL("https://api.crossref.org/works");
  crossrefUrl.search = new URLSearchParams({
    "query.bibliographic": normalized, rows: String(RESULT_LIMIT),
    select: "DOI,title,author,published,created,type"
  }).toString();
  const openAlexUrl = new URL("https://api.openalex.org/works");
  openAlexUrl.search = new URLSearchParams({
    search: normalized, "per-page": String(RESULT_LIMIT),
    select: "id,doi,title,publication_date,authorships,primary_location,type"
  }).toString();

  const configuredInterval = Number.isInteger(options.arxivMinIntervalMs)
    ? Math.max(0, Math.min(options.arxivMinIntervalMs, 60000))
    : 3000;
  const providers = await Promise.all([
    requestJson(crossrefUrl, fetchImpl).then(data => ({ name: "Crossref", records: normalizeCrossref(data, now) }))
      .catch(error => ({ name: "Crossref", records: [], error: error.name === "AbortError" ? "timeout" : "unavailable" })),
    requestJson(openAlexUrl, fetchImpl).then(data => ({ name: "OpenAlex", records: normalizeOpenAlex(data, now) }))
      .catch(error => ({ name: "OpenAlex", records: [], error: error.name === "AbortError" ? "timeout" : "unavailable" })),
    searchArxiv(normalized, fetchImpl, configuredInterval, now)
      .then(records => ({ name: "arXiv", records }))
      .catch(error => ({ name: "arXiv", records: [], error: error.name === "AbortError" ? "timeout" : "unavailable" })),
    searchSemanticScholar(normalized, fetchImpl, now)
      .then(records => ({ name: "Semantic Scholar", records }))
      .catch(error => ({ name: "Semantic Scholar", records: [], error: error.name === "AbortError" ? "timeout" : "unavailable" }))
  ]);

  const records = providers.flatMap(provider => provider.records);
  if (!providers.some(provider => provider.records.length)) {
    throw new Error("No scholarly research provider returned verifiable metadata.");
  }
  const unique = deduplicateRecords(records);
  const providerStatus = providers.map(({ name, records: found, error }) => ({
    provider: name, status: error ? error : found.length ? "ok" : "no-results", resultCount: found.length
  }));
  return {
    query: normalized,
    researchType: "scholarly-discovery",
    providers: providerStatus,
    providerCountSucceeded: providers.filter(p => !p.error).length,
    results: unique,
    uniqueRecordCount: unique.length,
    freshnessSummary: {
      recent: unique.filter(r => r.freshness === "recent").length,
      older: unique.filter(r => r.freshness === "older").length,
      unknown: unique.filter(r => r.freshness === "unknown").length,
      futureDated: unique.filter(r => r.freshness === "future-dated").length
    },
    limitations: [
      "These providers supply scholarly metadata, not proof that a paper's claims are correct.",
      "arXiv and Semantic Scholar abstracts may be included as retrieval context; this connector does not add source material to training data, and it does not retrieve full text.",
      "Matching DOI records are deduplicated and are not counted as independent corroboration."
    ]
  };
}
