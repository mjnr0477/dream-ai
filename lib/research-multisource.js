const MAX_QUERY_CHARS = 200;
const RESULT_LIMIT = 5;
const REQUEST_TIMEOUT_MS = 7000;
const USER_AGENT = "DREAM-AI/0.3 (multi-source public research)";

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
    return url.protocol === "https:" && allowedHosts.includes(url.hostname) ? url.href : null;
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

export async function searchMultiSourceResearch(query, fetchImpl = fetch, now = new Date()) {
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

  const providers = await Promise.all([
    requestJson(crossrefUrl, fetchImpl).then(data => ({ name: "Crossref", records: normalizeCrossref(data, now) }))
      .catch(error => ({ name: "Crossref", records: [], error: error.name === "AbortError" ? "timeout" : "unavailable" })),
    requestJson(openAlexUrl, fetchImpl).then(data => ({ name: "OpenAlex", records: normalizeOpenAlex(data, now) }))
      .catch(error => ({ name: "OpenAlex", records: [], error: error.name === "AbortError" ? "timeout" : "unavailable" }))
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
      "Abstracts and full text are not included; inspect accessible papers before asserting findings.",
      "Matching DOI records are deduplicated and are not counted as independent corroboration."
    ]
  };
}
