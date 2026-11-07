const REAL_TIME_INTENT = /\b(latest|current|right now|today|tonight|yesterday|this week|this month|live|score|scores|fixture|fixtures|lineup|line-up|standings|league table|injur(?:y|ies)|transfer news|breaking news|weather|forecast|stock price|share price|exchange rate|traffic|open now)\b/i;
const NON_NEWS_LIVE_INTENT = /\b(weather|forecast|stock price|share price|exchange rate|traffic|open now)\b/i;
const LIVE_SCORE_INTENT = /\b(live\\s+(?:match\\s+)?scores?|current\\s+scores?|scores?\\s+(?:right now|currently|at the moment)|match clock|in-play score)\b/i;
const CREATIVE_OR_PERSONAL_INTENT = /\b(write (?:me )?(?:a|an|the)|compose|poem|short story|fiction|brainstorm|what do you think|your opinion|my personal|help me decide|roleplay|role-play)\b/i;
const FACTUAL_INTENT = /\b(who|what|where|when|which|why|how|tell me about|information about|history of|facts about|explain|describe|define|meaning of|overview of)\b/i;
const SUBJECT_HINT = /\b(team|club|city|country|company|university|person|organization|organisation|brand|scientist|artist|leader|historical figure)\b/i;
const NAMED_ENTITY = /(?:\b[A-Z][\p{L}\p{M}.'’-]*\s+){1,3}\b[A-Z][\p{L}\p{M}.'’-]*\b/u;

const ENCYCLOPEDIA_HOSTS = new Set(["en.wikipedia.org"]);
const SCHOLARLY_HOSTS = new Set([
  "doi.org",
  "openalex.org",
  "www.ncbi.nlm.nih.gov",
  "arxiv.org"
]);
const NEWS_HOSTS = new Set(["news.google.com"]);
const MAX_CONTEXT_CHARS = 3800;
const MAX_SOURCES = 5;


export function isUnsupportedLiveDataIntent(query, mode = "general") {
  if (mode !== "general" || typeof query !== "string") return false;
  const normalized = query.trim();
  if (normalized.length < 8 || normalized.length > 8000) return false;
  return NON_NEWS_LIVE_INTENT.test(normalized) || LIVE_SCORE_INTENT.test(normalized);
}

export function isTimeSensitiveResearchIntent(query, mode = "general") {
  if (mode !== "general" || typeof query !== "string") return false;
  const normalized = query.trim();
  if (normalized.length < 8 || normalized.length > 8000) return false;
  if (CREATIVE_OR_PERSONAL_INTENT.test(normalized) || NON_NEWS_LIVE_INTENT.test(normalized)) return false;
  return REAL_TIME_INTENT.test(normalized) ||
    /\b(recent news|latest news|last match|last game|next match|how did .* play|performance this season)\b/i.test(normalized);
}

export function shouldAutoResearch(query, mode = "general") {
  if (mode !== "general" || typeof query !== "string") return false;
  const normalized = query.trim();
  if (normalized.length < 10 || normalized.length > 8000) return false;
  if (REAL_TIME_INTENT.test(normalized) || CREATIVE_OR_PERSONAL_INTENT.test(normalized)) return false;
  if (FACTUAL_INTENT.test(normalized)) return true;
  return SUBJECT_HINT.test(normalized) && NAMED_ENTITY.test(normalized);
}

function cleanResearchText(value, maxLength) {
  if (typeof value !== "string") return "";
  return value
    .replace(/<[^>]*>/g, " ")
    .replace(/[\u0000-\u001f\u007f]/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, maxLength);
}

function safeResearchUrl(value, allowedHosts) {
  try {
    const url = new URL(value);
    if (url.protocol !== "https:" || !allowedHosts.has(url.hostname) ||
        url.username || url.password || url.port) return null;
    return url.href;
  } catch {
    return null;
  }
}

function encyclopediaCandidates(research) {
  const results = Array.isArray(research?.results) ? research.results : [];
  return results.slice(0, MAX_SOURCES).flatMap(result => {
    const title = cleanResearchText(result?.title, 120);
    const excerpt = cleanResearchText(result?.extract || result?.snippet, 620);
    const url = safeResearchUrl(result?.url, ENCYCLOPEDIA_HOSTS);
    if (!title || !excerpt || !url) return [];
    return [{ kind: "encyclopedia", title, excerpt, url }];
  });
}

function scholarlyCandidates(research) {
  const results = Array.isArray(research?.scholarlyResearch?.results)
    ? research.scholarlyResearch.results
    : [];
  return results.slice(0, MAX_SOURCES).flatMap(result => {
    const title = cleanResearchText(result?.title, 240);
    const provider = result?.provider === "Crossref" || result?.provider === "OpenAlex"
      ? result.provider
      : "";
    const url = safeResearchUrl(result?.url, SCHOLARLY_HOSTS);
    if (!title || !provider || !url) return [];
    const authors = Array.isArray(result?.authors)
      ? result.authors.slice(0, 3).map(author => cleanResearchText(author, 90)).filter(Boolean)
      : [];
    const publicationDate = typeof result?.publicationDate === "string" &&
      /^\d{4}(?:-\d{2}(?:-\d{2})?)?$/.test(result.publicationDate)
      ? result.publicationDate
      : "not supplied";
    return [{
      kind: "scholarly-metadata",
      provider,
      title,
      authors,
      publicationDate,
      url
    }];
  });
}

export function buildResearchContext(research) {
  const encyclopedia = encyclopediaCandidates(research);
  const scholarly = scholarlyCandidates(research);
  // When both kinds exist, keep a balanced evidence bundle rather than allowing
  // encyclopedia results to crowd out the bibliographic records.
  const selected = [
    ...encyclopedia.slice(0, scholarly.length ? 3 : MAX_SOURCES),
    ...scholarly.slice(0, encyclopedia.length ? 2 : MAX_SOURCES)
  ].slice(0, MAX_SOURCES);

  const sources = [];
  const blocks = [];
  const preamble = "Retrieved evidence bundle. Encyclopedia excerpts are short secondary reference material and may be incomplete or outdated. Scholarly entries contain bibliographic metadata only; they do not include abstracts or full text and do not establish a paper's methods, results, or conclusions. Never infer findings from a paper title or treat multiple database records for one DOI as independent corroboration. Use only the supplied excerpt for factual support, state evidence limits, and do not present this bundle as live web search.";
  let length = preamble.length;
  for (const item of selected) {
    const block = item.kind === "encyclopedia"
      ? [
          "ENCYCLOPEDIA EXCERPT (secondary reference, not live reporting): " + item.title,
          "URL: " + item.url,
          "EXCERPT: " + item.excerpt
        ].join("\n")
      : [
          "SCHOLARLY PUBLICATION METADATA ONLY (not an abstract, full text, or verified finding): " + item.title,
          "PROVIDER: " + item.provider,
          "AUTHORS: " + (item.authors.length ? item.authors.join("; ") : "not supplied"),
          "PUBLICATION DATE: " + item.publicationDate,
          "URL: " + item.url
        ].join("\n");
    if (length + block.length + 2 > MAX_CONTEXT_CHARS) continue;
    sources.push({
      kind: item.kind,
      title: item.title,
      url: item.url,
      ...(item.provider ? { provider: item.provider } : {})
    });
    blocks.push(block);
    length += block.length + 2;
  }

  if (!blocks.length) return { context: "", sources: [] };
  return {
    context: [preamble, ...blocks].join("\n\n"),
    sources
  };
}

function recentNewsCandidates(research, now) {
  const results = Array.isArray(research?.results) ? research.results : [];
  const nowMs = now.getTime();
  return results.slice(0, 5).flatMap(result => {
    const title = cleanResearchText(result?.title, 240);
    const publisher = cleanResearchText(result?.publisher, 100);
    const excerpt = cleanResearchText(result?.excerpt, 600);
    const url = safeResearchUrl(result?.url, NEWS_HOSTS);
    const timestamp = typeof result?.publishedAt === "string" ? Date.parse(result.publishedAt) : NaN;
    if (!title || !url || !Number.isFinite(timestamp)) return [];
    const age = nowMs - timestamp;
    if (age < -5 * 60 * 1000 || age > 8 * 24 * 60 * 60 * 1000) return [];
    return [{
      kind: "news",
      title,
      publisher: publisher || "Publisher not supplied",
      excerpt,
      publishedAt: new Date(timestamp).toISOString(),
      url
    }];
  });
}

export function buildLiveNewsContext(research, now = new Date()) {
  const candidates = recentNewsCandidates(research, now);
  if (!candidates.length) return { context: "", sources: [] };

  const retrievedAt = typeof research?.retrievedAt === "string" &&
    Number.isFinite(Date.parse(research.retrievedAt))
    ? new Date(research.retrievedAt).toISOString()
    : now.toISOString();
  const preamble = [
    "RECENT NEWS SEARCH RESULTS (retrieved at " + retrievedAt + " UTC).",
    "These are recent news headlines and short feed excerpts, not a guaranteed live news feed, live match clock, or official scoreboard.",
    "Use the publication timestamps to describe how recent each report is. Do not claim a report is live unless its contents explicitly establish that.",
    "A headline alone does not verify all article details. Do not invent events, scores, goals, causes, statistics, or quotes not present in the supplied text.",
    "Cite the exact Google News article links supplied below. If the headlines do not answer the question, say what remains unverified."
  ].join(" ");
  const blocks = [];
  const sources = [];
  let length = preamble.length;
  for (const item of candidates) {
    const block = [
      "HEADLINE: " + item.title,
      "PUBLISHER: " + item.publisher,
      "PUBLISHED AT (UTC): " + item.publishedAt,
      "ARTICLE LINK: " + item.url,
      ...(item.excerpt ? ["FEED EXCERPT (may be incomplete): " + item.excerpt] : [])
    ].join("\n");
    if (length + block.length + 2 > 3800) continue;
    blocks.push(block);
    sources.push({
      kind: "news",
      title: item.title,
      publisher: item.publisher,
      publishedAt: item.publishedAt,
      url: item.url
    });
    length += block.length + 2;
  }
  if (!blocks.length) return { context: "", sources: [] };
  return { context: [preamble, ...blocks].join("\n\n"), sources };
}
