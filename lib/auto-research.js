const REAL_TIME_INTENT = /\b(latest|current|right now|today|tonight|yesterday|this week|this month|live|score|scores|fixture|fixtures|lineup|line-up|standings|league table|injur(?:y|ies)|transfer news|breaking news|weather|forecast|stock price|share price|exchange rate|traffic|open now)\b/i;
const CREATIVE_OR_PERSONAL_INTENT = /\b(write (?:me )?(?:a|an|the)|compose|poem|short story|fiction|brainstorm|what do you think|your opinion|my personal|help me decide|roleplay|role-play)\b/i;
const FACTUAL_INTENT = /\b(who|what|where|when|which|why|how|tell me about|information about|history of|facts about|explain|describe|define|meaning of|overview of)\b/i;
const SUBJECT_HINT = /\b(team|club|city|country|company|university|person|organization|organisation|brand|scientist|artist|leader|historical figure)\b/i;
const NAMED_ENTITY = /(?:\b[A-Z][\p{L}\p{M}.'’-]*\s+){1,3}\b[A-Z][\p{L}\p{M}.'’-]*\b/u;

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
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, maxLength);
}

export function buildResearchContext(research) {
  const results = Array.isArray(research?.results) ? research.results : [];
  const sources = [];
  const blocks = [];
  let length = 0;
  const maximum = 3800;

  for (const result of results.slice(0, 5)) {
    const title = cleanResearchText(result?.title, 120);
    const excerpt = cleanResearchText(result?.extract || result?.snippet, 620);
    let url;
    try {
      url = new URL(result?.url);
      if (url.protocol !== "https:" || url.hostname !== "en.wikipedia.org") continue;
    } catch { continue; }
    if (!title || !excerpt) continue;

    const source = { title, url: url.href };
    const block = ["SOURCE: " + title, "URL: " + url.href, "EXCERPT: " + excerpt].join("\n");
    if (length + block.length > maximum) break;
    sources.push(source);
    blocks.push(block);
    length += block.length;
  }

  if (!blocks.length) return { context: "", sources: [] };
  return {
    context: ["Retrieved public encyclopedia excerpts. They may be incomplete or outdated and are not live reporting.", ...blocks].join("\n\n"),
    sources
  };
}
