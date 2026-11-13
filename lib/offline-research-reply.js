const MAX_SOURCES = 6;
const MAX_TITLE = 180;
const MAX_EXCERPT = 420;

function clean(value, maxLength) {
  if (typeof value !== "string") return "";
  return value
    .replace(/<[^>]*>/g, " ")
    .replace(/[\u0000-\u001f\u007f]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, maxLength);
}

function safeUrl(value) {
  try {
    const url = new URL(value);
    if (url.protocol !== "https:" || url.username || url.password || url.port) return "";
    return url.href;
  } catch {
    return "";
  }
}

export function isOfflineGreetingQuery(query) {
  return typeof query === "string"
    && /^(?:hi|hello|hey|good morning|good afternoon|good evening|habari|mambo)(?:\\s+dream(?:\\s+ai)?)?[.!?\\s]*$/i.test(query.trim());
}

export function buildOfflineResearchReply({ query, sources = [], mode = "general" } = {}) {
  const question = clean(query, 500);
  if (isOfflineGreetingQuery(question)) {
    return "Hello. DREAM AI's application is online, but its language-model worker is currently offline. I can still retrieve public-source evidence for factual questions; original model-generated reasoning and coding are unavailable until compute is connected.";
  }

  const safeSources = [];
  const seen = new Set();
  for (const item of Array.isArray(sources) ? sources : []) {
    const title = clean(item?.title, MAX_TITLE);
    const url = safeUrl(item?.url);
    if (!title || !url || seen.has(url)) continue;
    seen.add(url);
    safeSources.push({
      title,
      url,
      excerpt: clean(item?.excerpt, MAX_EXCERPT),
      provider: clean(item?.provider, 80) || "Public source",
      sourceType: clean(item?.sourceType, 80),
      publishedAt: clean(item?.publishedAt || item?.seenAt || item?.indexedAt || "", 50)
    });
    if (safeSources.length >= MAX_SOURCES) break;
  }

  if (!safeSources.length) {
    return "DREAM AI is reachable, but its language-model worker is offline and public research returned no usable source records for this request. I won't invent an answer. Original reasoning, writing and code generation will resume when a permanent model runtime is connected.";
  }

  const lines = [
    "DREAM AI is operating in source-only mode because its language-model worker is offline. This is a retrieval digest, not a model-generated synthesis.",
    "",
    "Public-source records returned for: " + question,
    ""
  ];
  for (let index = 0; index < safeSources.length; index++) {
    const item = safeSources[index];
    const details = [item.provider, item.sourceType, item.publishedAt].filter(Boolean).join(" · ");
    lines.push((index + 1) + ". **" + item.title + "**");
    if (item.excerpt) lines.push("   " + item.excerpt);
    else lines.push("   The provider returned a title or metadata record but no usable excerpt.");
    lines.push("   " + details + " — " + item.url);
    lines.push("");
  }
  lines.push("These records are leads, snippets or abstracts according to their source type; they are not proof that the full underlying pages or papers were read. Compare the original sources before relying on a claim.");
  return lines.join("\n");
}
