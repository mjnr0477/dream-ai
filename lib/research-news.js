const MAX_QUERY_CHARS = 200;
const MAX_FEED_BYTES = 512 * 1024;
const RESULT_LIMIT = 5;
const REQUEST_TIMEOUT_MS = 8000;
const FEED_HOST = "news.google.com";
const EDITION = Object.freeze({ hl: "en-KE", gl: "KE", ceid: "KE:en" });

function decodeXml(value) {
  return value
    .replace(/^\s*<!\[CDATA\[/i, "")
    .replace(/\]\]>\s*$/i, "")
    .replace(/&amp;/gi, "&")
    .replace(/&quot;/gi, '"')
    .replace(/&apos;|&#39;/gi, "'")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&#(\d+);/g, (_match, digits) => {
      const codePoint = Number(digits);
      return Number.isInteger(codePoint) && codePoint >= 0 && codePoint <= 0x10ffff
        ? String.fromCodePoint(codePoint)
        : "";
    })
    .replace(/&#x([\da-f]+);/gi, (_match, digits) => {
      const codePoint = Number.parseInt(digits, 16);
      return Number.isInteger(codePoint) && codePoint >= 0 && codePoint <= 0x10ffff
        ? String.fromCodePoint(codePoint)
        : "";
    });
}

function readTag(xml, name) {
  const match = xml.match(new RegExp("<" + name + "(?:\\s[^>]*)?>([\\s\\S]*?)</" + name + ">", "i"));
  return match ? decodeXml(match[1]) : "";
}

function cleanText(value, limit = 700) {
  return decodeXml(String(value || ""))
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, " ")
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, " ")
    .replace(/<[^>]*>/g, " ")
    .replace(/[\u0000-\u001f\u007f]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, limit);
}

function safeGoogleNewsLink(value) {
  try {
    const url = new URL(value);
    if (
      url.protocol !== "https:" ||
      url.hostname !== FEED_HOST ||
      url.username ||
      url.password ||
      url.port ||
      !/^\/rss\/articles\//.test(url.pathname)
    ) return null;
    return url.href;
  } catch {
    return null;
  }
}

function getSearchWindow(query) {
  return /\b(today|tonight|live|right now|currently|this hour|this morning)\b/i.test(query)
    ? "1d"
    : "7d";
}

function parseItems(xml, now, window) {
  const maximumAgeMs = (window === "1d" ? 36 : 8 * 24) * 60 * 60 * 1000;
  const nowMs = now.getTime();
  const seen = new Set();
  const items = [];
  const matches = xml.matchAll(/<item\b[^>]*>([\s\S]*?)<\/item>/gi);
  for (const match of matches) {
    const item = match[1];
    const title = cleanText(readTag(item, "title"), 240);
    const url = safeGoogleNewsLink(cleanText(readTag(item, "link"), 1000));
    const rawDate = cleanText(readTag(item, "pubDate"), 120);
    const timestamp = Date.parse(rawDate);
    if (!title || !url || !Number.isFinite(timestamp)) continue;
    const age = nowMs - timestamp;
    if (age < -5 * 60 * 1000 || age > maximumAgeMs) continue;
    const key = title.toLocaleLowerCase() + "|" + url;
    if (seen.has(key)) continue;
    seen.add(key);
    const sourceMatch = item.match(/<source\b[^>]*>([\s\S]*?)<\/source>/i);
    const publisher = sourceMatch ? cleanText(sourceMatch[1], 100) : "";
    const excerpt = cleanText(readTag(item, "description"), 700);
    items.push({
      title,
      publisher,
      publishedAt: new Date(timestamp).toISOString(),
      url,
      excerpt: excerpt && excerpt !== title ? excerpt : ""
    });
    if (items.length >= RESULT_LIMIT) break;
  }
  return items;
}

export async function searchRecentNews(query, fetchImpl = fetch, now = new Date()) {
  if (typeof query !== "string") throw new TypeError("News query must be text.");
  const normalized = query.trim().replace(/\s+/g, " ");
  if (!normalized || normalized.length > MAX_QUERY_CHARS) {
    throw new RangeError("News query must contain 1 to 200 characters.");
  }

  const window = getSearchWindow(normalized);
  const endpoint = new URL("https://news.google.com/rss/search");
  endpoint.search = new URLSearchParams({
    q: normalized + " when:" + window,
    ...EDITION
  }).toString();

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetchImpl(endpoint, {
      method: "GET",
      headers: {
        accept: "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8",
        "user-agent": "DREAM-AI/0.3 (recent public news search)"
      },
      redirect: "error",
      signal: controller.signal
    });
    if (!response.ok) throw new Error("Recent news source returned an error.");
    const xml = await response.text();
    if (typeof xml !== "string" || !xml.trim() || Buffer.byteLength(xml, "utf8") > MAX_FEED_BYTES) {
      throw new Error("Recent news source returned an invalid or oversized feed.");
    }
    const results = parseItems(xml, now, window);
    return {
      provider: "Google News RSS",
      query: normalized,
      window,
      retrievedAt: now.toISOString(),
      results,
      limitations: [
        "This is a best-effort recent-news feed, not a guaranteed live-news service or official live-score feed.",
        "Headlines and publisher metadata do not prove the full details of a linked article; inspect the article before making detailed claims.",
        "The RSS endpoint is not a versioned API and has no availability or freshness guarantee."
      ]
    };
  } catch (error) {
    if (error?.name === "AbortError") throw new Error("Recent news search timed out.");
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}
