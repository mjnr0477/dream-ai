const MAX_QUERY_CHARS = 200;
const SEARCH_LIMIT = 5;

function cleanSnippet(value) {
  return value
    .replace(/<[^>]*>/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 500);
}

export async function searchPublicKnowledge(query, fetchImpl = fetch) {
  if (typeof query !== "string") {
    throw new TypeError("Research query must be text.");
  }
  const normalized = query.trim();
  if (!normalized || normalized.length > MAX_QUERY_CHARS) {
    throw new RangeError("Research query must contain 1 to 200 characters.");
  }

  const endpoint = new URL("https://en.wikipedia.org/w/api.php");
  endpoint.search = new URLSearchParams({
    action: "query",
    list: "search",
    srsearch: normalized,
    srlimit: String(SEARCH_LIMIT),
    format: "json",
    utf8: "1"
  }).toString();

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 7000);
  try {
    const response = await fetchImpl(endpoint, {
      method: "GET",
      headers: {
        accept: "application/json",
        "user-agent": "DREAM-AI/0.3 (public knowledge search)"
      },
      redirect: "error",
      signal: controller.signal
    });
    if (!response.ok) {
      throw new Error("The public research source returned an error.");
    }
    const payload = await response.json();
    const entries = payload?.query?.search;
    if (!Array.isArray(entries)) {
      throw new Error("The public research source returned an invalid response.");
    }

    const results = entries.slice(0, SEARCH_LIMIT).flatMap(entry => {
      if (
        !entry ||
        typeof entry.title !== "string" ||
        typeof entry.snippet !== "string"
      ) return [];
      const title = entry.title.trim();
      if (!title) return [];
      return [{
        title,
        snippet: cleanSnippet(entry.snippet),
        url: "https://en.wikipedia.org/wiki/" +
          encodeURIComponent(title.replace(/ /g, "_"))
      }];
    });

    return {
      provider: "Wikipedia",
      query: normalized,
      results
    };
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("Public research search timed out.");
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}
