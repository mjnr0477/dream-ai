const MAX_QUERY_CHARS = 200;
const SEARCH_LIMIT = 5;
const MAX_EXTRACT_CHARS = 350;

function cleanSnippet(value, maxLength = 500) {
  return value
    .replace(/<[^>]*>/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, maxLength);
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
        !Number.isSafeInteger(entry.pageid) ||
        entry.pageid < 1 ||
        typeof entry.title !== "string" ||
        typeof entry.snippet !== "string"
      ) return [];
      const title = entry.title.trim();
      if (!title) return [];
      return [{
        pageId: entry.pageid,
        title,
        snippet: cleanSnippet(entry.snippet),
        extract: null,
        extractTruncated: false,
        lastModified: null,
        url: "https://en.wikipedia.org/wiki/" +
          encodeURIComponent(title.replace(/ /g, "_"))
      }];
    });

    if (results.length) {
      const evidenceEndpoint = new URL("https://en.wikipedia.org/w/api.php");
      evidenceEndpoint.search = new URLSearchParams({
        action: "query",
        prop: "extracts|info",
        pageids: results.map(item => String(item.pageId)).join("|"),
        exintro: "1",
        explaintext: "1",
        exchars: "2400",
        inprop: "url",
        format: "json",
        utf8: "1"
      }).toString();

      const evidenceController = new AbortController();
      const evidenceTimeout = setTimeout(() => evidenceController.abort(), 7000);
      try {
        const evidenceResponse = await fetchImpl(evidenceEndpoint, {
          method: "GET",
          headers: {
            accept: "application/json",
            "user-agent": "DREAM-AI/0.3 (public knowledge research)"
          },
          redirect: "error",
          signal: evidenceController.signal
        });
        if (evidenceResponse.ok) {
          const evidencePayload = await evidenceResponse.json();
          const pages = evidencePayload?.query?.pages;
          if (pages && typeof pages === "object" && !Array.isArray(pages)) {
            const byId = new Map(Object.values(pages)
              .filter(page => page && Number.isSafeInteger(page.pageid))
              .map(page => [page.pageid, page]));
            for (const item of results) {
              const page = byId.get(item.pageId);
              if (!page) continue;
              if (typeof page.extract === "string") {
                item.extract = cleanSnippet(page.extract, MAX_EXTRACT_CHARS);
                item.extractTruncated = page.extract.length > MAX_EXTRACT_CHARS;
              }
              if (typeof page.touched === "string") item.lastModified = page.touched;
              if (typeof page.fullurl === "string") {
                try {
                  const sourceUrl = new URL(page.fullurl);
                  if (sourceUrl.origin === "https://en.wikipedia.org" &&
                      sourceUrl.protocol === "https:") item.url = sourceUrl.href;
                } catch {}
              }
            }
          }
        }
      } catch (error) {
        if (error.name === "AbortError") {
          // Keep the search results and mark article extracts as unavailable.
        }
      } finally {
        clearTimeout(evidenceTimeout);
      }
    }

    return {
      provider: "Wikipedia",
      query: normalized,
      evidenceLevel: results.some(item => item.extract)
        ? "encyclopedia-introduction"
        : "search-snippet",
      results: results.map(({ pageId, ...item }) => item)
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
