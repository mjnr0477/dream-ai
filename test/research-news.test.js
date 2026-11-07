import test from "node:test";
import assert from "node:assert/strict";
import { searchRecentNews } from "../lib/research-news.js";

const NOW = new Date("2026-11-08T12:00:00Z");
const VALID_LINK = "https://news.google.com/rss/articles/CBMiExample";

function rssItem({ title, date, link = VALID_LINK, publisher = "Example News", description = "" }) {
  return [
    "<item>",
    "<title><![CDATA[" + title + "]]></title>",
    "<link>" + link + "</link>",
    "<pubDate>" + date + "</pubDate>",
    '<source url="https://example.com">' + publisher + "</source>",
    "<description><![CDATA[" + description + "]]></description>",
    "</item>"
  ].join("");
}

function xmlResponse(items, status = 200) {
  return new Response("<rss><channel>" + items.join("") + "</channel></rss>", {
    status,
    headers: { "content-type": "application/rss+xml" }
  });
}

test("recent-news search queries the regional RSS feed and returns timestamped sources", async () => {
  let requestedUrl;
  let requestOptions;
  const result = await searchRecentNews(
    "latest Manchester City performance",
    async (url, options) => {
      requestedUrl = new URL(url);
      requestOptions = options;
      return xmlResponse([rssItem({
        title: "Manchester City lose 3-1 & need a response",
        date: "Sun, 08 Nov 2026 10:30:00 GMT",
        description: "<b>Match report</b> with recent details &amp; reaction."
      })]);
    },
    NOW
  );

  assert.equal(requestedUrl.origin, "https://news.google.com");
  assert.equal(requestedUrl.pathname, "/rss/search");
  assert.equal(requestedUrl.searchParams.get("q"), "latest Manchester City performance when:7d");
  assert.equal(requestedUrl.searchParams.get("hl"), "en-KE");
  assert.equal(requestedUrl.searchParams.get("gl"), "KE");
  assert.equal(requestOptions.method, "GET");
  assert.equal(requestOptions.redirect, "error");
  assert.equal(result.provider, "Google News RSS");
  assert.equal(result.window, "7d");
  assert.equal(result.results.length, 1);
  assert.equal(result.results[0].title, "Manchester City lose 3-1 & need a response");
  assert.equal(result.results[0].publisher, "Example News");
  assert.equal(result.results[0].publishedAt, "2026-11-08T10:30:00.000Z");
  assert.match(result.results[0].excerpt, /Match report with recent details & reaction/);
  assert.ok(result.limitations.some(note => /not a guaranteed live-news service or official live-score feed/.test(note)));
});

test("today and live queries restrict the RSS search window to one day", async () => {
  let requestedUrl;
  const result = await searchRecentNews("What is the live score today?", async url => {
    requestedUrl = new URL(url);
    return xmlResponse([rssItem({
      title: "Recent match coverage",
      date: "Sun, 08 Nov 2026 11:00:00 GMT"
    })]);
  }, NOW);
  assert.equal(requestedUrl.searchParams.get("q"), "What is the live score today? when:1d");
  assert.equal(result.window, "1d");
});

test("rejects unsafe article URLs and filters stale, future and duplicate records", async () => {
  const duplicate = rssItem({
    title: "Same story",
    date: "Sun, 08 Nov 2026 11:00:00 GMT"
  });
  const result = await searchRecentNews("today Manchester City news", async () => xmlResponse([
    duplicate,
    duplicate,
    rssItem({
      title: "Unsafe destination",
      date: "Sun, 08 Nov 2026 10:00:00 GMT",
      link: "https://attacker.example/story"
    }),
    rssItem({
      title: "Old story",
      date: "Sun, 01 Nov 2026 10:00:00 GMT",
      link: "https://news.google.com/rss/articles/CBMiOld"
    }),
    rssItem({
      title: "Future story",
      date: "Mon, 09 Nov 2026 12:00:00 GMT",
      link: "https://news.google.com/rss/articles/CBMiFuture"
    })
  ]), NOW);

  assert.equal(result.results.length, 1);
  assert.equal(result.results[0].title, "Same story");
});

test("limits returned records to five", async () => {
  const items = Array.from({ length: 8 }, (_, index) => rssItem({
    title: "Headline " + index,
    date: "Sun, 08 Nov 2026 " + String(11 - index).padStart(2, "0") + ":00:00 GMT",
    link: "https://news.google.com/rss/articles/CBMi" + index
  }));
  const result = await searchRecentNews("latest world news", async () => xmlResponse(items), NOW);
  assert.equal(result.results.length, 5);
});

test("rejects empty and oversized queries before upstream access", async () => {
  let called = false;
  const fetchImpl = async () => {
    called = true;
    return xmlResponse([]);
  };
  await assert.rejects(searchRecentNews(" ", fetchImpl, NOW), /1 to 200 characters/);
  await assert.rejects(searchRecentNews("x".repeat(201), fetchImpl, NOW), /1 to 200 characters/);
  assert.equal(called, false);
});

test("fails safely on upstream errors and oversized feeds", async () => {
  await assert.rejects(
    searchRecentNews("latest news", async () => xmlResponse([], 503), NOW),
    /Recent news source returned an error/
  );
  await assert.rejects(
    searchRecentNews("latest news", async () => new Response("x".repeat(512 * 1024 + 1), { status: 200 }), NOW),
    /invalid or oversized feed/
  );
});
