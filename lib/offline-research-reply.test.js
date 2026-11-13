import test from "node:test";
import assert from "node:assert/strict";
import { buildOfflineResearchReply } from "./offline-research-reply.js";

test("offline greeting clearly states available and unavailable capabilities", () => {
  const reply = buildOfflineResearchReply({ query: "Hello Dream AI" });
  assert.match(reply, /application is online/i);
  assert.match(reply, /language-model worker is currently offline/i);
  assert.match(reply, /public-source evidence/i);
});

test("offline factual response includes only safe source-backed records", () => {
  const reply = buildOfflineResearchReply({
    query: "What is photosynthesis?",
    sources: [
      {
        title: "Photosynthesis",
        excerpt: "Plants convert light energy into chemical energy.",
        provider: "Wikipedia",
        sourceType: "encyclopedia excerpt",
        url: "https://en.wikipedia.org/wiki/Photosynthesis"
      },
      {
        title: "Unsafe source",
        excerpt: "Do not include.",
        provider: "Public source",
        url: "javascript:alert(1)"
      },
      {
        title: "Credential source",
        excerpt: "Do not include.",
        provider: "Public source",
        url: "https://user:pass@example.com/article"
      }
    ]
  });
  assert.match(reply, /source-only mode/i);
  assert.match(reply, /Plants convert light energy/);
  assert.match(reply, /https:\/\/en\.wikipedia\.org\/wiki\/Photosynthesis/);
  assert.doesNotMatch(reply, /Unsafe source|Credential source|javascript:/);
  assert.match(reply, /not a model-generated synthesis/i);
});

test("offline factual response does not invent a source when retrieval is empty", () => {
  const reply = buildOfflineResearchReply({ query: "Explain a new discovery", sources: [] });
  assert.match(reply, /returned no usable source records/i);
  assert.match(reply, /I won't invent an answer/i);
});

test("offline source list is bounded and duplicate links are removed", () => {
  const sources = Array.from({ length: 10 }, (_, index) => ({
    title: "Source " + index,
    excerpt: "Evidence " + index,
    provider: "Public source",
    url: "https://example.org/" + index
  }));
  sources.push(sources[0]);
  const reply = buildOfflineResearchReply({ query: "A factual question", sources });
  assert.match(reply, /Source 5/);
  assert.doesNotMatch(reply, /Source 6/);
  assert.equal((reply.match(/https:\/\/example\.org\//g) || []).length, 6);
});
