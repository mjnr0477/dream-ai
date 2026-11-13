import test from "node:test";
import assert from "node:assert/strict";
import { buildForecastGuidance, classifyForecastDomain, isForecastRequest, buildResearchContext, buildResearchAttemptContext, shouldAutoResearch } from "../lib/auto-research.js";

test("automatically researches stable factual questions in normal chat", () => {
  assert.equal(shouldAutoResearch("Tell me about Manchester City"), true);
  assert.equal(shouldAutoResearch("Who is Ada Lovelace?"), true);
  assert.equal(shouldAutoResearch("What is photosynthesis?"), true);
  assert.equal(shouldAutoResearch("Manchester City team"), true);
});

test("routes current and time-sensitive questions to fresh public-source research", () => {
  assert.equal(shouldAutoResearch("What are the latest Manchester City results?"), true);
  assert.equal(shouldAutoResearch("What is the score today?"), true);
  assert.equal(shouldAutoResearch("Who is leading the league right now?"), true);
});

test("recognizes conversational news and transfer questions as current research requests", () => {
  assert.equal(shouldAutoResearch("I am hearing a lot of news around man city."), true);
  assert.equal(shouldAutoResearch("What is happening with Manchester City?"), true);
  assert.equal(shouldAutoResearch("Any transfer rumours around City?"), true);
  assert.equal(shouldAutoResearch("What's new around the club?"), true);
});

test("routes current-information questions in supported African and global languages to research", () => {
  const questions = [
    "Nani alishinda uchaguzi leo?",
    "Matokeo ya mechi yalikuwa nini jana?",
    "Kí ni tuntun loni nípa ìdìbò?",
    "Menene labarai yau game da zaben?",
    "ዛሬ ምን ዜና አለ?",
    "Kedu akụkọ ọhụrụ taa?",
    "Waa maxay wararka maanta?",
    "Ni ayahe makuru mashya uyu munsi?",
    "Amawulire ki agaliwo leero?",
    "Yiziphi izindaba zanamuhla?",
    "Zithini iindaba zanamhlanje?",
    "¿Qué noticias hay hoy?",
    "Quelles sont les nouvelles aujourd'hui ?",
    "ما آخر الأخبار اليوم؟",
    "आज की ताज़ा खबरें क्या हैं?",
    "What is happening with the election today?"
  ];

  for (const question of questions) {
    assert.equal(shouldAutoResearch(question), true, question);
  }
});

test("routes static factual questions in supported non-English languages to research", () => {
  for (const question of [
    "Ni nini maana ya demokrasia?",
    "Ṣàlàyé itan ti ìlú náà.",
    "Menene tarihin Najeriya?",
    "Kedu ihe bụ photosynthesis?",
    "¿Qué significa la fotosíntesis?",
    "Explique l'histoire du Kenya."
  ]) {
    assert.equal(shouldAutoResearch(question), true, question);
  }
});

test("detects CJK factual and current questions without requiring spaces", () => {
  assert.equal(shouldAutoResearch("北京今天有什么新闻？"), true);
  assert.equal(shouldAutoResearch("请解释人工智能。"), true);
  assert.equal(shouldAutoResearch("今日のニュースは何ですか？"), true);
  assert.equal(shouldAutoResearch("오늘 어떤 뉴스가 있나요?"), true);
  assert.equal(shouldAutoResearch("请写一个关于今天的故事。"), false);
});

test("localized creative prompts and greetings do not trigger unnecessary research", () => {
  assert.equal(shouldAutoResearch("Andika hadithi kuhusu leo"), false);
  assert.equal(shouldAutoResearch("Bhala indaba emayelana nezindaba zanamuhla"), false);
  assert.equal(shouldAutoResearch("Escribe una historia sobre mañana"), false);
  assert.equal(shouldAutoResearch("Habari yako?"), false);
  assert.equal(shouldAutoResearch("Sawubona"), false);
});

test("researches factual questions in general, coding and learning modes", () => {
  for (const mode of ["general", "coding", "learning", "research"]) {
    assert.equal(shouldAutoResearch("Who is Ada Lovelace?", mode), true, mode);
    assert.equal(shouldAutoResearch("How do JavaScript promises work?", mode), true, mode);
    assert.equal(shouldAutoResearch("What are the latest Python releases today?", mode), true, mode);
  }
});

test("does not research creative, personal, greeting or unsupported-mode requests", () => {
  assert.equal(shouldAutoResearch("Hi Dream AI"), false);
  assert.equal(shouldAutoResearch("Write me a short story about a moon robot"), false);
  assert.equal(shouldAutoResearch("What do you think about my plan?"), false);
  assert.equal(shouldAutoResearch("What is photosynthesis?", "unknown"), false);
});

test("Google's aggregate grounding synthesis is not attributed to an individual publisher", () => {
  const answer = "A grounded summary that draws on multiple search results.";
  const prepared = buildResearchContext({
    query: "latest technology update",
    platformResearch: {
      googleGroundedAnswer: answer,
      results: [
        { provider: "Google Search", title: "Publisher one", url: "https://www.bbc.com/news/one", snippet: "" },
        { provider: "Google Search", title: "Publisher two", url: "https://www.reuters.com/world/two", snippet: "" }
      ]
    }
  });
  assert.equal(prepared.context.split(answer).length - 1, 1);
  assert.match(prepared.context, /not a verbatim excerpt or a claim attributed to any single listed publisher/);
  assert.match(prepared.context, /https:\/\/www\.bbc\.com\/news\/one/);
  assert.match(prepared.context, /https:\/\/www\.reuters\.com\/world\/two/);
  assert.ok(prepared.sources.every(source => source.provider === "Google Search"));
});

test("research context accepts only bounded encyclopedia source URLs and text", () => {
  const prepared = buildResearchContext({ results: [
    { title: "Manchester City F.C.", url: "https://en.wikipedia.org/wiki/Manchester_City_F.C.", snippet: "A football club based in Manchester." },
    { title: "Unsafe destination", url: "https://attacker.example/redirect", extract: "Must be excluded." },
    { title: "Unsafe scheme", url: "javascript:alert(1)", extract: "Must be excluded." }
  ] });
  assert.equal(prepared.sources.length, 1);
  assert.equal(prepared.sources[0].title, "Manchester City F.C.");
  assert.match(prepared.context, /A football club based in Manchester/);
  assert.doesNotMatch(prepared.context, /attacker|javascript:/);
});

test("accepts only approved language-specific Wikipedia hosts", () => {
  const prepared = buildResearchContext({
    query: "Nini maana ya elimu?",
    results: [
      { title: "Elimu", url: "https://sw.wikipedia.org/wiki/Elimu", extract: "Elimu ni mchakato wa kujifunza." },
      { title: "Spoofed host", url: "https://sw.wikipedia.org.attacker.example/wiki/Elimu", extract: "Must be excluded." },
      { title: "Unsupported wiki language", url: "https://xx.wikipedia.org/wiki/Elimu", extract: "Must be excluded." }
    ]
  });
  assert.equal(prepared.sources.length, 1);
  assert.equal(prepared.sources[0].url, "https://sw.wikipedia.org/wiki/Elimu");
  assert.match(prepared.context, /Elimu ni mchakato/);
  assert.doesNotMatch(prepared.context, /Must be excluded/);
});

test("Google grounding retains secure publisher URLs returned by Google Search", () => {
  const prepared = buildResearchContext({
    query: "latest technology news",
    platformResearch: { results: [
      { provider: "Google Search", title: "Publisher report", url: "https://www.bbc.com/news/technology-123", snippet: "A sourced current report." },
      { provider: "Google Search", title: "Insecure link", url: "http://www.example.com/report", snippet: "Must be excluded." },
      { provider: "Google Search", title: "Credential link", url: "https://user:pass@example.com/report", snippet: "Must be excluded." }
    ] }
  });
  assert.equal(prepared.sources.length, 1);
  assert.equal(prepared.sources[0].provider, "Google Search");
  assert.equal(prepared.sources[0].url, "https://www.bbc.com/news/technology-123");
  assert.match(prepared.context, /A sourced current report/);
  assert.doesNotMatch(prepared.context, /Must be excluded/);
});

test("time-sensitive research context balances providers instead of letting Wikipedia crowd out current sources", () => {
  const prepared = buildResearchContext({
    query: "What are the latest developments today?",
    results: Array.from({ length: 5 }, (_, index) => ({
      title: "Encyclopedia source " + index,
      url: "https://en.wikipedia.org/wiki/Encyclopedia_" + index,
      extract: "A".repeat(500)
    })),
    globalResearch: {
      webResults: [{
        provider: "Brave Search LLM Context",
        title: "Independent web source",
        url: "https://www.example.org/current-report",
        snippet: "A relevant source from the web index."
      }],
      news: [{
        provider: "GDELT",
        title: "Recent publisher report",
        url: "https://www.bbc.com/news/current-report",
        domain: "bbc.com",
        seenAt: "2026-11-08T10:00:00Z"
      }]
    },
    platformResearch: { results: [{
      provider: "Google Search",
      title: "Google-grounded publisher",
      url: "https://www.reuters.com/world/current-report",
      snippet: "Google-grounded current reporting."
    }] }
  });
  const providers = new Set(prepared.sources.map(source => source.provider));
  assert.ok(providers.has("Google Search"));
  assert.ok(providers.has("Brave Search LLM Context"));
  assert.ok(providers.has("GDELT"));
  assert.ok(providers.has("Wikipedia"));
  assert.ok(prepared.sources.length <= 14);
});

test("Google News publisher excerpts are preserved in automatic answer context", () => {
  const prepared = buildResearchContext({
    query: "news around Manchester City",
    platformResearch: {
      results: [{
        provider: "Google News RSS",
        title: "Manchester City update",
        url: "https://www.bbc.com/sport/football/example",
        snippet: "Publisher: BBC Sport — The report describes a squad update.",
        publisher: "BBC Sport",
        publishedAt: "2026-11-09T12:00:00Z"
      }]
    }
  });
  assert.equal(prepared.sources.length, 1);
  assert.match(prepared.context, /Publisher: BBC Sport/);
  assert.match(prepared.context, /report describes a squad update/);
  assert.match(prepared.context, /PUBLISHED AT: 2026-11-09T12:00:00Z/);
});

test("research context enforces a strict evidence size limit", () => {
  const prepared = buildResearchContext({ results: Array.from({ length: 5 }, (_, index) => ({
    title: "Source " + index,
    url: "https://en.wikipedia.org/wiki/Source_" + index,
    extract: "A".repeat(620)
  })) });
  assert.ok(prepared.context.length <= 4000);
  assert.ok(prepared.sources.length <= 5);
});


test("Stack Exchange evidence is accepted only from the Stack Overflow host", () => {
  const prepared = buildResearchContext({ platformResearch: { results: [
    {
      provider: "Stack Exchange",
      title: "A public programming answer",
      url: "https://stackoverflow.com/questions/123456/example",
      snippet: "A concise answer excerpt."
    },
    {
      provider: "Stack Exchange",
      title: "Untrusted host",
      url: "https://attacker.example/questions/123",
      snippet: "Must not be included."
    }
  ] } });
  assert.equal(prepared.sources.length, 1);
  assert.equal(prepared.sources[0].provider, "Stack Exchange");
  assert.match(prepared.context, /A concise answer excerpt/);
  assert.doesNotMatch(prepared.context, /attacker\.example/);
});

test("PubMed abstracts are accepted from the official PubMed host and cited as abstracts", () => {
  const prepared = buildResearchContext({
    query: "What does recent PubMed research say about malaria prevention?",
    platformResearch: {
      results: [
        {
          provider: "PubMed",
          title: "Malaria prevention study",
          url: "https://pubmed.ncbi.nlm.nih.gov/12345678/",
          snippet: "The abstract reports a reduction in infection in the study population.",
          sourceType: "pubmed-abstract",
          publishedAt: "2025"
        },
        {
          provider: "PubMed",
          title: "Untrusted PubMed lookalike",
          url: "https://pubmed.ncbi.nlm.nih.gov.attacker.example/123/",
          snippet: "Must be excluded."
        }
      ]
    }
  });
  assert.equal(prepared.sources.length, 1);
  assert.equal(prepared.sources[0].provider, "PubMed");
  assert.match(prepared.context, /abstract reports a reduction/);
  assert.match(prepared.context, /PubMed abstracts are summaries, not full-text articles/);
  assert.doesNotMatch(prepared.context, /attacker\\.example|Must be excluded/);
});


test("arXiv and Semantic Scholar abstracts enter context only through their trusted source hosts", () => {
  const prepared = buildResearchContext({
    scholarlyResearch: { results: [
      {
        provider: "arXiv",
        title: "A preprint about reliable AI",
        url: "https://arxiv.org/abs/2601.01234v1",
        publicationDate: "2026-01-12",
        abstract: "An abstract retrieved from the arXiv API."
      },
      {
        provider: "Semantic Scholar",
        title: "A peer-reviewed paper about verification",
        url: "https://www.semanticscholar.org/paper/Verification/0123456789abcdef",
        publicationDate: "2025",
        abstract: "An abstract retrieved from Semantic Scholar."
      },
      {
        provider: "arXiv",
        title: "Untrusted destination",
        url: "https://attacker.example/abs/2601.99999",
        abstract: "This must not enter context."
      }
    ] }
  });
  assert.equal(prepared.sources.length, 2);
  assert.ok(prepared.sources.some(source => source.provider === "arXiv"));
  assert.ok(prepared.sources.some(source => source.provider === "Semantic Scholar"));
  assert.match(prepared.context, /An abstract retrieved from the arXiv API/);
  assert.match(prepared.context, /An abstract retrieved from Semantic Scholar/);
  assert.doesNotMatch(prepared.context, /attacker\.example|This must not enter context/);
});


test("GitHub repository metadata enters coding research only from github.com", () => {
  const prepared = buildResearchContext({
    query: "Find a Python CSV library",
    platformResearch: { results: [
      {
        provider: "GitHub Public Repositories",
        title: "example/csv-parser",
        url: "https://github.com/example/csv-parser",
        snippet: "A small Python CSV parser.",
        sourceType: "public-repository-metadata"
      },
      {
        provider: "GitHub Public Repositories",
        title: "unsafe/repository",
        url: "https://attacker.example/unsafe",
        snippet: "Untrusted URL."
      }
    ] }
  });
  assert.equal(prepared.sources.length, 1);
  assert.equal(prepared.sources[0].provider, "GitHub Public Repositories");
  assert.match(prepared.context, /A small Python CSV parser/);
  assert.doesNotMatch(prepared.context, /attacker\.example|Untrusted URL/);
});


test("forecast requests across weather, sport, markets and general events route into fresh research", () => {
  const cases = [
    ["Will it rain in Bungoma tomorrow?", "weather"],
    ["Predict Arsenal vs Manchester City", "sports"],
    ["Could Bitcoin rise next week?", "markets"],
    ["What will happen in the next national election?", "general"]
  ];
  for (const [query, domain] of cases) {
    assert.equal(isForecastRequest(query), true, query);
    assert.equal(classifyForecastDomain(query), domain, query);
    assert.equal(shouldAutoResearch(query), true, query);
  }
});

test("implicit weather horizons activate forecasting without requiring the word forecast", () => {
  assert.equal(isForecastRequest("Weather in Bungoma tomorrow"), true);
  assert.equal(classifyForecastDomain("Weather in Bungoma tomorrow"), "weather");
  assert.equal(shouldAutoResearch("Weather in Bungoma tomorrow"), true);
  assert.equal(isForecastRequest("Weather in Bungoma for the next 5 days"), true);
  assert.equal(isForecastRequest("Bitcoin next week"), true);
  assert.equal(classifyForecastDomain("Bitcoin next week"), "markets");
  assert.equal(isForecastRequest("Arsenal vs Chelsea tomorrow"), true);
  assert.equal(classifyForecastDomain("Arsenal vs Chelsea tomorrow"), "sports");
  assert.equal(isForecastRequest("Explain how rain forms"), false);
});

test("forecast guidance rejects false precision and is domain-aware", () => {
  const weather = buildForecastGuidance("Forecast the weather in Bungoma tomorrow");
  assert.match(weather, /official meteorological observations/);
  assert.match(weather, /probability of precipitation/);
  const sports = buildForecastGuidance("Predict who will win the football match");
  assert.match(sports, /include the draw/);
  const markets = buildForecastGuidance("Predict the stock market next month");
  assert.match(markets, /timestamped price history/);
  assert.match(markets, /losses are possible/);
  const general = buildForecastGuidance("What will happen in the election?");
  assert.match(general, /historical base rate or reference class/);
  for (const guidance of [weather, sports, markets, general]) {
    assert.match(guidance, /Do not invent data/);
    assert.match(guidance, /calibrated forecasting model/);
  }
});

test("forecasting does not turn creative writing requests into research", () => {
  assert.equal(shouldAutoResearch("Write a short story about who will win the football match"), false);
  assert.equal(buildForecastGuidance("Write a short story about who will win the football match"), "");
});

test("forecast research context warns against deriving probabilities from headlines alone", () => {
  const prepared = buildResearchContext({
    query: "Predict whether Bitcoin will rise",
    platformResearch: {
      results: [{
        provider: "Google News RSS",
        title: "Crypto market update",
        url: "https://www.reuters.com/markets/example",
        snippet: "Publisher: Reuters — A market report."
      }]
    }
  });
  assert.match(prepared.context, /not itself a validated forecasting model/);
  assert.match(prepared.context, /Do not calculate precise event probabilities from retrieved headlines alone/);
});

test("automatic research status tells the model what sources actually returned", () => {
  const context = buildResearchAttemptContext({
    globalResearch: {
      providers: [
        { provider: "Wikidata", status: "no-results" },
        { provider: "GDELT", status: "unavailable" }
      ],
      news: []
    },
    platformResearch: {
      providers: [
        { provider: "Google News RSS", status: "no-results" },
        { provider: "Bluesky", status: "ok" }
      ],
      results: []
    },
    scholarlyResearch: {
      providers: [{ provider: "Crossref", status: "ok" }]
    }
  }, { sourceCount: 0 });

  assert.match(context, /workflow was attempted/);
  assert.match(context, /Usable, source-validated records passed to the model: 0/);
  assert.match(context, /GDELT: unavailable/);
  assert.match(context, /Google News RSS: no-results/);
  assert.match(context, /No usable source records were returned for this query/);
  assert.match(context, /Do not claim that DREAM has no real-time news access/);
  assert.match(context, /Do not send the user away to check BBC, Reuters/);
});

test("automatic research failure is described as a failed retrieval attempt, not permanent lack of access", () => {
  const context = buildResearchAttemptContext(null, { failed: true });

  assert.match(context, /workflow was attempted but failed/);
  assert.match(context, /retrieval attempt failed/);
  assert.match(context, /no real-time access in general/);
});

test("background-only sources are not presented as current news coverage", () => {
  const context = buildResearchAttemptContext({
    globalResearch: {
      providers: [{ provider: "GDELT", status: "no-results" }],
      news: []
    },
    platformResearch: {
      providers: [{ provider: "Google News RSS", status: "no-results" }],
      results: []
    },
    scholarlyResearch: {
      providers: [{ provider: "Crossref", status: "ok" }]
    }
  }, { sourceCount: 2 });

  assert.match(context, /Some public sources returned background or reference records/);
  assert.match(context, /no recent-news records were returned/);
  assert.match(context, /do not present them as current reporting/);
  assert.match(context, /Do not claim that DREAM has no real-time news access/);
});
