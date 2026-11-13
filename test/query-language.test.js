import test from "node:test";
import assert from "node:assert/strict";
import { detectQueryLanguage } from "../lib/query-language.js";

test("detects supported African language queries", () => {
  const cases = [
    ["Nani alishinda uchaguzi leo?", "sw"],
    ["Kí ni tuntun loni nípa ìdìbò?", "yo"],
    ["Menene labarai yau game da zaben?", "ha"],
    ["Kedu akụkọ ọhụrụ taa?", "ig"],
    ["Waa maxay wararka maanta?", "so"],
    ["Ni ayahe makuru mashya uyu munsi?", "rw"],
    ["Amawulire ki agaliwo leero?", "lg"],
    ["Yiziphi izindaba zanamuhla?", "zu"],
    ["Zithini iindaba zanamhlanje?", "xh"],
    ["ዛሬ ምን ዜና አለ?", "am"]
  ];
  for (const [query, expected] of cases) {
    assert.equal(detectQueryLanguage(query), expected, query);
  }
});

test("detects widely used global languages and writing systems", () => {
  const cases = [
    ["Quelles sont les nouvelles aujourd'hui ?", "fr"],
    ["¿Qué noticias hay hoy?", "es"],
    ["Quais são as notícias hoje?", "pt"],
    ["ما آخر الأخبار اليوم؟", "ar"],
    ["आज की ताज़ा खबरें क्या हैं?", "hi"],
    ["আজকের খবর কী?", "bn"],
    ["Что происходит сегодня?", "ru"],
    ["北京今天有什么新闻？", "zh"],
    ["今日のニュースは何ですか？", "ja"],
    ["오늘 어떤 뉴스가 있나요?", "ko"]
  ];
  for (const [query, expected] of cases) {
    assert.equal(detectQueryLanguage(query), expected, query);
  }
});

test("uses English for English or unclassified text and handles invalid input", () => {
  assert.equal(detectQueryLanguage("What is photosynthesis?"), "en");
  assert.equal(detectQueryLanguage("Who is Leo Messi?"), "en");
  assert.equal(detectQueryLanguage("How do I leave a comment on this page?"), "en");
  assert.equal(detectQueryLanguage(""), "en");
  assert.equal(detectQueryLanguage(null), "en");
});
