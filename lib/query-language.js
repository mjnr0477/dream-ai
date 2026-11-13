const LATIN_LANGUAGE_PATTERNS = [
  ["sw", /(?:^|[^\\p{L}\\p{N}])(?:leo|jana|kesho|sasa hivi|habari|matokeo|nani|nini|wapi|lini|kwa nini|maana ya|historia ya|eleza)(?=$|[^\\p{L}\\p{N}])/iu],
  ["yo", /(?:^|[^\\p{L}\\p{N}])(?:loni|lọla|lọwọlọwọ|iroyin|kí ni|itan|báwo|ṣàlàyé|ẹ wo)(?=$|[^\\p{L}\\p{N}])/iu],
  ["ha", /(?:^|[^\\p{L}\\p{N}])(?:yau|jiya|gobe|yanzu|labarai|menene|tarihin|me ke faruwa|me ya sa)(?=$|[^\\p{L}\\p{N}])/iu],
  ["ig", /(?:^|[^\\p{L}\\p{N}])(?:taa|ụnyaahụ|echi|ugbu a|akụkọ|gịnị|kedu|kọwaa)(?=$|[^\\p{L}\\p{N}])/iu],
  ["so", /(?:^|[^\\p{L}\\p{N}])(?:maanta|shalay|berri|hadda|wararka|maxaa|taariikhda|waa maxay|sidee)(?=$|[^\\p{L}\\p{N}])/iu],
  ["rw", /(?:^|[^\\p{L}\\p{N}])(?:uyu munsi|amakuru|amateka|sobanura|ni ayahe|ni iki|ejo)(?=$|[^\\p{L}\\p{N}])/iu],
  ["lg", /(?:^|[^\\p{L}\\p{N}])(?:leero|eggulo|enkya|amawulire|kiki|lwaki|ani|ddi)(?=$|[^\\p{L}\\p{N}])/iu],
  ["zu", /(?:^|[^\\p{L}\\p{N}])(?:namuhla|izolo|kusasa|izindaba|umlando|kungani|ubani|kuphi|yiziphi)(?=$|[^\\p{L}\\p{N}])/iu],
  ["xh", /(?:^|[^\\p{L}\\p{N}])(?:namhlanje|ngomso|ngoku|iindaba|imbali|kutheni|ngubani|zithini|phi)(?=$|[^\\p{L}\\p{N}])/iu],
  ["fr", /(?:^|[^\\p{L}\\p{N}])(?:aujourd'hui|hier|demain|maintenant|actualités|nouvelles|résultats|météo|pourquoi|comment|explique|histoire de)(?=$|[^\\p{L}\\p{N}])/iu],
  ["es", /(?:^|[^\\p{L}\\p{N}])(?:hoy|ayer|mañana|ahora|noticias|resultados|clima|quién|dónde|cuándo|por qué|cómo|explica|historia de)(?=$|[^\\p{L}\\p{N}])/iu],
  ["pt", /(?:^|[^\\p{L}\\p{N}])(?:hoje|ontem|amanhã|agora|notícias|resultados|previsão do tempo|quem|onde|quando|por que|como|explique|história de)(?=$|[^\\p{L}\\p{N}])/iu]
];

export function detectQueryLanguage(query) {
  if (typeof query !== "string" || !query.trim()) return "en";
  const normalized = query.normalize("NFKC").trim();

  if (/[\\uac00-\\ud7af]/u.test(normalized)) return "ko";
  if (/[\\u3040-\\u30ff]/u.test(normalized)) return "ja";
  if (/[\\u0600-\\u06ff]/u.test(normalized)) return "ar";
  if (/[\\u0900-\\u097f]/u.test(normalized)) return "hi";
  if (/[\\u0980-\\u09ff]/u.test(normalized)) return "bn";
  if (/[\\u1200-\\u137f]/u.test(normalized)) return "am";
  if (/[\\u0400-\\u04ff]/u.test(normalized)) return "ru";
  if (/[\\u4e00-\\u9fff]/u.test(normalized)) return "zh";

  for (const [language, pattern] of LATIN_LANGUAGE_PATTERNS) {
    if (pattern.test(normalized)) return language;
  }
  return "en";
}
