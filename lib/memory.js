const STOP_WORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
  "i", "in", "is", "it", "me", "my", "of", "on", "or", "our", "the",
  "this", "to", "we", "with", "you", "your"
]);

function tokens(value) {
  return new Set(
    String(value).toLowerCase().match(/[\p{L}\p{N}]{2,}/gu)?.filter(token => !STOP_WORDS.has(token)) || []
  );
}

function canonical(value) {
  return value.normalize("NFKC").toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();
}

function keyedNote(value) {
  const match = value.match(/^\s*([^:=]{2,60})\s*[:=]\s*(.+)$/u);
  return match ? canonical(match[1]) : null;
}

/**
 * Select useful lines from explicitly user-supplied memory.
 * Later values replace earlier values for simple "key: value" notes.
 * This is a deterministic selector, not a persistent store or identity boundary.
 */
export function selectMemoryContext(memoryContext, query = "", options = {}) {
  if (typeof memoryContext !== "string" || !memoryContext.trim()) return "";

  const maxChars = Number.isInteger(options.maxChars)
    ? Math.max(0, Math.min(options.maxChars, 3000))
    : 3000;
  const maxNotes = Number.isInteger(options.maxNotes)
    ? Math.max(0, Math.min(options.maxNotes, 30))
    : 30;
  if (!maxChars || !maxNotes) return "";

  const latestByKey = new Map();
  const seen = new Set();
  const notes = [];

  for (const raw of memoryContext.split(/\r?\n/u)) {
    const text = raw.trim();
    const normalized = canonical(text);
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);

    const key = keyedNote(text);
    if (key) {
      const previous = latestByKey.get(key);
      if (previous) previous.superseded = true;
      const note = { text, key, superseded: false, index: notes.length };
      latestByKey.set(key, note);
      notes.push(note);
    } else {
      notes.push({ text, key: null, superseded: false, index: notes.length });
    }
  }

  const queryTokens = tokens(query);
  const candidates = notes.filter(note => !note.superseded).map(note => {
    const noteTokens = tokens(note.text);
    let score = 0;
    for (const token of queryTokens) {
      if (noteTokens.has(token)) score += 1;
    }
    return { ...note, score };
  });

  candidates.sort((a, b) => b.score - a.score || a.index - b.index);
  const selected = [];
  let length = 0;
  for (const note of candidates) {
    if (selected.length >= maxNotes) break;
    const extra = note.text.length + (selected.length ? 1 : 0);
    if (length + extra > maxChars) continue;
    selected.push(note.text);
    length += extra;
  }
  return selected.join("\n");
}
