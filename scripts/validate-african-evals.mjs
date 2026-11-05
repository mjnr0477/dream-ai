import { readFile } from "node:fs/promises";

const path = new URL("../evals/african-languages-v1.jsonl", import.meta.url);
const lines = (await readFile(path, "utf8")).split(/\r?\n/).filter(Boolean);
const allowed = new Set(["sw", "yo", "ha", "am", "ig", "so", "rw", "lg", "zu", "xh"]);
const ids = new Set();
const counts = new Map();

for (const [index, line] of lines.entries()) {
  let item;
  try {
    item = JSON.parse(line);
  } catch {
    throw new Error("Invalid JSON on African-language evaluation line " + (index + 1));
  }
  if (
    !item ||
    typeof item.id !== "string" ||
    !item.id.trim() ||
    typeof item.category !== "string" ||
    !["multilingual-math", "multilingual-explanation"].includes(item.category) ||
    typeof item.language !== "string" ||
    !allowed.has(item.language) ||
    typeof item.language_name !== "string" ||
    typeof item.script !== "string" ||
    typeof item.prompt !== "string" ||
    !item.prompt.trim() ||
    !Array.isArray(item.rubric) ||
    item.rubric.length < 2 ||
    item.rubric.some(point => typeof point !== "string" || !point.trim()) ||
    item.review_status !== "pending_native_review"
  ) {
    throw new Error("Invalid or unreviewed African-language evaluation case on line " + (index + 1));
  }
  if (ids.has(item.id)) throw new Error("Duplicate African-language evaluation id: " + item.id);
  ids.add(item.id);
  counts.set(item.language, (counts.get(item.language) || 0) + 1);
}
if (lines.length !== 20 || counts.size !== allowed.size ||
    [...counts.values()].some(count => count !== 2)) {
  throw new Error("Expected exactly two cases for each of the ten declared African languages.");
}
process.stdout.write("Validated " + lines.length + " African-language evaluation cases across " + counts.size + " languages. Native-speaker review is still required.\n");
