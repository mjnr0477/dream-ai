import { readFile } from "node:fs/promises";

const path = new URL("../evals/core-v1.jsonl", import.meta.url);
const raw = await readFile(path, "utf8");
const rows = raw.split(/\r?\n/).filter(Boolean);
const ids = new Set();

for (const [index, line] of rows.entries()) {
  let item;
  try {
    item = JSON.parse(line);
  } catch {
    throw new Error("Invalid JSON on eval line " + (index + 1));
  }
  if (
    !item ||
    typeof item.id !== "string" ||
    typeof item.category !== "string" ||
    typeof item.prompt !== "string" ||
    !Array.isArray(item.rubric) ||
    item.rubric.length < 1 ||
    item.rubric.some(point => typeof point !== "string" || !point.trim()) ||
    (item.language !== undefined &&
      (typeof item.language !== "string" || !/^[A-Za-z]{2,3}(?:[-+][A-Za-z0-9]{2,8})*$/.test(item.language)))
  ) {
    throw new Error("Invalid evaluation case schema on line " + (index + 1));
  }
  if (ids.has(item.id)) throw new Error("Duplicate evaluation id: " + item.id);
  ids.add(item.id);
}

const multilingual = rows.map(line => JSON.parse(line)).filter(item => item.category === "multilingual");
if (multilingual.length < 10 || multilingual.some(item => !item.language)) {
  throw new Error("Multilingual evaluation requires at least 10 cases with language tags.");
}
const languageTags = new Set(multilingual.map(item => item.language));
if (languageTags.size !== multilingual.length) {
  throw new Error("Multilingual evaluation language tags must be unique.");
}
if (rows.length < 10) throw new Error("Core evaluation set must contain at least 10 cases.");
process.stdout.write("Validated " + rows.length + " DREAM evaluation cases, including " + languageTags.size + " multilingual cases.\n");
