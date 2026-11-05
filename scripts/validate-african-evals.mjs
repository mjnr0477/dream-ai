import { readFile } from "node:fs/promises";

const path = new URL("../evals/african-languages-v1.jsonl", import.meta.url);
const lines = (await readFile(path, "utf8")).split(/\r?\n/).filter(Boolean);
const allowed = new Set(["sw", "yo", "ha", "am", "ig", "so", "rw", "lg", "zu", "xh"]);
const ids = new Set();
const counts = new Map();

function evaluateIntegerExpression(expression) {
  const tokens = expression.match(/\d+|[+-]/g) || [];
  if (tokens.join("") !== expression.replace(/\s+/g, "")) return null;
  if (tokens.length === 0 || tokens.length % 2 === 0 || !/^\d+$/.test(tokens[0])) return null;
  let value = Number(tokens[0]);
  for (let index = 1; index < tokens.length; index += 2) {
    const operator = tokens[index];
    const operand = Number(tokens[index + 1]);
    if (!Number.isSafeInteger(operand)) return null;
    value = operator === "+" ? value + operand : value - operand;
    if (!Number.isSafeInteger(value)) return null;
  }
  return value;
}

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
    item.review_status !== "pending_native_review" ||
    (item.category === "multilingual-math" &&
      (!Number.isSafeInteger(item.expected_answer) ||
       typeof item.expected_expression !== "string" ||
       !item.expected_expression.trim() ||
       evaluateIntegerExpression(item.expected_expression) !== item.expected_answer)) ||
    (item.category === "multilingual-explanation" &&
      (Object.hasOwn(item, "expected_answer") || Object.hasOwn(item, "expected_expression")))
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
