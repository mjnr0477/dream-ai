import { readFile, writeFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

function aggregate(rows) {
  const scores = rows.map(row => row.humanScore).filter(score => score !== null);
  const latencies = rows.map(row => row.latencyMs).filter(Number.isFinite);
  return {
    cases: rows.length,
    scored: scores.length,
    unscored: rows.length - scores.length,
    averageHumanScore: scores.length
      ? Number((scores.reduce((sum, score) => sum + score, 0) / scores.length).toFixed(3))
      : null,
    averageLatencyMs: latencies.length
      ? Math.round(latencies.reduce((sum, value) => sum + value, 0) / latencies.length)
      : null
  };
}

export function summarizeRows(rows) {
  if (!Array.isArray(rows) || rows.length === 0) {
    throw new Error("Evaluation input must contain at least one result.");
  }
  const ids = new Set();
  for (const [index, row] of rows.entries()) {
    if (!row || typeof row.id !== "string" || !row.id.trim()) {
      throw new Error("Missing evaluation id at row " + (index + 1));
    }
    if (ids.has(row.id)) throw new Error("Duplicate evaluation id: " + row.id);
    ids.add(row.id);
    if (typeof row.category !== "string" || !row.category.trim()) {
      throw new Error("Missing category for " + row.id);
    }
    if (
      row.humanScore !== null &&
      (!Number.isInteger(row.humanScore) || row.humanScore < 0 || row.humanScore > 4)
    ) {
      throw new Error("humanScore must be null or an integer from 0 to 4 for " + row.id);
    }
    if (!Number.isFinite(row.latencyMs) || row.latencyMs < 0) {
      throw new Error("latencyMs must be a non-negative finite number for " + row.id);
    }
    if (row.language !== null && row.language !== undefined &&
        (typeof row.language !== "string" || !/^[a-z]{2,3}(?:-[A-Z]{2})?$/.test(row.language))) {
      throw new Error("Invalid language tag for " + row.id);
    }
  }

  const groupBy = key => {
    const groups = new Map();
    for (const row of rows) {
      const value = row[key];
      if (value === null || value === undefined || value === "") continue;
      if (!groups.has(value)) groups.set(value, []);
      groups.get(value).push(row);
    }
    return Object.fromEntries([...groups.entries()].sort(([a], [b]) => a.localeCompare(b))
      .map(([name, items]) => [name, aggregate(items)]));
  };

  return {
    schema: "dream-human-evaluation-summary-v1",
    scoring: {
      scale: "0-4",
      guidance: {
        "0": "Incorrect, unsafe or does not address the task",
        "1": "Major errors or mostly incomplete",
        "2": "Partly correct; important gaps remain",
        "3": "Correct and useful with minor gaps",
        "4": "Accurate, complete and well-reasoned"
      },
      judge: "human reviewer; no automatic score is inferred"
    },
    overall: aggregate(rows),
    byCategory: groupBy("category"),
    byLanguage: groupBy("language")
  };
}

async function main() {
  const input = process.argv[2];
  const output = process.argv[3];
  if (!input) {
    throw new Error("Usage: node scripts/summarize-evals.mjs <scored-results.jsonl> [summary.json]");
  }
  const lines = (await readFile(input, "utf8")).split(/\r?\n/).filter(Boolean);
  const rows = lines.map((line, index) => {
    try { return JSON.parse(line); }
    catch { throw new Error("Invalid JSON on result line " + (index + 1)); }
  });
  const summary = summarizeRows(rows);
  const json = JSON.stringify(summary, null, 2) + "\n";
  if (output) {
    const destination = resolve(output);
    await mkdir(dirname(destination), { recursive: true });
    await writeFile(destination, json, "utf8");
    process.stdout.write("Saved human evaluation summary to " + destination + "\n");
  } else {
    process.stdout.write(json);
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => {
    process.stderr.write(error.message + "\n");
    process.exitCode = 1;
  });
}
