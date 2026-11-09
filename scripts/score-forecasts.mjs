import { readFile, writeFile, stat, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { scoreForecasts } from "../lib/forecast-evaluation.js";

async function main() {
  const [inputArg, outputArg] = process.argv.slice(2);
  if (!inputArg) {
    throw new Error("Usage: npm run score:forecasts -- <resolved-forecasts.jsonl> [summary.json]");
  }
  const input = resolve(inputArg);
  const info = await stat(input);
  if (!info.isFile() || info.size > 5 * 1024 * 1024) {
    throw new Error("Forecast input must be a regular file no larger than 5 MiB.");
  }
  const lines = (await readFile(input, "utf8")).split(/\r?\n/).filter(line => line.trim());
  const records = lines.map((line, index) => {
    try { return JSON.parse(line); }
    catch { throw new Error("Invalid JSON in forecast record " + (index + 1) + "."); }
  });
  const report = scoreForecasts(records);
  const output = resolve(outputArg || input + ".summary.json");
  if (output === input) throw new Error("Summary output must not overwrite the input forecast records.");
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, JSON.stringify(report, null, 2) + "\n", "utf8");
  process.stdout.write("Scored " + report.overall.samples + " resolved forecasts.\n");
  process.stdout.write("Brier: " + report.overall.brierScore +
    " | Log loss: " + report.overall.logLoss +
    " | ECE: " + report.overall.expectedCalibrationError + "\n");
  process.stdout.write("Saved evaluation summary to " + output + "\n");
}

main().catch(error => {
  process.stderr.write(error.message + "\n");
  process.exitCode = 1;
});
