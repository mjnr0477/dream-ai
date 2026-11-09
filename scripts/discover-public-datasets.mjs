#!/usr/bin/env node
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { pathToFileURL } from "node:url";

const API_URL = "https://huggingface.co/api/datasets";
const REQUEST_TIMEOUT_MS = 12000;
const DEFAULT_LIMIT = 12;
const MAX_LIMIT = 25;

function cleanText(value, maxLength = 500) {
  return typeof value === "string"
    ? value.replace(/<[^>]*>/g, " ")
      .replace(/[\u0000-\u001f\u007f]/g, " ")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, maxLength)
    : "";
}

function declaredLicense(dataset) {
  const cardLicense = dataset?.cardData?.license;
  if (typeof cardLicense === "string" && cardLicense.trim()) {
    return { value: cleanText(cardLicense, 100), evidence: "dataset-card-metadata" };
  }
  const tags = Array.isArray(dataset?.tags) ? dataset.tags : [];
  const tag = tags.find(value => typeof value === "string" && /^license:/i.test(value));
  return tag
    ? { value: cleanText(tag.replace(/^license:/i, ""), 100), evidence: "hub-license-tag" }
    : { value: null, evidence: "not-declared" };
}

function normalizeCandidate(dataset) {
  const id = typeof dataset?.id === "string" ? dataset.id.trim() : "";
  if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(id) ||
      id.split("/").some(part => part === "." || part === "..")) return null;
  const privateFlag = dataset.private;
  const gatedFlag = dataset.gated;
  const isPrivate = privateFlag === true || privateFlag === "true";
  const isGated = gatedFlag !== undefined && gatedFlag !== null &&
    gatedFlag !== false && gatedFlag !== "false";
  if (isPrivate || isGated) return null;

  const license = declaredLicense(dataset);
  const tags = Array.isArray(dataset.tags)
    ? dataset.tags.filter(tag => typeof tag === "string").slice(0, 16).map(tag => cleanText(tag, 80))
    : [];
  const downloads = Number.isSafeInteger(dataset.downloads) && dataset.downloads >= 0
    ? dataset.downloads : null;
  const likes = Number.isSafeInteger(dataset.likes) && dataset.likes >= 0
    ? dataset.likes : null;

  return {
    id,
    title: id,
    url: "https://huggingface.co/datasets/" + id.split("/").map(encodeURIComponent).join("/"),
    description: cleanText(dataset.description || dataset.cardData?.description, 500),
    declaredLicense: license.value,
    licenseEvidence: license.evidence,
    downloads,
    likes,
    lastModified: cleanText(dataset.lastModified, 40) || null,
    tags,
    candidateStatus: "unreviewed",
    trainingApproved: false,
    review: {
      rights: "pending",
      privacy: "pending",
      quality: "pending"
    }
  };
}

export async function discoverPublicDatasets(query, {
  fetchImpl = fetch,
  limit = DEFAULT_LIMIT,
  now = new Date()
} = {}) {
  if (typeof query !== "string" || !query.trim() || query.trim().length > 120 ||
      /[\u0000-\u001f\u007f]/.test(query)) {
    throw new RangeError("Dataset search query must contain 1 to 120 printable characters.");
  }
  if (!Number.isSafeInteger(limit) || limit < 1 || limit > MAX_LIMIT) {
    throw new RangeError("Dataset discovery limit must be between 1 and 25.");
  }

  const endpoint = new URL(API_URL);
  endpoint.search = new URLSearchParams({
    search: query.trim(),
    limit: String(limit),
    full: "true"
  }).toString();

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetchImpl(endpoint, {
      method: "GET",
      headers: {
        accept: "application/json",
        "user-agent": "DREAM-AI/0.1 (public dataset metadata discovery)"
      },
      redirect: "error",
      signal: controller.signal
    });
    if (!response.ok) throw new Error("Hugging Face dataset discovery returned HTTP " + response.status + ".");
    const payload = await response.json();
    if (!Array.isArray(payload)) throw new Error("Hugging Face dataset discovery returned an invalid response.");

    const candidates = payload.map(normalizeCandidate).filter(Boolean)
      .sort((a, b) => (b.downloads ?? -1) - (a.downloads ?? -1))
      .slice(0, limit);

    return {
      schema: "dream-public-dataset-discovery-v1",
      retrievedAt: now.toISOString(),
      query: query.trim(),
      provider: "Hugging Face Hub public dataset metadata",
      candidateCount: candidates.length,
      trainingStarted: false,
      datasetFilesDownloaded: false,
      candidates,
      limitations: [
        "This operation discovers public dataset metadata only; it does not download dataset files or start training.",
        "A declared license or Hub tag is an unverified claim, not proof of rights for every record or AI training.",
        "Every candidate remains unapproved until rights, privacy, provenance, quality and source-group separation are reviewed.",
        "Gated and private datasets are excluded; DREAM does not request access or bypass restrictions."
      ]
    };
  } catch (error) {
    if (error?.name === "AbortError") throw new Error("Hugging Face dataset discovery timed out.");
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

function parseArgs(argv) {
  const args = { query: "", limit: DEFAULT_LIMIT, output: "" };
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    if (flag === "--query" && argv[index + 1]) args.query = argv[++index];
    else if (flag === "--limit" && argv[index + 1]) args.limit = Number(argv[++index]);
    else if (flag === "--out" && argv[index + 1]) args.output = argv[++index];
    else if (flag === "--help" || flag === "-h") args.help = true;
    else throw new Error("Unknown or incomplete argument: " + flag);
  }
  return args;
}

async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  if (args.help) {
    process.stdout.write("Usage: npm run discover:datasets -- --query \"Kenya education\" [--limit 12] [--out eval-results/candidates.json]\n");
    return 0;
  }
  if (!args.query) throw new Error("A search phrase is required. Use --query \"...\".");
  const report = await discoverPublicDatasets(args.query, { limit: args.limit });
  const evaluationDirectory = resolve(process.cwd(), "eval-results");
  const defaultName = "public-dataset-discovery-" +
    report.retrievedAt.replace(/[:.]/g, "-") + ".json";
  const output = resolve(process.cwd(), args.output || "eval-results/" + defaultName);
  const fromEvaluationDirectory = relative(evaluationDirectory, output);
  if (isAbsolute(fromEvaluationDirectory) ||
      fromEvaluationDirectory === "" ||
      fromEvaluationDirectory === ".." ||
      fromEvaluationDirectory.startsWith(".." + sep)) {
    throw new Error("Output must remain inside the eval-results directory.");
  }
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, JSON.stringify(report, null, 2) + "\n", { encoding: "utf8", flag: "wx" });
  process.stdout.write(JSON.stringify({
    status: "completed",
    output: relative(process.cwd(), output),
    candidateCount: report.candidateCount,
    trainingStarted: report.trainingStarted,
    datasetFilesDownloaded: report.datasetFilesDownloaded
  }) + "\n");
  return 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(error => {
    process.stderr.write("Dataset discovery failed: " + error.message + "\n");
    process.exitCode = 1;
  });
}
