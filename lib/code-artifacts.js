const DEFAULT_LIMITS = Object.freeze({
  maxFiles: 50,
  maxTotalBytes: 256 * 1024,
  maxFileBytes: 64 * 1024,
  maxPathLength: 240
});

function validateRelativePath(path, maxPathLength) {
  if (typeof path !== "string" || path.length === 0 || path.length > maxPathLength) {
    throw new TypeError("Each code artifact needs a non-empty, bounded relative path.");
  }
  if (path !== path.trim()) {
    throw new TypeError("Code artifact paths must not contain surrounding whitespace.");
  }
  if (
    path.startsWith("/") ||
    path.includes("\\") ||
    path.includes("\0") ||
    /^[a-zA-Z]:/.test(path) ||
    path.split("/").some(segment => !segment || segment === "." || segment === "..")
  ) {
    throw new TypeError("Code artifact paths must be normalized relative POSIX paths without traversal.");
  }
  return path;
}

/**
 * Parse a model-authored, non-executable multi-file response.
 * This validates and returns text only. It never writes files, invokes a shell,
 * installs packages, or executes generated code.
 */
export function parseCodeArtifacts(output, limits = {}) {
  if (typeof output !== "string") {
    throw new TypeError("Code artifact output must be text.");
  }
  if (!limits || typeof limits !== "object" || Array.isArray(limits)) {
    throw new TypeError("Code artifact limits must be an object.");
  }

  const config = { ...DEFAULT_LIMITS, ...limits };
  for (const key of Object.keys(DEFAULT_LIMITS)) {
    if (!Number.isSafeInteger(config[key]) || config[key] < 1) {
      throw new TypeError(`Invalid code artifact limit: ${key}.`);
    }
  }

  const artifacts = [];
  const paths = new Set();
  let cursor = 0;
  let totalBytes = 0;
  const pattern = /^--- DREAM FILE: ([^\r\n]*) ---\r?\n([\s\S]*?)^--- END DREAM FILE ---[ \t]*$/gm;
  let match;

  while ((match = pattern.exec(output)) !== null) {
    if (output.slice(cursor, match.index).trim()) {
      throw new TypeError("Unexpected text outside code artifact blocks.");
    }
    if (artifacts.length >= config.maxFiles) {
      throw new RangeError("Code artifact output exceeds the file-count limit.");
    }

    const path = validateRelativePath(match[1], config.maxPathLength);
    if (paths.has(path)) {
      throw new TypeError(`Duplicate code artifact path: ${path}.`);
    }
    paths.add(path);

    const content = match[2];
    const fileBytes = Buffer.byteLength(content, "utf8");
    if (fileBytes > config.maxFileBytes) {
      throw new RangeError(`Code artifact exceeds the per-file size limit: ${path}.`);
    }
    totalBytes += fileBytes;
    if (totalBytes > config.maxTotalBytes) {
      throw new RangeError("Code artifact output exceeds the total size limit.");
    }

    artifacts.push(Object.freeze({ path, content }));
    cursor = pattern.lastIndex;
  }

  if (artifacts.length === 0) {
    throw new TypeError("No complete DREAM code artifact blocks were found.");
  }
  if (output.slice(cursor).trim()) {
    throw new TypeError("Unexpected text outside code artifact blocks.");
  }

  return Object.freeze(artifacts);
}

const LANGUAGE_BY_EXTENSION = Object.freeze({
  js: "javascript", mjs: "javascript", cjs: "javascript",
  ts: "typescript", tsx: "tsx", jsx: "jsx",
  py: "python", java: "java", kt: "kotlin", kts: "kotlin",
  html: "html", htm: "html", css: "css", scss: "scss",
  json: "json", md: "markdown", yml: "yaml", yaml: "yaml",
  sh: "bash", bash: "bash", sql: "sql", go: "go", rs: "rust",
  rb: "ruby", php: "php", cs: "csharp", cpp: "cpp", c: "c",
  h: "c", swift: "swift", dart: "dart", xml: "xml"
});

/**
 * Convert validated file artifacts into the existing chat code-block format.
 * The client renders these as separate editable/copyable code blocks.
 */
export function formatCodeArtifacts(artifacts) {
  if (!Array.isArray(artifacts) || artifacts.length === 0) {
    throw new TypeError("At least one code artifact is required.");
  }
  return artifacts.map(({ path, content }) => {
    if (typeof path !== "string" || typeof content !== "string") {
      throw new TypeError("Each code artifact must contain a path and text content.");
    }
    const extension = path.includes(".") ? path.split(".").pop().toLowerCase() : "";
    const language = LANGUAGE_BY_EXTENSION[extension] || "";
    const source = content.endsWith("\n") ? content : content + "\n";
    return `**${path}**\n\n\`\`\`${language}\n${source}\`\`\``;
  }).join("\n");
}

/** Return null for ordinary coding prose; strictly validate responses that declare files. */
export function parseCodeArtifactsIfPresent(output, limits = {}) {
  if (typeof output !== "string") {
    throw new TypeError("Code artifact output must be text.");
  }
  if (!output.includes("--- DREAM FILE:")) return null;
  return parseCodeArtifacts(output, limits);
}
