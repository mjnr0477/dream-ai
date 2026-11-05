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
 * This validates and returns text only; it never writes or executes generated code.
 */
export function parseCodeArtifacts(output, limits = {}) {
  if (typeof output !== "string") {
    throw new TypeError("Code artifact output must be text.");
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
