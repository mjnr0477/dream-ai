const DEFAULT_LIMITS = Object.freeze({
  maxFiles: 50,
  maxTotalBytes: 256 * 1024,
  maxFileBytes: 64 * 1024,
  maxPathLength: 240
});
const UTF8_FLAG = 0x0800;
const DOS_DATE_1980_01_01 = 0x0021;
const encoder = new TextEncoder();
const CRC_TABLE = new Uint32Array(256);

for (let value = 0; value < 256; value += 1) {
  let crc = value;
  for (let bit = 0; bit < 8; bit += 1) {
    crc = (crc & 1) ? (0xedb88320 ^ (crc >>> 1)) : (crc >>> 1);
  }
  CRC_TABLE[value] = crc >>> 0;
}

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function validateLimits(limits) {
  if (!limits || typeof limits !== "object" || Array.isArray(limits)) {
    throw new TypeError("Project ZIP limits must be an object.");
  }
  const config = { ...DEFAULT_LIMITS, ...limits };
  for (const [key, value] of Object.entries(config)) {
    if (!Object.hasOwn(DEFAULT_LIMITS, key) || !Number.isSafeInteger(value) || value < 1) {
      throw new TypeError("Invalid project ZIP limit: " + key + ".");
    }
  }
  return config;
}

function validatePath(path, maxLength) {
  if (typeof path !== "string" || path.length < 1 || path.length > maxLength || path !== path.trim()) {
    throw new TypeError("Each project file needs a non-empty, bounded relative path.");
  }
  if (path.startsWith("/") || path.includes("\\") || path.includes("\0") ||
      /^[a-zA-Z]:/.test(path) ||
      path.split("/").some(segment => !segment || segment === "." || segment === "..")) {
    throw new TypeError("Project ZIP paths must be normalized relative POSIX paths without traversal.");
  }
  return path;
}

function makeLocalHeader(nameBytes, size, checksum) {
  const header = new Uint8Array(30);
  const view = new DataView(header.buffer);
  view.setUint32(0, 0x04034b50, true);
  view.setUint16(4, 20, true);
  view.setUint16(6, UTF8_FLAG, true);
  view.setUint16(8, 0, true);
  view.setUint16(10, 0, true);
  view.setUint16(12, DOS_DATE_1980_01_01, true);
  view.setUint32(14, checksum, true);
  view.setUint32(18, size, true);
  view.setUint32(22, size, true);
  view.setUint16(26, nameBytes.length, true);
  view.setUint16(28, 0, true);
  return header;
}

function makeCentralHeader(nameBytes, size, checksum, localOffset) {
  const header = new Uint8Array(46);
  const view = new DataView(header.buffer);
  view.setUint32(0, 0x02014b50, true);
  view.setUint16(4, 20, true);
  view.setUint16(6, 20, true);
  view.setUint16(8, UTF8_FLAG, true);
  view.setUint16(10, 0, true);
  view.setUint16(12, 0, true);
  view.setUint16(14, DOS_DATE_1980_01_01, true);
  view.setUint32(16, checksum, true);
  view.setUint32(20, size, true);
  view.setUint32(24, size, true);
  view.setUint16(28, nameBytes.length, true);
  view.setUint16(30, 0, true);
  view.setUint16(32, 0, true);
  view.setUint16(34, 0, true);
  view.setUint16(36, 0, true);
  view.setUint32(38, 0, true);
  view.setUint32(42, localOffset, true);
  return header;
}

/**
 * Build a bounded, uncompressed ZIP archive in the browser.
 * Only validated text files are packaged; this function never executes them.
 */
export function createProjectZip(artifacts, limits = {}) {
  const config = validateLimits(limits);
  if (!Array.isArray(artifacts) || artifacts.length < 1) {
    throw new TypeError("A project ZIP needs at least one file.");
  }
  if (artifacts.length > config.maxFiles) {
    throw new RangeError("Project ZIP exceeds the configured file count.");
  }

  const names = new Set();
  const files = [];
  let totalBytes = 0;

  for (const artifact of artifacts) {
    if (!artifact || typeof artifact !== "object" || Array.isArray(artifact) ||
        typeof artifact.content !== "string") {
      throw new TypeError("Each project file must contain a path and text content.");
    }
    const path = validatePath(artifact.path, config.maxPathLength);
    if (names.has(path)) throw new TypeError("Duplicate project ZIP path: " + path + ".");
    names.add(path);

    const nameBytes = encoder.encode(path);
    const contentBytes = encoder.encode(artifact.content);
    if (nameBytes.length > 0xffff) throw new RangeError("Project ZIP filename is too long.");
    if (contentBytes.length > config.maxFileBytes) {
      throw new RangeError("Project ZIP exceeds the per-file size limit: " + path + ".");
    }
    totalBytes += contentBytes.length;
    if (totalBytes > config.maxTotalBytes) {
      throw new RangeError("Project ZIP exceeds the total size limit.");
    }
    if (contentBytes.length > 0xffffffff) {
      throw new RangeError("Project ZIP file exceeds the classic ZIP size limit.");
    }
    files.push({ nameBytes, contentBytes, checksum: crc32(contentBytes) });
  }

  const localParts = [];
  const centralParts = [];
  let localOffset = 0;
  let centralSize = 0;

  for (const file of files) {
    const localHeader = makeLocalHeader(file.nameBytes, file.contentBytes.length, file.checksum);
    localParts.push(localHeader, file.nameBytes, file.contentBytes);
    const localLength = localHeader.length + file.nameBytes.length + file.contentBytes.length;
    centralParts.push(makeCentralHeader(file.nameBytes, file.contentBytes.length, file.checksum, localOffset), file.nameBytes);
    centralSize += 46 + file.nameBytes.length;
    localOffset += localLength;
    if (localOffset > 0xffffffff || centralSize > 0xffffffff) {
      throw new RangeError("Project ZIP exceeds the classic ZIP archive size limit.");
    }
  }

  const end = new Uint8Array(22);
  const endView = new DataView(end.buffer);
  endView.setUint32(0, 0x06054b50, true);
  endView.setUint16(4, 0, true);
  endView.setUint16(6, 0, true);
  endView.setUint16(8, files.length, true);
  endView.setUint16(10, files.length, true);
  endView.setUint32(12, centralSize, true);
  endView.setUint32(16, localOffset, true);
  endView.setUint16(20, 0, true);

  return new Blob([...localParts, ...centralParts, end], { type: "application/zip" });
}
