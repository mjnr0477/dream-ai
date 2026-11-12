import { createHash, randomUUID } from "node:crypto";
import { mkdir, open, readFile, readdir, rename, rm, stat } from "node:fs/promises";
import { createReadStream } from "node:fs";
import path from "node:path";

const SHA256_RE = /^[a-f0-9]{64}$/;
const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$/;

export class ArtifactStoreError extends Error {
  constructor(message, status = 400) {
    super(message);
    this.status = status;
  }
}

function safeName(value) {
  const name = String(value || "");
  if (!NAME_RE.test(name) || name === "." || name === "..") {
    throw new ArtifactStoreError("Artifact name must use 1-120 letters, numbers, dots, underscores or hyphens.");
  }
  return name;
}

function pathsFor(root) {
  const base = path.resolve(root);
  return {
    base,
    objects: path.join(base, "objects"),
    metadata: path.join(base, "metadata"),
    temporary: path.join(base, ".tmp")
  };
}

async function ensureDirectories(paths) {
  await Promise.all([
    mkdir(paths.objects, { recursive: true }),
    mkdir(paths.metadata, { recursive: true }),
    mkdir(paths.temporary, { recursive: true })
  ]);
}

export async function putArtifact({ root, name, source, maxBytes = 512 * 1024 * 1024 }) {
  const safe = safeName(name);
  if (!root) throw new ArtifactStoreError("Persistent artifact storage is not configured.", 503);
  const paths = pathsFor(root);
  await ensureDirectories(paths);
  const temporaryPath = path.join(paths.temporary, randomUUID() + ".part");
  const handle = await open(temporaryPath, "wx", 0o600);
  const digest = createHash("sha256");
  let size = 0;
  try {
    for await (const chunk of source) {
      const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      size += bytes.length;
      if (size > maxBytes) throw new ArtifactStoreError("Artifact exceeds the configured upload limit.", 413);
      digest.update(bytes);
      await handle.write(bytes);
    }
    if (size < 1) throw new ArtifactStoreError("Artifact upload is empty.");
    await handle.sync();
    await handle.close();
    const sha256 = digest.digest("hex");
    const objectPath = path.join(paths.objects, sha256 + ".bin");
    try {
      await rename(temporaryPath, objectPath);
    } catch (error) {
      if (error.code !== "EEXIST") throw error;
      await rm(temporaryPath, { force: true });
    }
    const metadata = {
      schema: "dream-artifact-v1",
      sha256,
      name: safe,
      size_bytes: size,
      created_at: new Date().toISOString()
    };
    const metadataPath = path.join(paths.metadata, sha256 + ".json");
    try {
      const tempMetadata = path.join(paths.temporary, randomUUID() + ".json");
      await open(tempMetadata, "wx", 0o600).then(async file => {
        try { await file.writeFile(JSON.stringify(metadata) + "\n"); await file.sync(); }
        finally { await file.close(); }
      });
      try { await rename(tempMetadata, metadataPath); }
      catch (error) {
        await rm(tempMetadata, { force: true });
        if (error.code !== "EEXIST") throw error;
      }
    } catch (error) {
      if (error.code !== "EEXIST") throw error;
    }
    return metadata;
  } catch (error) {
    await handle.close().catch(() => {});
    await rm(temporaryPath, { force: true }).catch(() => {});
    throw error;
  }
}

export async function listArtifacts(root) {
  if (!root) throw new ArtifactStoreError("Persistent artifact storage is not configured.", 503);
  const paths = pathsFor(root);
  await ensureDirectories(paths);
  const names = (await readdir(paths.metadata)).filter(name => SHA256_RE.test(name.replace(/\.json$/, "")) && name.endsWith(".json"));
  const items = [];
  for (const name of names) {
    try {
      const value = JSON.parse(await readFile(path.join(paths.metadata, name), "utf8"));
      if (value?.schema === "dream-artifact-v1" && SHA256_RE.test(value.sha256) && value.sha256 === name.slice(0, -5)) {
        items.push(value);
      }
    } catch {
      // Ignore incomplete or invalid metadata; never expose raw files.
    }
  }
  return items.sort((a, b) => a.created_at.localeCompare(b.created_at));
}

export async function getArtifact(root, sha256) {
  if (!root) throw new ArtifactStoreError("Persistent artifact storage is not configured.", 503);
  if (!SHA256_RE.test(String(sha256 || ""))) throw new ArtifactStoreError("Artifact SHA-256 is invalid.");
  const paths = pathsFor(root);
  const objectPath = path.join(paths.objects, sha256 + ".bin");
  const metadataPath = path.join(paths.metadata, sha256 + ".json");
  let metadata;
  try {
    metadata = JSON.parse(await readFile(metadataPath, "utf8"));
    const info = await stat(objectPath);
    if (metadata?.schema !== "dream-artifact-v1" || metadata.sha256 !== sha256 || info.size !== metadata.size_bytes) {
      throw new Error("Artifact metadata mismatch");
    }
  } catch {
    throw new ArtifactStoreError("Artifact not found or failed integrity metadata checks.", 404);
  }
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(objectPath)) digest.update(chunk);
  if (digest.digest("hex") !== sha256) {
    throw new ArtifactStoreError("Artifact content failed SHA-256 integrity verification.", 409);
  }
  return { metadata, stream: createReadStream(objectPath) };
}
