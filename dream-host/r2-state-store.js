import { Readable } from "node:stream";

const SAFE_KEY = /^[A-Za-z0-9][A-Za-z0-9/_-]{0,180}\.json$/;

export class R2StateStoreError extends Error {
  constructor(message, status = 500) {
    super(message);
    this.name = "R2StateStoreError";
    this.status = status;
  }
}

function validateKey(key) {
  const value = String(key || "");
  if (!SAFE_KEY.test(value) || value.includes("..") || value.startsWith("/") || value.includes("//")) {
    throw new R2StateStoreError("Invalid persistent state key.", 400);
  }
  return value;
}

function isNotFound(error) {
  return ["NoSuchKey", "NotFound", "NoSuchObject"].includes(error?.name)
    || error?.$metadata?.httpStatusCode === 404;
}

async function readBody(body, maxBytes) {
  if (!body) throw new R2StateStoreError("Persistent state object has no body.", 502);
  const chunks = [];
  let size = 0;
  if (typeof body.transformToByteArray === "function") {
    const bytes = Buffer.from(await body.transformToByteArray());
    if (bytes.length > maxBytes) throw new R2StateStoreError("Persistent state object exceeds the allowed size.", 413);
    return bytes;
  }
  if (typeof body[Symbol.asyncIterator] !== "function") {
    throw new R2StateStoreError("Persistent state body is not readable.", 502);
  }
  for await (const chunk of body) {
    const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    size += bytes.length;
    if (size > maxBytes) throw new R2StateStoreError("Persistent state object exceeds the allowed size.", 413);
    chunks.push(bytes);
  }
  return Buffer.concat(chunks, size);
}

export function createR2StateStore({ client, bucket, commands, prefix = "dream-ai-state/" }) {
  if (!client || typeof client.send !== "function") throw new TypeError("An R2-compatible client is required.");
  if (!/^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$/.test(String(bucket || ""))) throw new TypeError("A valid R2 bucket is required.");
  for (const name of ["GetObjectCommand", "PutObjectCommand", "ListObjectsV2Command", "DeleteObjectCommand"]) {
    if (typeof commands?.[name] !== "function") throw new TypeError("Missing object command: " + name);
  }
  const { GetObjectCommand, PutObjectCommand, ListObjectsV2Command, DeleteObjectCommand } = commands;
  const base = String(prefix).replace(/^\/+|\/+$/g, "") + "/";

  function objectKey(key) {
    return base + validateKey(key);
  }

  async function readJson(key, maxBytes = 4 * 1024 * 1024) {
    try {
      const response = await client.send(new GetObjectCommand({ Bucket: bucket, Key: objectKey(key) }));
      const bytes = await readBody(response.Body, maxBytes);
      try { return JSON.parse(bytes.toString("utf8")); }
      catch { throw new R2StateStoreError("Persistent state JSON is invalid.", 409); }
    } catch (error) {
      if (isNotFound(error)) return null;
      if (error instanceof R2StateStoreError) throw error;
      throw new R2StateStoreError("Persistent state could not be read from R2.", 502);
    }
  }

  async function writeJson(key, value, maxBytes = 4 * 1024 * 1024) {
    const bytes = Buffer.from(JSON.stringify(value));
    if (!bytes.length || bytes.length > maxBytes) throw new R2StateStoreError("Persistent state exceeds the configured size limit.", 413);
    try {
      await client.send(new PutObjectCommand({
        Bucket: bucket,
        Key: objectKey(key),
        Body: bytes,
        ContentLength: bytes.length,
        ContentType: "application/json; charset=utf-8",
        CacheControl: "no-store"
      }));
      return { key: objectKey(key), bytes: bytes.length };
    } catch {
      throw new R2StateStoreError("Persistent state could not be written to R2.", 503);
    }
  }

  async function list(prefix = "", maxKeys = 100) {
    const safePrefix = prefix ? validateKey(prefix.endsWith(".json") ? prefix : prefix + "index.json").replace(/index\.json$/, "") : "";
    const keys = [];
    let continuationToken;
    do {
      let page;
      try {
        page = await client.send(new ListObjectsV2Command({
          Bucket: bucket,
          Prefix: base + safePrefix,
          MaxKeys: Math.min(1000, Math.max(1, maxKeys)),
          ...(continuationToken ? { ContinuationToken: continuationToken } : {})
        }));
      } catch {
        throw new R2StateStoreError("Persistent state index could not be listed from R2.", 502);
      }
      for (const item of page.Contents || []) {
        const key = String(item.Key || "");
        if (!key.startsWith(base)) continue;
        const relative = key.slice(base.length);
        if (relative.endsWith(".json")) keys.push(relative);
        if (keys.length >= maxKeys) return keys.slice(0, maxKeys);
      }
      continuationToken = page.IsTruncated ? page.NextContinuationToken : undefined;
    } while (continuationToken);
    return keys;
  }

  async function remove(key) {
    try {
      await client.send(new DeleteObjectCommand({ Bucket: bucket, Key: objectKey(key) }));
    } catch {
      throw new R2StateStoreError("Persistent state object could not be deleted from R2.", 502);
    }
  }

  return { readJson, writeJson, list, remove };
}
