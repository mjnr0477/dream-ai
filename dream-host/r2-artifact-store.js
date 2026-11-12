import { createHash, randomUUID } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { Transform } from "node:stream";
import { pipeline } from "node:stream/promises";

const SHA256_RE = /^[a-f0-9]{64}$/;
const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$/;
const METADATA_SCHEMA = "dream-artifact-v1";

export class R2ArtifactStoreError extends Error {
  constructor(message, status = 500) {
    super(message);
    this.name = "R2ArtifactStoreError";
    this.status = status;
  }
}

function safeName(value) {
  const name = String(value || "");
  if (!NAME_RE.test(name) || name === "." || name === "..") {
    throw new R2ArtifactStoreError("Artifact name must use 1-120 letters, numbers, dots, underscores or hyphens.", 400);
  }
  return name;
}

function assertHash(value) {
  const hash = String(value || "");
  if (!SHA256_RE.test(hash)) throw new R2ArtifactStoreError("Artifact SHA-256 is invalid.", 400);
  return hash;
}

async function bodyToBuffer(body, maxBytes = 64 * 1024) {
  if (!body) throw new R2ArtifactStoreError("R2 returned an empty metadata body.", 502);
  const chunks = [];
  let size = 0;
  for await (const chunk of body) {
    const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    size += bytes.length;
    if (size > maxBytes) throw new R2ArtifactStoreError("R2 metadata object exceeds the allowed size.", 502);
    chunks.push(bytes);
  }
  return Buffer.concat(chunks, size);
}

function metadataIsValid(value, sha256) {
  return value?.schema === METADATA_SCHEMA
    && value.sha256 === sha256
    && Number.isSafeInteger(value.size_bytes)
    && value.size_bytes > 0
    && typeof value.name === "string"
    && NAME_RE.test(value.name)
    && typeof value.created_at === "string";
}

/**
 * Content-addressed R2 object store. Uploads stream to a private staging key,
 * then use R2 CopyObject to move the completed bytes to their SHA-256 key.
 * The client and command constructors are injected for credential-free tests.
 */
export function createR2ArtifactStore({ client, bucket, commands }) {
  if (!client || typeof client.send !== "function") {
    throw new TypeError("An S3-compatible client is required.");
  }
  if (!/^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$/.test(String(bucket || ""))) {
    throw new TypeError("A valid R2 bucket name is required.");
  }
  for (const name of ["PutObjectCommand", "GetObjectCommand", "ListObjectsV2Command", "CopyObjectCommand", "DeleteObjectCommand"]) {
    if (typeof commands?.[name] !== "function") throw new TypeError(`Missing S3 command: ${name}`);
  }
  const { PutObjectCommand, GetObjectCommand, ListObjectsV2Command, CopyObjectCommand, DeleteObjectCommand } = commands;

  async function putArtifact({ name, source, contentLength, maxBytes = 512 * 1024 * 1024 }) {
    const safe = safeName(name);
    const expectedBytes = Number(contentLength);
    if (!Number.isSafeInteger(expectedBytes) || expectedBytes < 1) {
      source?.resume?.();
      throw new R2ArtifactStoreError("Artifact upload must include a valid Content-Length header.", 411);
    }
    if (expectedBytes > maxBytes) {
      source.resume?.();
      throw new R2ArtifactStoreError("Artifact exceeds the configured upload limit.", 413);
    }
    if (!source || typeof source.pipe !== "function") {
      throw new R2ArtifactStoreError("Artifact upload stream is invalid.", 400);
    }

    const digest = createHash("sha256");
    let size = 0;
    let streamFailure = null;
    const meter = new Transform({
      transform(chunk, encoding, callback) {
        const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
        size += bytes.length;
        if (size > maxBytes || size > expectedBytes) {
          streamFailure = new R2ArtifactStoreError("Artifact exceeds its declared or configured upload size.", 413);
          callback(streamFailure);
          return;
        }
        digest.update(bytes);
        callback(null, bytes);
      }
    });
    const temporaryKey = "uploads/staging/" + randomUUID() + ".part";

    try {
      source.on("error", error => meter.destroy(error));
      const uploadPromise = client.send(new PutObjectCommand({
        Bucket: bucket,
        Key: temporaryKey,
        Body: meter,
        ContentLength: expectedBytes,
        ContentType: "application/octet-stream"
      }));
      source.pipe(meter);
      await uploadPromise;
      if (size < 1) throw new R2ArtifactStoreError("Artifact upload is empty.", 400);
      if (size !== expectedBytes) throw new R2ArtifactStoreError("Artifact byte count did not match Content-Length.", 400);

      const sha256 = digest.digest("hex");
      const objectKey = "objects/" + sha256 + ".bin";
      await client.send(new CopyObjectCommand({
        Bucket: bucket,
        Key: objectKey,
        CopySource: bucket + "/" + temporaryKey,
        MetadataDirective: "COPY"
      }));
      await client.send(new DeleteObjectCommand({ Bucket: bucket, Key: temporaryKey })).catch(() => {});

      const metadata = {
        schema: METADATA_SCHEMA,
        sha256,
        name: safe,
        size_bytes: size,
        created_at: new Date().toISOString()
      };
      const metadataBytes = Buffer.from(JSON.stringify(metadata) + "\n");
      await client.send(new PutObjectCommand({
        Bucket: bucket,
        Key: "metadata/" + sha256 + ".json",
        Body: metadataBytes,
        ContentLength: metadataBytes.length,
        ContentType: "application/json"
      }));
      return metadata;
    } catch (error) {
      await client.send(new DeleteObjectCommand({ Bucket: bucket, Key: temporaryKey })).catch(() => {});
      if (streamFailure) throw streamFailure;
      if (error instanceof R2ArtifactStoreError) throw error;
      const failure = new R2ArtifactStoreError("R2 could not store the artifact.", 502);
      failure.cause = error;
      throw failure;
    }
  }

  async function getMetadata(sha256) {
    const hash = assertHash(sha256);
    try {
      const result = await client.send(new GetObjectCommand({
        Bucket: bucket,
        Key: "metadata/" + hash + ".json"
      }));
      const bytes = await bodyToBuffer(result.Body);
      const metadata = JSON.parse(bytes.toString("utf8"));
      if (!metadataIsValid(metadata, hash)) {
        throw new R2ArtifactStoreError("Artifact metadata failed validation.", 409);
      }
      return metadata;
    } catch (error) {
      if (error instanceof R2ArtifactStoreError) throw error;
      if (["NoSuchKey", "NotFound", "NoSuchObject"].includes(error?.name) || error?.$metadata?.httpStatusCode === 404) {
        throw new R2ArtifactStoreError("Artifact not found.", 404);
      }
      throw new R2ArtifactStoreError("R2 metadata could not be read.", 502);
    }
  }

  async function listArtifacts() {
    const items = [];
    let continuationToken;
    do {
      const page = await client.send(new ListObjectsV2Command({
        Bucket: bucket,
        Prefix: "metadata/",
        MaxKeys: 1000,
        ...(continuationToken ? { ContinuationToken: continuationToken } : {})
      }));
      for (const object of page.Contents || []) {
        const match = String(object.Key || "").match(/^metadata\/([a-f0-9]{64})\.json$/);
        if (!match) continue;
        try {
          items.push(await getMetadata(match[1]));
        } catch {
          // Ignore incomplete metadata records; never expose raw object keys.
        }
      }
      continuationToken = page.IsTruncated ? page.NextContinuationToken : undefined;
    } while (continuationToken);
    return items.sort((a, b) => a.created_at.localeCompare(b.created_at));
  }

  async function getArtifact(sha256) {
    const hash = assertHash(sha256);
    const metadata = await getMetadata(hash);
    let response;
    try {
      response = await client.send(new GetObjectCommand({
        Bucket: bucket,
        Key: "objects/" + hash + ".bin"
      }));
    } catch (error) {
      if (["NoSuchKey", "NotFound", "NoSuchObject"].includes(error?.name) || error?.$metadata?.httpStatusCode === 404) {
        throw new R2ArtifactStoreError("Artifact object is missing.", 404);
      }
      throw new R2ArtifactStoreError("R2 artifact could not be read.", 502);
    }
    if (!response?.Body) throw new R2ArtifactStoreError("R2 returned an empty artifact body.", 502);

    const directory = await mkdtemp(path.join(os.tmpdir(), "dream-r2-download-"));
    const temporaryPath = path.join(directory, hash + ".bin");
    const digest = createHash("sha256");
    let size = 0;
    const verifier = new Transform({
      transform(chunk, encoding, callback) {
        const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
        size += bytes.length;
        digest.update(bytes);
        callback(null, bytes);
      }
    });
    try {
      await pipeline(response.Body, verifier, createWriteStream(temporaryPath, { flags: "wx", mode: 0o600 }));
      if (size !== metadata.size_bytes || digest.digest("hex") !== hash) {
        throw new R2ArtifactStoreError("Artifact content failed SHA-256 integrity verification.", 409);
      }
    } catch (error) {
      await rm(directory, { recursive: true, force: true }).catch(() => {});
      if (error instanceof R2ArtifactStoreError) throw error;
      throw new R2ArtifactStoreError("R2 artifact download failed.", 502);
    }

    const stream = createReadStream(temporaryPath);
    const cleanup = () => { rm(directory, { recursive: true, force: true }).catch(() => {}); };
    stream.once("close", cleanup);
    stream.once("error", cleanup);
    return { metadata, stream };
  }

  return { putArtifact, listArtifacts, getArtifact };
}
