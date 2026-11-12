import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import { createHash } from "node:crypto";
import { getArtifact, listArtifacts, putArtifact } from "./artifact-store.js";

test("artifact vault stores content by SHA-256 and verifies metadata on retrieval", async t => {
  const root = await mkdtemp(path.join(tmpdir(), "dream-artifacts-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const payload = Buffer.from("DREAM native checkpoint test payload");
  const expected = createHash("sha256").update(payload).digest("hex");
  const stored = await putArtifact({
    root,
    name: "dream-training-artifacts.zip",
    source: Readable.from([payload]),
    maxBytes: 1024
  });
  assert.equal(stored.size_bytes, payload.length);
  assert.match(stored.sha256, /^[a-f0-9]{64}$/);
  assert.equal(stored.sha256, expected);
  const items = await listArtifacts(root);
  assert.equal(items.length, 1);
  assert.equal(items[0].sha256, stored.sha256);
  const retrieved = await getArtifact(root, stored.sha256);
  const chunks = [];
  for await (const chunk of retrieved.stream) chunks.push(chunk);
  assert.deepEqual(Buffer.concat(chunks), payload);
});

test("artifact vault rejects unsafe names, empty files and oversized uploads", async t => {
  const root = await mkdtemp(path.join(tmpdir(), "dream-artifacts-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await assert.rejects(
    putArtifact({ root, name: "../secret", source: Readable.from(["x"]) }),
    /Artifact name/
  );
  await assert.rejects(
    putArtifact({ root, name: "empty.bin", source: Readable.from([]) }),
    /empty/
  );
  await assert.rejects(
    putArtifact({ root, name: "large.bin", source: Readable.from([Buffer.alloc(9)]), maxBytes: 8 }),
    /upload limit/
  );
});

test("artifact vault requires a valid content hash for downloads", async t => {
  const root = await mkdtemp(path.join(tmpdir(), "dream-artifacts-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await assert.rejects(getArtifact(root, "not-a-hash"), /SHA-256/);
});

test("artifact vault detects stored-byte tampering", async t => {
  const root = await mkdtemp(path.join(tmpdir(), "dream-artifacts-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const payload = Buffer.from("original checkpoint bytes");
  const stored = await putArtifact({
    root,
    name: "checkpoint.bin",
    source: Readable.from([payload])
  });
  await writeFile(path.join(root, "objects", stored.sha256 + ".bin"), Buffer.from("tampered"));
  await assert.rejects(getArtifact(root, stored.sha256), /SHA-256 integrity/);
});


test("R2 artifact vault streams to content-addressed objects, lists metadata and verifies downloads", async () => {
  const { createR2ArtifactStore } = await import("./r2-artifact-store.js");
  const objects = new Map();
  class PutObjectCommand { constructor(input) { this.input = input; } }
  class GetObjectCommand { constructor(input) { this.input = input; } }
  class ListObjectsV2Command { constructor(input) { this.input = input; } }
  class CopyObjectCommand { constructor(input) { this.input = input; } }
  class DeleteObjectCommand { constructor(input) { this.input = input; } }
  const commands = { PutObjectCommand, GetObjectCommand, ListObjectsV2Command, CopyObjectCommand, DeleteObjectCommand };
  const client = {
    async send(command) {
      const input = command.input;
      if (command instanceof PutObjectCommand) {
        let bytes;
        if (Buffer.isBuffer(input.Body) || input.Body instanceof Uint8Array || typeof input.Body === "string") {
          bytes = Buffer.from(input.Body);
        } else {
          const chunks = [];
          for await (const chunk of input.Body) chunks.push(Buffer.from(chunk));
          bytes = Buffer.concat(chunks);
        }
        objects.set(input.Key, bytes);
        return {};
      }
      if (command instanceof CopyObjectCommand) {
        const sourceKey = input.CopySource.slice(input.CopySource.indexOf("/") + 1);
        objects.set(input.Key, Buffer.from(objects.get(sourceKey)));
        return {};
      }
      if (command instanceof DeleteObjectCommand) {
        objects.delete(input.Key);
        return {};
      }
      if (command instanceof CopyObjectCommand) {
        const sourceKey = input.CopySource.slice(input.CopySource.indexOf("/") + 1);
        const sourceBytes = objects.get(sourceKey);
        if (!sourceBytes) throw new Error("Copy source missing");
        objects.set(input.Key, Buffer.from(sourceBytes));
        return {};
      }
      if (command instanceof DeleteObjectCommand) {
        objects.delete(input.Key);
        return {};
      }
      if (command instanceof GetObjectCommand) {
        const value = objects.get(input.Key);
        if (!value) {
          const error = new Error("Not found");
          error.name = "NoSuchKey";
          throw error;
        }
        return { Body: Readable.from([value]), ContentLength: value.length };
      }
      if (command instanceof ListObjectsV2Command) {
        return {
          Contents: [...objects.keys()]
            .filter(key => key.startsWith(input.Prefix))
            .map(Key => ({ Key })),
          IsTruncated: false
        };
      }
      throw new Error("Unexpected R2 command");
    }
  };
  const store = createR2ArtifactStore({ client, bucket: "dream-ai-model-vault", commands });
  const payload = Buffer.from("DREAM candidate checkpoint stored in private R2");
  const expected = createHash("sha256").update(payload).digest("hex");
  const stored = await store.putArtifact({
    name: "dream-candidate.zip",
    source: Readable.from([payload]),
    contentLength: payload.length,
    maxBytes: 1024
  });
  assert.equal(stored.sha256, expected);
  assert.equal(objects.has("objects/" + expected + ".bin"), true);
  const listed = await store.listArtifacts();
  assert.equal(listed.length, 1);
  assert.equal(listed[0].name, "dream-candidate.zip");

  const downloaded = await store.getArtifact(expected);
  const chunks = [];
  for await (const chunk of downloaded.stream) chunks.push(chunk);
  assert.deepEqual(Buffer.concat(chunks), payload);

  objects.set("objects/" + expected + ".bin", Buffer.from("tampered"));
  await assert.rejects(store.getArtifact(expected), /SHA-256 integrity/);
});

test("R2 artifact vault rejects missing lengths and unsafe artifact names", async () => {
  const { createR2ArtifactStore } = await import("./r2-artifact-store.js");
  class PutObjectCommand { constructor(input) { this.input = input; } }
  class GetObjectCommand { constructor(input) { this.input = input; } }
  class ListObjectsV2Command { constructor(input) { this.input = input; } }
  const store = createR2ArtifactStore({
    bucket: "dream-ai-model-vault",
    commands: { PutObjectCommand, GetObjectCommand, ListObjectsV2Command, CopyObjectCommand, DeleteObjectCommand },
    client: { async send() { throw new Error("Should not reach R2"); } }
  });
  await assert.rejects(
    store.putArtifact({ name: "bad/name", source: Readable.from(["x"]), contentLength: 1 }),
    /Artifact name/
  );
  await assert.rejects(
    store.putArtifact({ name: "checkpoint.bin", source: Readable.from(["x"]) }),
    /Content-Length/
  );
});
