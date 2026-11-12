import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import { getArtifact, listArtifacts, putArtifact } from "./artifact-store.js";

test("artifact vault stores content by SHA-256 and verifies metadata on retrieval", async t => {
  const root = await mkdtemp(path.join(tmpdir(), "dream-artifacts-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const payload = Buffer.from("DREAM native checkpoint test payload");
  const expected = "d5b3b72bb0c7c0d4c09f2e4e2f8a6d9c4a5a2f4e0b3e8d2a9c7f5a1b3d6e9f0a";
  const stored = await putArtifact({
    root,
    name: "dream-training-artifacts.zip",
    source: Readable.from([payload]),
    maxBytes: 1024
  });
  assert.equal(stored.size_bytes, payload.length);
  assert.match(stored.sha256, /^[a-f0-9]{64}$/);
  assert.notEqual(stored.sha256, expected);
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
