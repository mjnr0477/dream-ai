import test from "node:test";
import assert from "node:assert/strict";
import { Readable } from "node:stream";
import { createR2StateStore } from "./r2-state-store.js";

class GetObjectCommand { constructor(input) { this.input = input; } }
class PutObjectCommand { constructor(input) { this.input = input; } }
class ListObjectsV2Command { constructor(input) { this.input = input; } }
class DeleteObjectCommand { constructor(input) { this.input = input; } }

function fixture() {
  const objects = new Map();
  const client = {
    async send(command) {
      const { Bucket, Key } = command.input;
      assert.equal(Bucket, "dream-test");
      if (command instanceof PutObjectCommand) {
        objects.set(Key, Buffer.from(command.input.Body));
        return {};
      }
      if (command instanceof GetObjectCommand) {
        const value = objects.get(Key);
        if (!value) {
          const error = new Error("missing");
          error.name = "NoSuchKey";
          throw error;
        }
        return { Body: Readable.from([value]) };
      }
      if (command instanceof ListObjectsV2Command) {
        const keys = [...objects.keys()].filter(key => key.startsWith(command.input.Prefix));
        return {
          Contents: keys.map(Key => ({ Key })),
          IsTruncated: false
        };
      }
      if (command instanceof DeleteObjectCommand) {
        objects.delete(Key);
        return {};
      }
      throw new Error("Unexpected command");
    }
  };
  return {
    objects,
    store: createR2StateStore({
      client,
      bucket: "dream-test",
      commands: { GetObjectCommand, PutObjectCommand, ListObjectsV2Command, DeleteObjectCommand }
    })
  };
}

test("R2 state store writes and reads JSON from a stable object key", async () => {
  const { objects, store } = fixture();
  const value = { schema: "dream-test-v1", jobs: [{ id: "job-1" }] };
  const saved = await store.writeJson("notebook/queue.json", value);
  assert.equal(saved.bytes, Buffer.byteLength(JSON.stringify(value)));
  assert.deepEqual(await store.readJson("notebook/queue.json"), value);
  assert.ok(objects.has("dream-ai-state/notebook/queue.json"));
});

test("R2 state store returns null for a missing state object", async () => {
  const { store } = fixture();
  assert.equal(await store.readJson("notebook/missing.json"), null);
});

test("R2 state store enforces key traversal and payload size limits", async () => {
  const { store } = fixture();
  await assert.rejects(store.writeJson("../secrets.json", { token: "x" }), /key/i);
  await assert.rejects(store.readJson("notebook//queue.json"), /key/i);
  await assert.rejects(store.writeJson("notebook/large.json", { value: "x".repeat(100) }, 20), /size limit/i);
});

test("R2 state store lists and removes only JSON objects under its namespace", async () => {
  const { objects, store } = fixture();
  await store.writeJson("notebooks/nb-1.json", { id: "nb-1" });
  await store.writeJson("notebooks/nb-2.json", { id: "nb-2" });
  assert.deepEqual((await store.list("notebooks/")).sort(), ["notebooks/nb-1.json", "notebooks/nb-2.json"]);
  await store.remove("notebooks/nb-1.json");
  assert.equal(await store.readJson("notebooks/nb-1.json"), null);
  assert.equal(objects.size, 1);
});
