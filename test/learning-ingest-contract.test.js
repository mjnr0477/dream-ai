import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

test("learning ingest preserves v1 compatibility and requires v2 session grouping", async () => {
  const source = await readFile(
    new URL("../supabase/functions/dream-ai-learning-ingest-v3/index.ts", import.meta.url),
    "utf8"
  );
  assert.match(source, /dream-learning-opt-in-v1/);
  assert.match(source, /dream-learning-opt-in-v2/);
  assert.match(source, /UUID_V4\.test\(sourceGroupId\)/);
  assert.match(source, /source_group_sha256:\s*await digest\("dream-ai-session-group-v1\\0"\s*\+\s*sourceGroupId\)/);
});

test("learning ingest stores the source-group fingerprint, never the raw UUID", async () => {
  const source = await readFile(
    new URL("../supabase/functions/dream-ai-learning-ingest-v3/index.ts", import.meta.url),
    "utf8"
  );
  const start = source.indexOf('.from("dream_ai_training_examples").insert({');
  const end = source.indexOf('}).select("id").single()', start);
  assert.notEqual(start, -1);
  assert.notEqual(end, -1);
  const insertPayload = source.slice(start, end);
  assert.match(insertPayload, /metadata/);
  assert.doesNotMatch(insertPayload, /sourceGroupId/);
  assert.match(source, /review_status:\s*"pending"/);
  assert.match(source, /status:\s*"pending"/);
});
