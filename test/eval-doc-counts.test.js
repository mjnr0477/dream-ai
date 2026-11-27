import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

const root = new URL("../", import.meta.url);
const read = path => readFile(new URL(path, root), "utf8");
const recordCount = source => source.split(/\r?\n/).filter(line => line.trim()).length;

test("evaluation documentation counts match the checked-in JSONL sets", async () => {
  const [core, african, readme, chatGuide] = await Promise.all([
    read("evals/core-v1.jsonl"),
    read("evals/african-languages-v1.jsonl"),
    read("evals/README.md"),
    read("docs/DREAM_CHAT_EVALUATION.md")
  ]);
  const coreCount = recordCount(core);
  const africanCount = recordCount(african);

  assert.match(readme, new RegExp("^\\| `core-v1` \\| " + coreCount + " \\|", "m"));
  assert.match(readme, new RegExp("^\\| `african-languages-v1` \\| " + africanCount + " \\|", "m"));
  assert.match(
    chatGuide,
    new RegExp("core set currently contains " + coreCount + " cases across \\d+ categories")
  );
});
