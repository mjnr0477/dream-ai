import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const workflows = [
  "../.github/workflows/dream-ai-auto-training.yml",
  "../.github/workflows/dream-ai-autotrain.yml",
  "../.github/workflows/dream-ai-autonomous-training.yml"
];

for (const relativePath of workflows) {
  test(`${relativePath} stays retired and cannot schedule or submit remote training`, async () => {
    const source = await readFile(new URL(relativePath, import.meta.url), "utf8");

    assert.match(source, /^name: .*retired/im);
    assert.match(source, /^on:\s*\n\s+workflow_dispatch:/m);
    assert.doesNotMatch(source, /^\s+schedule:/m);
    assert.doesNotMatch(source, /\b(?:kaggle|hugging\s*face|hf_token|kaggle_[a-z_]+|lightning_api_key)\b/i);
    assert.doesNotMatch(source, /uses:\s*actions\/checkout|pip install|python .*launch/i);
    assert.match(source, /No training was started and no model files were changed\./);
  });
}
