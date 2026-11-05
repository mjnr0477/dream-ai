import test from "node:test";
import assert from "node:assert/strict";
import { formatCodeArtifacts, parseCodeArtifacts, parseCodeArtifactsIfPresent } from "./code-artifacts.js";

test("parses multiple files while preserving content and order", () => {
  const result = parseCodeArtifacts([
    "--- DREAM FILE: src/index.js ---",
    "export { answer } from './answer.js';",
    "--- END DREAM FILE ---",
    "",
    "--- DREAM FILE: src/answer.js ---",
    "export const answer = 42;",
    "--- END DREAM FILE ---"
  ].join("\n"));

  assert.deepEqual(result, [
    { path: "src/index.js", content: "export { answer } from './answer.js';\n" },
    { path: "src/answer.js", content: "export const answer = 42;\n" }
  ]);
  assert.equal(Object.isFrozen(result), true);
  assert.equal(Object.isFrozen(result[0]), true);
});

test("rejects non-text, invalid limits and incomplete artifact output", () => {
  assert.throws(() => parseCodeArtifacts(null), /must be text/i);
  assert.throws(() => parseCodeArtifacts("Here is your code"), /No complete/i);
  assert.throws(() => parseCodeArtifacts("--- DREAM FILE: a.js ---\ncode"), /No complete/i);
  assert.throws(() => parseCodeArtifacts("--- DREAM FILE: a.js ---\ncode\n--- END DREAM FILE ---", null), /must be an object/i);
  assert.throws(() => parseCodeArtifacts("--- DREAM FILE: a.js ---\ncode\n--- END DREAM FILE ---", []), /must be an object/i);
});

test("rejects prose outside artifact blocks to avoid silently dropping content", () => {
  const output = "Generated files:\n--- DREAM FILE: a.js ---\ncode\n--- END DREAM FILE ---";
  assert.throws(() => parseCodeArtifacts(output), /Unexpected text/i);
});

test("rejects absolute, traversal, non-normalized and platform-specific paths", () => {
  for (const path of ["/etc/passwd", "../secret", "src/../../secret", "src\\app.js", "C:/temp/a.js", "src//app.js", "./app.js", "src/./app.js", "", " src/app.js"]) {
    const output = `--- DREAM FILE: ${path} ---\ncode\n--- END DREAM FILE ---`;
    assert.throws(() => parseCodeArtifacts(output), /relative POSIX paths|non-empty|surrounding whitespace/i, path);
  }
});

test("rejects duplicate paths", () => {
  const output = [
    "--- DREAM FILE: src/app.js ---", "one", "--- END DREAM FILE ---",
    "--- DREAM FILE: src/app.js ---", "two", "--- END DREAM FILE ---"
  ].join("\n");
  assert.throws(() => parseCodeArtifacts(output), /Duplicate/i);
});

test("enforces file count, per-file bytes and aggregate bytes", () => {
  const twoFiles = [
    "--- DREAM FILE: a.js ---", "1234", "--- END DREAM FILE ---",
    "--- DREAM FILE: b.js ---", "5678", "--- END DREAM FILE ---"
  ].join("\n");

  assert.throws(() => parseCodeArtifacts(twoFiles, { maxFiles: 1 }), /file-count/i);
  assert.throws(() => parseCodeArtifacts(twoFiles, { maxFileBytes: 3 }), /per-file size/i);
  assert.throws(() => parseCodeArtifacts(twoFiles, { maxTotalBytes: 7 }), /total size/i);
});

test("measures UTF-8 bytes and validates limit configuration", () => {
  const output = "--- DREAM FILE: src/hello.js ---\n世界\n--- END DREAM FILE ---";
  assert.throws(() => parseCodeArtifacts(output, { maxFileBytes: 3 }), /per-file size/i);
  assert.throws(() => parseCodeArtifacts(output, { maxFiles: 0 }), /Invalid code artifact limit/i);
});

test("rejects paths that exceed the configured bound", () => {
  const output = "--- DREAM FILE: " + "a".repeat(241) + " ---\ncode\n--- END DREAM FILE ---";
  assert.throws(() => parseCodeArtifacts(output), /bounded relative path/i);
});

test("formats parsed files into separate language-aware editor code blocks", () => {
  const artifacts = parseCodeArtifacts([
    "--- DREAM FILE: src/main.ts ---",
    "export const value: number = 42;",
    "--- END DREAM FILE ---",
    "",
    "--- DREAM FILE: scripts/run.py ---",
    "print('hello')",
    "--- END DREAM FILE ---"
  ].join("\n"));
  const formatted = formatCodeArtifacts(artifacts);
  assert.ok(formatted.includes("**src/main.ts**"));
  assert.ok(formatted.includes("```typescript\nexport const value: number = 42;"));
  assert.ok(formatted.includes("**scripts/run.py**"));
  assert.ok(formatted.includes("```python\nprint('hello')"));
});

test("rejects empty or malformed artifact lists when formatting", () => {
  assert.throws(() => formatCodeArtifacts([]), /At least one/i);
  assert.throws(() => formatCodeArtifacts([{ path: "a.js", content: 1 }]), /path and text/i);
});

test("maps major language file extensions to the correct editor labels", () => {
  const files = [
    ["app.js", "javascript"], ["app.ts", "typescript"], ["app.py", "python"],
    ["Main.java", "java"], ["Main.kt", "kotlin"], ["index.html", "html"],
    ["styles.css", "css"], ["component.jsx", "jsx"], ["component.tsx", "tsx"],
    ["main.go", "go"], ["main.rs", "rust"], ["query.sql", "sql"]
  ].map(([path]) => ({ path, content: "sample" }));
  const formatted = formatCodeArtifacts(files);
  for (const [, language] of [
    ["app.js", "javascript"], ["app.ts", "typescript"], ["app.py", "python"],
    ["Main.java", "java"], ["Main.kt", "kotlin"], ["index.html", "html"],
    ["styles.css", "css"], ["component.jsx", "jsx"], ["component.tsx", "tsx"],
    ["main.go", "go"], ["main.rs", "rust"], ["query.sql", "sql"]
  ]) {
    assert.ok(formatted.includes("```" + language + "\nsample"), `Missing language label: ${language}`);
  }
});

test("optional extraction leaves ordinary coding prose unchanged", () => {
  assert.equal(parseCodeArtifactsIfPresent("Here is a function example."), null);
});

test("optional extraction returns validated file artifacts when declared", () => {
  const output = "--- DREAM FILE: src/app.js ---\nexport const ready = true;\n--- END DREAM FILE ---";
  assert.deepEqual(parseCodeArtifactsIfPresent(output), [
    { path: "src/app.js", content: "export const ready = true;\n" }
  ]);
});

test("optional extraction rejects malformed declared artifact bundles", () => {
  assert.throws(() => parseCodeArtifactsIfPresent("--- DREAM FILE: ../secret ---\nunsafe\n--- END DREAM FILE ---"), /relative POSIX paths/i);
});
