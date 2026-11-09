import test from "node:test";
import assert from "node:assert/strict";
import { createProjectZip } from "./project-archive.js";

function readU16(view, offset) { return view.getUint16(offset, true); }
function readU32(view, offset) { return view.getUint32(offset, true); }

test("creates a classic ZIP archive with local records, central directory and end record", async () => {
  const blob = createProjectZip([
    { path: "src/index.js", content: "export const ok = true;\n" },
    { path: "README.md", content: "# Demo\n" }
  ]);
  assert.equal(blob.type, "application/zip");
  const bytes = new Uint8Array(await blob.arrayBuffer());
  const view = new DataView(bytes.buffer);
  assert.equal(readU32(view, 0), 0x04034b50);
  assert.equal(readU16(view, 6), 0x0800, "UTF-8 filename flag");
  const firstNameLength = readU16(view, 26);
  const firstDataLength = readU32(view, 18);
  assert.equal(new TextDecoder().decode(bytes.slice(30, 30 + firstNameLength)), "src/index.js");
  const firstDataOffset = 30 + firstNameLength;
  assert.equal(new TextDecoder().decode(bytes.slice(firstDataOffset, firstDataOffset + firstDataLength)), "export const ok = true;\n");
  const centralOffset = bytes.length - 22 - (46 + 12) - (46 + 9);
  assert.equal(readU32(view, centralOffset), 0x02014b50);
  assert.equal(readU32(view, bytes.length - 22), 0x06054b50);
  assert.equal(readU16(view, bytes.length - 14), 2);
});

test("stores the standard CRC-32 for a known payload", async () => {
  const blob = createProjectZip([{ path: "check.txt", content: "123456789" }]);
  const view = new DataView(await blob.arrayBuffer());
  assert.equal(readU32(view, 14), 0xcbf43926);
});

test("preserves Unicode filenames and file contents", async () => {
  const content = "export const city = 'Bungoma';\n";
  const blob = createProjectZip([{ path: "src/こんにちは.js", content }]);
  const bytes = new Uint8Array(await blob.arrayBuffer());
  const view = new DataView(bytes.buffer);
  const nameLength = readU16(view, 26);
  assert.equal(new TextDecoder().decode(bytes.slice(30, 30 + nameLength)), "src/こんにちは.js");
  const contentLength = readU32(view, 18);
  const contentOffset = 30 + nameLength;
  assert.equal(new TextDecoder().decode(bytes.slice(contentOffset, contentOffset + contentLength)), content);
});

test("rejects unsafe paths, duplicate names and malformed entries", () => {
  for (const path of ["/etc/passwd", "../secret", "src/../../secret", "src\\app.js", "C:/temp/app.js", "src//app.js", "./app.js"]) {
    assert.throws(() => createProjectZip([{ path, content: "x" }]), /path|relative|normalized|unsafe/i, path);
  }
  assert.throws(() => createProjectZip([
    { path: "src/app.js", content: "one" },
    { path: "src/app.js", content: "two" }
  ]), /duplicate/i);
  assert.throws(() => createProjectZip([{ path: "a.js", content: 5 }]), /text content/i);
  assert.throws(() => createProjectZip([]), /at least one/i);
});

test("enforces project file and byte limits", () => {
  assert.throws(() => createProjectZip([
    { path: "a.js", content: "1234" },
    { path: "b.js", content: "5678" }
  ], { maxFiles: 1 }), /file count/i);
  assert.throws(() => createProjectZip([{ path: "a.js", content: "1234" }], { maxFileBytes: 3 }), /per-file/i);
  assert.throws(() => createProjectZip([
    { path: "a.js", content: "1234" },
    { path: "b.js", content: "5678" }
  ], { maxTotalBytes: 7 }), /total size/i);
  assert.throws(() => createProjectZip([{ path: "a.js", content: "x" }], { maxFiles: 0 }), /limit/i);
});
