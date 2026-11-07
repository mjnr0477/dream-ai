import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const app = await readFile(new URL("../public/app.js", import.meta.url), "utf8");
const styles = await readFile(new URL("../public/styles.css", import.meta.url), "utf8");

test("assistant answers place copy and opt-in training actions below the answer in its content column", () => {
  assert.match(app, /main\.className = "message-main"/);
  assert.match(app, /main\.append\(body\)/);
  assert.match(app, /host\.append\(tools, note\)/);
  assert.match(app, /appendAssistantActions\(replyNode, data\.reply, content, requestMode\)/);
  assert.match(app, /share\.textContent = "Share with AI"/);
  assert.match(app, /LEARNING_CONSENT_VERSION/);
});

test("desktop and mobile answer surfaces use the available conversation width", () => {
  assert.match(styles, /\.topbar,\.conversation,\.composer-wrap\{max-width:1120px\}/);
  assert.match(styles, /\.message-content\{display:block;width:100%;max-width:none/);
  assert.match(styles, /@media\(max-width:680px\)/);
});
