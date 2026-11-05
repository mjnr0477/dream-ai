import test from "node:test";
import assert from "node:assert/strict";
import { appendUserMessage } from "../public/chat-state.js";

test("retry does not append a duplicate user prompt", () => {
  const messages = [{ role: "user", content: "Explain recursion" }];
  const added = appendUserMessage(messages, "Explain recursion", true);
  assert.equal(added, false);
  assert.deepEqual(messages, [{ role: "user", content: "Explain recursion" }]);
});

test("a new prompt is appended normally", () => {
  const messages = [];
  const added = appendUserMessage(messages, "Explain recursion");
  assert.equal(added, true);
  assert.deepEqual(messages, [{ role: "user", content: "Explain recursion" }]);
});

test("retry appends the prompt if it is not already the latest user message", () => {
  const messages = [{ role: "user", content: "First prompt" }, { role: "assistant", content: "Temporary failure" }];
  const added = appendUserMessage(messages, "First prompt", true);
  assert.equal(added, true);
  assert.equal(messages.at(-1).content, "First prompt");
});
