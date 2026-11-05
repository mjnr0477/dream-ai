import test from "node:test";
import assert from "node:assert/strict";
import { appendUserMessage } from "../public/chat-state.js";

test("retry does not append a duplicate user prompt", () => {
  const messages = [{ role: "user", content: "Explain recursion" }];
  assert.equal(appendUserMessage(messages, "Explain recursion", true), false);
  assert.deepEqual(messages, [{ role: "user", content: "Explain recursion" }]);
});

test("a new prompt is appended normally", () => {
  const messages = [];
  assert.equal(appendUserMessage(messages, "Explain recursion"), true);
  assert.deepEqual(messages, [{ role: "user", content: "Explain recursion" }]);
});

test("retry appends the prompt if it is not already the latest user message", () => {
  const messages = [{ role: "user", content: "First prompt" }, { role: "assistant", content: "Temporary failure" }];
  assert.equal(appendUserMessage(messages, "First prompt", true), true);
  assert.equal(messages.at(-1).content, "First prompt");
});
