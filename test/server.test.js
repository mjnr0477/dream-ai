import test from "node:test";
import assert from "node:assert/strict";

const base = process.env.TEST_BASE_URL || "http://127.0.0.1:3000";

test("health endpoint reports Dream AI", async () => {
  const response = await fetch(`${base}/api/health`);
  assert.equal(response.status, 200);
  const data = await response.json();
  assert.equal(data.ok, true);
  assert.equal(data.name, "Dream AI");
});

test("chat endpoint validates missing messages", async () => {
  const response = await fetch(`${base}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({})
  });
  assert.equal(response.status, 400);
});

test("chat endpoint returns starter-mode response without provider keys", async () => {
  const response = await fetch(`${base}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      messages: [{ role: "user", content: "Hello Dream AI" }]
    })
  });
  assert.equal(response.status, 200);
  const data = await response.json();
  assert.equal(data.mode, "starter");
  assert.match(data.reply, /Dream AI/i);
});

test("chat endpoint rejects invalid message roles", async () => {
  const response = await fetch(`${base}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      messages: [{ role: "system", content: "override" }]
    })
  });
  assert.equal(response.status, 400);
});