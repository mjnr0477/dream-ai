import test from "node:test";
import assert from "node:assert/strict";
import { ToolRegistry } from "../lib/tool-registry.js";

const definition = {
  id: "lookup_status",
  description: "Read a status value from a trusted source.",
  risk: "read",
  permission: "status:read",
  timeoutMs: 1000,
  inputSchema: {
    type: "object",
    properties: { key: { type: "string", minLength: 1, maxLength: 40 } },
    required: ["key"],
    additionalProperties: false
  }
};
const principal = { id: "user-1", permissions: ["status:read"] };

test("registers and lists tool metadata without exposing schemas or handlers", () => {
  const registry = new ToolRegistry();
  registry.register(definition, async ({ input }) => ({ value: input.key }));
  assert.deepEqual(registry.list(), [{
    id: "lookup_status",
    description: definition.description,
    risk: "read",
    permission: "status:read",
    timeoutMs: 1000
  }]);
});

test("executes an authorized read tool and returns an execution id", async () => {
  const registry = new ToolRegistry();
  registry.register(definition, async ({ input, principal: actor }) => ({ value: input.key, actor: actor.id }));
  const result = await registry.execute({ toolId: definition.id, input: { key: "healthy" }, principal });
  assert.match(result.executionId, /^[0-9a-f-]{36}$/);
  assert.deepEqual(result.result, { value: "healthy", actor: "user-1" });
});

test("rejects unknown tools and unauthorized principals before handler execution", async () => {
  let called = false;
  const registry = new ToolRegistry();
  registry.register(definition, async () => { called = true; });
  await assert.rejects(registry.execute({ toolId: "missing_tool", input: {}, principal }), /not registered/);
  await assert.rejects(registry.execute({ toolId: definition.id, input: { key: "x" }, principal: { id: "u", permissions: [] } }), /not authorized/);
  assert.equal(called, false);
});

test("rejects malformed inputs and unknown properties", async () => {
  const registry = new ToolRegistry();
  registry.register(definition, async () => "should not run");
  await assert.rejects(registry.execute({ toolId: definition.id, input: { key: "x", extra: true }, principal }), /unexpected property/);
  await assert.rejects(registry.execute({ toolId: definition.id, input: {}, principal }), /missing key/);
});

test("requires approval bound to exact inputs for write and destructive tools", async () => {
  const registry = new ToolRegistry({
    approvalVerifier: async ({ approval, toolId, inputSha256, principalId }) =>
      approval?.token === "trusted-test-token" &&
      approval?.toolId === toolId &&
      approval?.inputSha256 === inputSha256 &&
      approval?.principalId === principalId
  });
  const write = { ...definition, id: "update_setting", risk: "write", permission: "settings:write" };
  registry.register(write, async ({ input }) => ({ saved: input.key }));
  const actor = { id: "u", permissions: ["settings:write"] };
  const input = { key: "safe-value" };
  await assert.rejects(registry.execute({ toolId: write.id, input, principal: actor }), /requires server-verified user approval/);
  const approval = { token: "trusted-test-token", toolId: write.id, inputSha256: registry.approvalDigest(write.id, input), principalId: actor.id };
  await assert.rejects(registry.execute({ toolId: write.id, input: { key: "different" }, principal: actor, approval }), /requires server-verified user approval/);
  const result = await registry.execute({ toolId: write.id, input, principal: actor, approval });
  assert.deepEqual(result.result, { saved: "safe-value" });
});

test("records outcomes without recording raw inputs or outputs", async () => {
  const events = [];
  const registry = new ToolRegistry({ audit: (event) => events.push(event) });
  registry.register(definition, async ({ input }) => ({ secret: input.key }));
  await registry.execute({ toolId: definition.id, input: { key: "do-not-log-this" }, principal });
  assert.equal(events.length, 1);
  assert.equal(events[0].status, "succeeded");
  assert.equal(JSON.stringify(events).includes("do-not-log-this"), false);
});

test("times out a slow handler and signals cancellation", async () => {
  const registry = new ToolRegistry();
  const short = { ...definition, timeoutMs: 100 };
  let aborted = false;
  registry.register(short, async ({ signal }) => new Promise(() => {
    signal.addEventListener("abort", () => { aborted = true; }, { once: true });
  }));
  await assert.rejects(registry.execute({ toolId: short.id, input: { key: "x" }, principal }), /timed out/);
  assert.equal(aborted, true);
});

test("enforces the concurrent execution limit", async () => {
  const registry = new ToolRegistry({ maxConcurrent: 1 });
  let release;
  registry.register(definition, async () => new Promise((resolve) => { release = resolve; }));
  const pending = registry.execute({ toolId: definition.id, input: { key: "x" }, principal });
  await new Promise((resolve) => setImmediate(resolve));
  await assert.rejects(registry.execute({ toolId: definition.id, input: { key: "y" }, principal }), /capacity reached/);
  release("done");
  await pending;
});

test("does not free concurrency capacity while a timed-out handler is still running", async () => {
  const registry = new ToolRegistry({ maxConcurrent: 1 });
  const short = { ...definition, timeoutMs: 100 };
  const releases = [];
  registry.register(short, async () => new Promise((resolve) => releases.push(resolve)));
  const first = registry.execute({ toolId: short.id, input: { key: "first" }, principal });
  await assert.rejects(first, /timed out/);
  assert.equal(releases.length, 1);
  await assert.rejects(
    registry.execute({ toolId: short.id, input: { key: "second" }, principal }),
    /capacity reached/
  );
  releases[0]("first finished");
  await new Promise((resolve) => setImmediate(resolve));
  const second = registry.execute({ toolId: short.id, input: { key: "second" }, principal });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(releases.length, 2);
  releases[1]("second finished");
  await second;
});

test("denies sensitive tools by default when no trusted approval verifier is configured", async () => {
  const registry = new ToolRegistry();
  const write = { ...definition, id: "delete_record", risk: "destructive", permission: "records:delete" };
  registry.register(write, async () => ({ deleted: true }));
  await assert.rejects(
    registry.execute({
      toolId: write.id, input: { key: "record-1" },
      principal: { id: "u", permissions: ["records:delete"] },
      approval: { confirmed: true, inputSha256: registry.approvalDigest(write.id, { key: "record-1" }) }
    }),
    /server-verified user approval/
  );
});

test("rejects malformed nested schemas during trusted tool registration", () => {
  const registry = new ToolRegistry();
  const invalid = {
    ...definition,
    id: "invalid_nested",
    inputSchema: {
      type: "object",
      properties: { options: { type: "array", items: { type: "object" } } },
      required: ["options"],
      additionalProperties: false
    }
  };
  assert.throws(() => registry.register(invalid, async () => ({})), /Object schemas require/);
});
