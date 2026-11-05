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

test("requires one-time, expiring approval bound to exact inputs and principal", async () => {
  const registry = new ToolRegistry();
  const write = { ...definition, id: "update_setting", risk: "write", permission: "settings:write" };
  registry.register(write, async ({ input }) => ({ saved: input.key }));
  const actor = { id: "u", permissions: ["settings:write"] };
  const input = { key: "safe-value" };

  await assert.rejects(
    registry.issueApproval({ toolId: write.id, input, principal: actor }),
    /confirmation is required/
  );
  const approval = registry.issueApproval({ toolId: write.id, input, principal: actor, confirmed: true });
  assert.equal(typeof approval.token, "string");
  assert.ok(approval.token.length >= 40);

  await assert.rejects(
    registry.execute({ toolId: write.id, input: { key: "different" }, principal: actor, approval }),
    /fresh, explicit approval/
  );
  await assert.rejects(
    registry.execute({ toolId: write.id, input, principal: { id: "other", permissions: ["settings:write"] }, approval }),
    /fresh, explicit approval/
  );

  const freshApproval = registry.issueApproval({ toolId: write.id, input, principal: actor, confirmed: true });
  const result = await registry.execute({ toolId: write.id, input, principal: actor, approval: freshApproval });
  assert.deepEqual(result.result, { saved: "safe-value" });
  await assert.rejects(
    registry.execute({ toolId: write.id, input, principal: actor, approval: freshApproval }),
    /fresh, explicit approval/
  );
});

test("expires approval tokens before execution", async () => {
  let now = 1000;
  const registry = new ToolRegistry({ now: () => now, approvalTtlMs: 100 });
  const write = { ...definition, id: "update_setting", risk: "write", permission: "settings:write" };
  registry.register(write, async () => ({ saved: true }));
  const actor = { id: "u", permissions: ["settings:write"] };
  const input = { key: "safe-value" };
  const approval = registry.issueApproval({ toolId: write.id, input, principal: actor, confirmed: true });
  now += 101;
  await assert.rejects(
    registry.execute({ toolId: write.id, input, principal: actor, approval }),
    /fresh, explicit approval/
  );
});

test("validates nested schemas and rejects malformed or excessively deep schemas", () => {
  const registry = new ToolRegistry();
  const nested = {
    ...definition,
    id: "nested_lookup",
    inputSchema: {
      type: "object",
      properties: {
        filters: {
          type: "array",
          maxItems: 3,
          items: {
            type: "object",
            properties: { enabled: { type: "boolean" } },
            required: ["enabled"],
            additionalProperties: false
          }
        }
      },
      required: ["filters"],
      additionalProperties: false
    }
  };
  registry.register(nested, async ({ input }) => ({ count: input.filters.length }));
  assert.throws(() => registry.register({
    ...definition,
    id: "broken_array",
    inputSchema: { type: "object", properties: { values: { type: "array" } } }
  }, async () => ({})), /items schema/);

  let deep = { type: "string" };
  for (let index = 0; index < 10; index += 1) {
    deep = { type: "array", items: deep };
  }
  assert.throws(() => registry.register({
    ...definition,
    id: "too_deep",
    inputSchema: { type: "object", properties: { value: deep } }
  }, async () => ({})), /nested too deeply/);
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

test("times out a cooperative handler, signals cancellation and releases capacity", async () => {
  const registry = new ToolRegistry({ maxConcurrent: 1 });
  const short = { ...definition, timeoutMs: 100 };
  let aborted = false;
  registry.register(short, async ({ signal }) => new Promise((resolve) => {
    signal.addEventListener("abort", () => { aborted = true; resolve("late"); }, { once: true });
  }));
  await assert.rejects(registry.execute({ toolId: short.id, input: { key: "x" }, principal }), /timed out/);
  assert.equal(aborted, true);
  await new Promise((resolve) => setImmediate(resolve));
  const next = await registry.execute({ toolId: short.id, input: { key: "next" }, principal });
  assert.deepEqual(next.result, "late");
});

test("a non-cooperative timed-out handler retains its slot until it actually settles", async () => {
  const registry = new ToolRegistry({ maxConcurrent: 1 });
  const short = { ...definition, timeoutMs: 100 };
  let release;
  let signal;
  registry.register(short, async ({ signal: receivedSignal }) => {
    signal = receivedSignal;
    return new Promise((resolve) => { release = resolve; });
  });

  await assert.rejects(
    registry.execute({ toolId: short.id, input: { key: "slow" }, principal }),
    /timed out/
  );
  assert.equal(signal.aborted, true);
  await assert.rejects(
    registry.execute({ toolId: short.id, input: { key: "blocked" }, principal }),
    /capacity reached/
  );

  release("late");
  await new Promise((resolve) => setImmediate(resolve));
  const next = await registry.execute({ toolId: short.id, input: { key: "available" }, principal });
  assert.deepEqual(next.result, "late");
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
