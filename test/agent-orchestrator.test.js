import test from "node:test";
import assert from "node:assert/strict";
import { AgentOrchestrator } from "../lib/agent-orchestrator.js";

const principal = { id: "agent-test", permissions: ["read:data", "write:data"] };
const readTool = { id: "read_data", risk: "read", permission: "read:data" };
const writeTool = { id: "write_data", risk: "write", permission: "write:data" };

function registry({ readHandler, writeHandler, failRead = 0 } = {}) {
  let failures = 0;
  return {
    list: () => [readTool, writeTool],
    execute: async ({ toolId, input, approval }) => {
      if (toolId === "write_data" && !approval) throw new Error("approval required");
      if (toolId === "read_data" && failures++ < failRead) throw new Error("temporary failure");
      const result = toolId === "read_data"
        ? await (readHandler || (async () => ({ value: "read" })))(input)
        : await (writeHandler || (async () => ({ saved: true })))(input);
      return { executionId: "execution-" + failures, result };
    }
  };
}

test("executes a multi-step dependency plan and passes prior results to later tasks", async () => {
  const runner = new AgentOrchestrator({ registry: registry() });
  const report = await runner.run([
    { id: "first", toolId: "read_data", input: { query: "x" } },
    { id: "second", toolId: "read_data", dependsOn: ["first"], input: {} }
  ], {
    principal,
    resolveInput: ({ task, completed }) => task.id === "second"
      ? { previous: completed.get("first").value }
      : task.input
  });
  assert.equal(report.status, "finished");
  assert.equal(report.completed, 2);
  assert.deepEqual(report.tasks[1].result, { value: "read" });
});

test("retries transient read failures only within the configured bound", async () => {
  const runner = new AgentOrchestrator({ registry: registry({ failRead: 2 }) });
  const report = await runner.run([
    { id: "lookup", toolId: "read_data", input: {}, maxAttempts: 3 }
  ], { principal });
  assert.equal(report.tasks[0].status, "completed");
  assert.equal(report.tasks[0].attempts, 3);
});

test("never automatically retries side-effecting tools", async () => {
  let calls = 0;
  const runner = new AgentOrchestrator({
    registry: registry({ writeHandler: async () => { calls += 1; throw new Error("uncertain write"); } })
  });
  const report = await runner.run([
    { id: "write", toolId: "write_data", input: {}, maxAttempts: 3 }
  ], { principal, approvalProvider: async () => ({ token: "server-issued" }) });
  assert.equal(report.tasks[0].status, "failed");
  assert.equal(report.tasks[0].attempts, 1);
  assert.equal(calls, 1);
});

test("requires approval for write tasks and blocks downstream tasks when denied", async () => {
  const runner = new AgentOrchestrator({ registry: registry() });
  const report = await runner.run([
    { id: "write", toolId: "write_data", input: {} },
    { id: "after", toolId: "read_data", dependsOn: ["write"], input: {} }
  ], { principal, approvalProvider: async () => null });
  assert.equal(report.approvalRequired, 1);
  assert.equal(report.tasks[1].status, "blocked");
});

test("rejects invalid arguments, duplicate ids, missing dependencies and cycles", async () => {
  const runner = new AgentOrchestrator({ registry: registry() });
  await assert.rejects(runner.run([{ id: "bad", toolId: "missing", input: undefined }], { principal }), /JSON data/);
  await assert.rejects(runner.run([
    { id: "same", toolId: "read_data", input: {} },
    { id: "same", toolId: "read_data", input: {} }
  ], { principal }), /Duplicate task id/);
  await assert.rejects(runner.run([
    { id: "one", toolId: "read_data", dependsOn: ["missing"], input: {} }
  ], { principal }), /Unknown dependency/);
  await assert.rejects(runner.run([
    { id: "one", toolId: "read_data", dependsOn: ["two"], input: {} },
    { id: "two", toolId: "read_data", dependsOn: ["one"], input: {} }
  ], { principal }), /cycle/);
});

test("enforces result verification when a task requests it", async () => {
  const runner = new AgentOrchestrator({ registry: registry() });
  const failed = await runner.run([
    { id: "verify", toolId: "read_data", input: {}, verify: true }
  ], { principal, verifyResult: async () => ({ verified: false }) });
  assert.equal(failed.tasks[0].status, "failed");
  const passed = await runner.run([
    { id: "verify", toolId: "read_data", input: {}, verify: true }
  ], { principal, verifyResult: async () => ({ verified: true, evidence: "test" }) });
  assert.equal(passed.tasks[0].status, "completed");
});

test("records interruption between tasks without starting later work", async () => {
  const controller = new AbortController();
  const runner = new AgentOrchestrator({
    registry: registry({ readHandler: async () => {
      controller.abort();
      return { ok: true };
    } })
  });
  const report = await runner.run([
    { id: "first", toolId: "read_data", input: {} },
    { id: "second", toolId: "read_data", input: {} }
  ], { principal, signal: controller.signal });
  assert.equal(report.status, "interrupted");
  assert.equal(report.tasks[0].status, "completed");
  assert.equal(report.tasks[1].status, "interrupted");
});

test("counts every retry as an execution step and stops at the global cap", async () => {
  let calls = 0;
  const base = registry({ failRead: 5 });
  const counted = { list: base.list, execute: async (args) => { calls += 1; return base.execute(args); } };
  const runner = new AgentOrchestrator({ registry: counted, maxSteps: 2 });
  const report = await runner.run([
    { id: "lookup", toolId: "read_data", input: {}, maxAttempts: 3 },
    { id: "later", toolId: "read_data", input: {} }
  ], { principal });
  assert.equal(calls, 2);
  assert.equal(report.steps, 2);
  assert.equal(report.reason, "step_limit");
  assert.equal(report.tasks[0].attempts, 2);
});

test("cancellation during input resolution prevents tool execution", async () => {
  const controller = new AbortController();
  let calls = 0;
  let release;
  const pending = new Promise(resolve => { release = resolve; });
  const base = registry();
  const guarded = { list: base.list, execute: async (args) => { calls += 1; return base.execute(args); } };
  const runner = new AgentOrchestrator({ registry: guarded });
  const running = runner.run([
    { id: "lookup", toolId: "read_data", input: {} }
  ], { principal, signal: controller.signal, resolveInput: async ({ task }) => { await pending; return task.input; } });
  controller.abort();
  release();
  const report = await running;
  assert.equal(calls, 0);
  assert.equal(report.status, "interrupted");
  assert.equal(report.tasks[0].status, "interrupted");
});

test("cancellation during approval prevents side-effecting tool execution", async () => {
  const controller = new AbortController();
  let calls = 0;
  let release;
  const pending = new Promise(resolve => { release = resolve; });
  const base = registry();
  const guarded = { list: base.list, execute: async (args) => { calls += 1; return base.execute(args); } };
  const runner = new AgentOrchestrator({ registry: guarded });
  const running = runner.run([
    { id: "write", toolId: "write_data", input: {} }
  ], { principal, signal: controller.signal, approvalProvider: async () => { await pending; return { token: "approved" }; } });
  await Promise.resolve();
  controller.abort();
  release();
  const report = await running;
  assert.equal(calls, 0);
  assert.equal(report.status, "interrupted");
  assert.equal(report.tasks[0].status, "interrupted");
});

test("preflights trusted result verification before any tool can execute", async () => {
  let calls = 0;
  const base = registry();
  const guarded = { list: base.list, execute: async (args) => { calls += 1; return base.execute(args); } };
  const runner = new AgentOrchestrator({ registry: guarded });
  await assert.rejects(runner.run([
    { id: "first", toolId: "read_data", input: {} },
    { id: "second", toolId: "read_data", input: {}, verify: true }
  ], { principal }), /trusted result verifier is required before running/i);
  assert.equal(calls, 0);
});

test("caps the number of executable steps", async () => {
  const runner = new AgentOrchestrator({ registry: registry(), maxSteps: 1 });
  const report = await runner.run([
    { id: "one", toolId: "read_data", input: {} },
    { id: "two", toolId: "read_data", input: {} }
  ], { principal });
  assert.equal(report.steps, 1);
  assert.equal(report.reason, "step_limit");
});

test("preflights every tool and permission before executing the first task", async () => {
  let calls = 0;
  const base = registry();
  const guarded = {
    list: base.list,
    execute: async (args) => { calls += 1; return base.execute(args); }
  };
  const runner = new AgentOrchestrator({ registry: guarded });
  await assert.rejects(runner.run([
    { id: "first", toolId: "read_data", input: {} },
    { id: "second", toolId: "not_registered", input: {} }
  ], { principal }), /unregistered tool/);
  await assert.rejects(runner.run([
    { id: "first", toolId: "read_data", input: {} },
    { id: "second", toolId: "write_data", input: {} }
  ], { principal: { id: "limited", permissions: ["read:data"] } }), /not authorized/);
  assert.equal(calls, 0);
});
