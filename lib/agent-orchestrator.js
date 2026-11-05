import { randomUUID } from "node:crypto";

const TASK_ID = /^[a-z][a-z0-9_-]{1,63}$/;
const MAX_TASKS = 20;
const MAX_STEPS = 40;
const MAX_ATTEMPTS = 3;
const TERMINAL = new Set(["completed", "failed", "blocked", "approval_required", "interrupted"]);

function jsonClone(value, label) {
  let encoded;
  try { encoded = JSON.stringify(value); } catch { throw new Error(label + " must be JSON serializable."); }
  if (typeof encoded !== "string" || Buffer.byteLength(encoded, "utf8") > 16 * 1024) {
    throw new Error(label + " must be JSON data no larger than 16 KiB.");
  }
  return JSON.parse(encoded);
}

function validatePlan(tasks) {
  if (!Array.isArray(tasks) || tasks.length < 1 || tasks.length > MAX_TASKS) {
    throw new Error("An agent plan must contain between 1 and 20 tasks.");
  }
  const byId = new Map();
  for (const task of tasks) {
    if (!task || typeof task !== "object" || Array.isArray(task) || !TASK_ID.test(task.id || "")) {
      throw new Error("Every task needs a valid unique id.");
    }
    if (byId.has(task.id)) throw new Error("Duplicate task id: " + task.id);
    if (typeof task.toolId !== "string" || !/^[a-z][a-z0-9_]{1,63}$/.test(task.toolId)) {
      throw new Error("Task " + task.id + " needs a valid registered tool id.");
    }
    if (!Array.isArray(task.dependsOn || []) || (task.dependsOn || []).some((id) => typeof id !== "string")) {
      throw new Error("Task dependencies must be an array of task ids.");
    }
    if (new Set(task.dependsOn || []).size !== (task.dependsOn || []).length) {
      throw new Error("Task " + task.id + " has duplicate dependencies.");
    }
    const maxAttempts = task.maxAttempts ?? 1;
    if (!Number.isSafeInteger(maxAttempts) || maxAttempts < 1 || maxAttempts > MAX_ATTEMPTS) {
      throw new Error("Task maxAttempts must be between 1 and 3.");
    }
    if (task.verify !== undefined && typeof task.verify !== "boolean") {
      throw new Error("Task verify must be a boolean.");
    }
    byId.set(task.id, { ...task, dependsOn: [...(task.dependsOn || [])], maxAttempts });
  }
  for (const task of byId.values()) {
    for (const dependency of task.dependsOn) {
      if (!byId.has(dependency)) throw new Error("Unknown dependency " + dependency + " in task " + task.id + ".");
      if (dependency === task.id) throw new Error("A task cannot depend on itself.");
    }
  }
  const visiting = new Set(), visited = new Set();
  function visit(id) {
    if (visiting.has(id)) throw new Error("Agent plan contains a dependency cycle.");
    if (visited.has(id)) return;
    visiting.add(id);
    for (const dependency of byId.get(id).dependsOn) visit(dependency);
    visiting.delete(id);
    visited.add(id);
  }
  for (const id of byId.keys()) visit(id);
  return byId;
}

export class AgentOrchestrator {
  #registry;
  #maxSteps;

  constructor({ registry, maxSteps = MAX_STEPS } = {}) {
    if (!registry || typeof registry.execute !== "function" || typeof registry.list !== "function") {
      throw new TypeError("A trusted ToolRegistry is required.");
    }
    if (!Number.isSafeInteger(maxSteps) || maxSteps < 1 || maxSteps > MAX_STEPS) {
      throw new TypeError("maxSteps must be between 1 and 40.");
    }
    this.#registry = registry;
    this.#maxSteps = maxSteps;
  }

  async run(tasks, {
    principal,
    approvalProvider,
    resolveInput = ({ task }) => task.input,
    verifyResult,
    signal
  } = {}) {
    const plan = validatePlan(tasks);
    if (!principal || typeof principal.id !== "string" || !Array.isArray(principal.permissions)) {
      throw new Error("An authenticated principal is required.");
    }
    if (typeof resolveInput !== "function") throw new TypeError("resolveInput must be a trusted function.");
    const runId = randomUUID();
    const states = new Map([...plan.keys()].map((id) => [id, {
      id, status: "pending", attempts: 0, executionIds: [], result: null, error: null
    }]));
    const results = new Map();
    let steps = 0;
    let stoppedReason = null;

    const snapshot = () => ({
      runId, status: stoppedReason ? "interrupted" :
        [...states.values()].every((s) => TERMINAL.has(s.status)) ? "finished" : "running",
      steps, tasks: [...states.values()].map((state) => ({ ...state }))
    });

    while (true) {
      if (signal?.aborted) {
        stoppedReason = "aborted";
        for (const state of states.values()) if (state.status === "pending") state.status = "interrupted";
        break;
      }
      if (steps >= this.#maxSteps) {
        stoppedReason = "step_limit";
        for (const state of states.values()) if (state.status === "pending") state.status = "blocked";
        break;
      }

      let progressed = false;
      for (const task of plan.values()) {
        const state = states.get(task.id);
        if (state.status !== "pending") continue;
        const dependencyStates = task.dependsOn.map((id) => states.get(id));
        if (dependencyStates.some((item) => item.status === "failed" || item.status === "blocked" ||
          item.status === "approval_required" || item.status === "interrupted")) {
          state.status = "blocked";
          state.error = "A dependency did not complete.";
          progressed = true;
          continue;
        }
        if (!dependencyStates.every((item) => item.status === "completed")) continue;
        if (signal?.aborted) break;
        if (steps >= this.#maxSteps) {
          stoppedReason = "step_limit";
          break;
        }

        progressed = true;
        steps += 1;
        state.status = "running";
        let lastError = null;
        const tool = this.#registry.list().find((item) => item.id === task.toolId);
        if (!tool) {
          state.status = "failed";
          state.error = "Tool is not registered.";
          continue;
        }
        // Side-effecting actions are never automatically retried.
        const attemptsAllowed = tool.risk === "read" ? task.maxAttempts : 1;
        for (let attempt = 1; attempt <= attemptsAllowed; attempt += 1) {
          state.attempts = attempt;
          try {
            const input = jsonClone(await resolveInput({
              task: { ...task }, completed: new Map(results), principal: { id: principal.id }
            }), "Resolved tool input");
            let approval;
            if (tool.risk !== "read") {
              if (typeof approvalProvider !== "function") {
                state.status = "approval_required";
                state.error = "A server-side approval provider is required for this action.";
                break;
              }
              approval = await approvalProvider({
                task: { id: task.id, toolId: task.toolId },
                input: jsonClone(input, "Approval input"),
                principalId: principal.id
              });
              if (!approval) {
                state.status = "approval_required";
                state.error = "User approval was not granted.";
                break;
              }
            }
            const execution = await this.#registry.execute({
              toolId: task.toolId, input, principal, approval
            });
            state.executionIds.push(execution.executionId);
            const result = jsonClone(execution.result, "Tool result");
            if (task.verify === true) {
              if (typeof verifyResult !== "function") {
                throw new Error("This task requires a trusted result verifier.");
              }
              const verification = await verifyResult({
                task: { ...task }, result, executionId: execution.executionId
              });
              if (!verification || verification.verified !== true) {
                throw new Error("Tool result did not pass verification.");
              }
              state.verification = jsonClone(verification, "Verification report");
            }
            state.result = result;
            state.status = "completed";
            results.set(task.id, result);
            lastError = null;
            break;
          } catch (error) {
            lastError = error;
            state.error = String(error?.message || "Tool execution failed.").slice(0, 300);
            if (attempt < attemptsAllowed) state.status = "retrying";
          }
        }
        if (state.status === "running" || state.status === "retrying") {
          state.status = lastError ? "failed" : "completed";
        }
        if (signal?.aborted) {
          stoppedReason = "aborted";
          break;
        }
      }
      if (stoppedReason) {
        const pendingStatus = stoppedReason === "step_limit" ? "blocked" : "interrupted";
        for (const state of states.values()) if (state.status === "pending") state.status = pendingStatus;
        break;
      }
      if ([...states.values()].every((state) => TERMINAL.has(state.status))) break;
      if (!progressed) {
        for (const state of states.values()) if (state.status === "pending") {
          state.status = "blocked";
          state.error = "Task dependencies could not be satisfied.";
        }
        break;
      }
    }

    const report = snapshot();
    report.completed = report.tasks.filter((task) => task.status === "completed").length;
    report.failed = report.tasks.filter((task) => task.status === "failed").length;
    report.blocked = report.tasks.filter((task) => task.status === "blocked").length;
    report.approvalRequired = report.tasks.filter((task) => task.status === "approval_required").length;
    report.interrupted = report.tasks.filter((task) => task.status === "interrupted").length;
    if (stoppedReason) report.reason = stoppedReason;
    return report;
  }
}
