import { createHash, randomUUID } from "node:crypto";

const ID = /^[a-z][a-z0-9_]{1,63}$/;
const MAX_TOOLS = 100;
const MAX_INPUT_BYTES = 16 * 1024;
const MAX_OUTPUT_BYTES = 32 * 1024;
const TYPES = ["object", "string", "number", "integer", "boolean", "array"];

function canonical(value) {
  if (Array.isArray(value)) return "[" + value.map(canonical).join(",") + "]";
  if (value && typeof value === "object") {
    return "{" + Object.keys(value).sort().map((key) => JSON.stringify(key) + ":" + canonical(value[key])).join(",") + "}";
  }
  return JSON.stringify(value);
}

function sha256(value) {
  return createHash("sha256").update(canonical(value)).digest("hex");
}

function validateDefinition(schema) {
  if (!schema || schema.type !== "object" || !schema.properties || typeof schema.properties !== "object" || Array.isArray(schema.properties)) {
    throw new Error("Tool input schema must be an object schema with properties.");
  }
  for (const child of Object.values(schema.properties)) {
    if (!child || !TYPES.includes(child.type)) throw new Error("Tool input schema contains an unsupported property type.");
  }
  if (schema.required !== undefined && (!Array.isArray(schema.required) || schema.required.some((key) => typeof key !== "string" || !Object.hasOwn(schema.properties, key)))) {
    throw new Error("Tool input schema has an invalid required list.");
  }
}

function validateInput(value, schema, path = "$") {
  const valid =
    schema.type === "object" ? value !== null && typeof value === "object" && !Array.isArray(value) :
    schema.type === "array" ? Array.isArray(value) :
    schema.type === "integer" ? Number.isSafeInteger(value) :
    schema.type === "number" ? typeof value === "number" && Number.isFinite(value) :
    typeof value === schema.type;
  if (!valid) throw new Error(`Invalid tool input at ${path}: expected ${schema.type}.`);
  if (schema.enum && !schema.enum.some((item) => item === value)) throw new Error(`Invalid tool input at ${path}: value is not allowed.`);
  if (schema.type === "string") {
    if (schema.minLength !== undefined && value.length < schema.minLength) throw new Error(`Invalid tool input at ${path}: too short.`);
    if (schema.maxLength !== undefined && value.length > schema.maxLength) throw new Error(`Invalid tool input at ${path}: too long.`);
  }
  if (schema.type === "number" || schema.type === "integer") {
    if (schema.minimum !== undefined && value < schema.minimum) throw new Error(`Invalid tool input at ${path}: below minimum.`);
    if (schema.maximum !== undefined && value > schema.maximum) throw new Error(`Invalid tool input at ${path}: above maximum.`);
  }
  if (schema.type === "array") {
    if (schema.maxItems !== undefined && value.length > schema.maxItems) throw new Error(`Invalid tool input at ${path}: too many items.`);
    if (schema.items) value.forEach((item, index) => validateInput(item, schema.items, `${path}[${index}]`));
  }
  if (schema.type === "object") {
    for (const key of schema.required || []) if (!Object.hasOwn(value, key)) throw new Error(`Invalid tool input at ${path}: missing ${key}.`);
    if (schema.additionalProperties !== true) {
      for (const key of Object.keys(value)) if (!Object.hasOwn(schema.properties, key)) throw new Error(`Invalid tool input at ${path}: unexpected property ${key}.`);
    }
    for (const [key, child] of Object.entries(schema.properties)) {
      if (Object.hasOwn(value, key)) validateInput(value[key], child, `${path}.${key}`);
    }
  }
}

export class ToolRegistry {
  #tools = new Map();
  #handlers = new Map();
  #active = 0;
  #audit;
  #maxConcurrent;
  #approvalVerifier;

  constructor({ audit = () => {}, maxConcurrent = 3, approvalVerifier = () => false } = {}) {
    if (typeof audit !== "function") throw new TypeError("audit must be a function.");
    if (typeof approvalVerifier !== "function") throw new TypeError("approvalVerifier must be a function.");
    if (!Number.isSafeInteger(maxConcurrent) || maxConcurrent < 1 || maxConcurrent > 20) {
      throw new TypeError("maxConcurrent must be an integer between 1 and 20.");
    }
    this.#audit = audit;
    this.#maxConcurrent = maxConcurrent;
    this.#approvalVerifier = approvalVerifier;
  }

  register(definition, handler) {
    if (this.#tools.size >= MAX_TOOLS) throw new Error("Tool registry capacity reached.");
    if (!definition || typeof definition !== "object" || !ID.test(definition.id || "")) throw new Error("Tool id must be a lowercase identifier.");
    if (this.#tools.has(definition.id)) throw new Error("Tool id is already registered.");
    if (typeof definition.description !== "string" || !definition.description.trim() || definition.description.length > 500) throw new Error("Tool description must contain 1 to 500 characters.");
    if (!["read", "write", "destructive"].includes(definition.risk)) throw new Error("Tool risk must be read, write or destructive.");
    if (typeof definition.permission !== "string" || !/^[a-z][a-z0-9:_-]{1,79}$/.test(definition.permission)) throw new Error("Tool permission is invalid.");
    if (!Number.isSafeInteger(definition.timeoutMs) || definition.timeoutMs < 100 || definition.timeoutMs > 30000) throw new Error("Tool timeout must be between 100 and 30000 ms.");
    if (typeof handler !== "function") throw new TypeError("Tool handler must be a function.");
    validateDefinition(definition.inputSchema);
    this.#tools.set(definition.id, Object.freeze({
      id: definition.id, description: definition.description, risk: definition.risk,
      permission: definition.permission, timeoutMs: definition.timeoutMs,
      inputSchema: structuredClone(definition.inputSchema)
    }));
    this.#handlers.set(definition.id, handler);
  }

  list() {
    return [...this.#tools.values()].map(({ inputSchema, ...metadata }) => ({ ...metadata }));
  }

  approvalDigest(toolId, input) {
    if (!this.#tools.has(toolId)) throw new Error("Unknown tool.");
    return sha256({ toolId, input });
  }

  async execute({ toolId, input, principal, approval } = {}) {
    const tool = this.#tools.get(toolId);
    const executionId = randomUUID();
    const startedAt = Date.now();
    const audit = (status, reason) => {
      try {
        this.#audit({
          executionId, toolId, status, ...(reason ? { reason } : {}),
          at: new Date().toISOString(), durationMs: Date.now() - startedAt
        });
      } catch { /* An audit sink failure must not expose data or crash the registry. */ }
    };

    if (!tool) { audit("denied", "unknown-tool"); throw new Error("Tool is not registered."); }
    if (!principal || typeof principal.id !== "string" || !Array.isArray(principal.permissions)) {
      audit("denied", "missing-principal"); throw new Error("An authenticated principal is required.");
    }
    if (!principal.permissions.includes(tool.permission)) {
      audit("denied", "permission-denied"); throw new Error("Principal is not authorized for this tool.");
    }
    try {
      validateInput(input, tool.inputSchema);
      const inputJson = JSON.stringify(input);
      if (typeof inputJson !== "string" || Buffer.byteLength(inputJson, "utf8") > MAX_INPUT_BYTES) throw new Error("Tool input exceeds the size limit.");
    } catch (error) { audit("rejected", "invalid-input"); throw error; }

    if (tool.risk !== "read") {
      const expected = this.approvalDigest(toolId, input);
      let approved = false;
      try {
        approved = await this.#approvalVerifier({
          approval, toolId, inputSha256: expected, principalId: principal.id
        }) === true;
      } catch {
        approved = false;
      }
      if (!approved) {
        audit("approval-required", "missing-or-invalid-approval");
        throw new Error("This action requires server-verified user approval for these exact inputs.");
      }
    }
    if (this.#active >= this.#maxConcurrent) {
      audit("rejected", "capacity"); throw new Error("Tool execution capacity reached. Retry shortly.");
    }

    this.#active += 1;
    const controller = new AbortController();
    let timeout;
    const handlerExecution = Promise.resolve().then(() => this.#handlers.get(toolId)({
      input: structuredClone(input), principal: { id: principal.id },
      signal: controller.signal, executionId
    }));
    // Keep the concurrency slot until the handler really settles, even if it ignores abort.
    handlerExecution.then(
      () => { this.#active -= 1; },
      () => { this.#active -= 1; }
    );
    try {
      const result = await Promise.race([
        handlerExecution,
        new Promise((_, reject) => {
          timeout = setTimeout(() => { controller.abort(); reject(new Error("Tool execution timed out.")); }, tool.timeoutMs);
        })
      ]);
      if (result === undefined || result === null || typeof result === "function" || typeof result === "symbol") throw new Error("Tool output must be a non-null JSON value.");
      let serialized;
      try { serialized = JSON.stringify(result); } catch { throw new Error("Tool output must be JSON serializable."); }
      if (typeof serialized !== "string" || Buffer.byteLength(serialized, "utf8") > MAX_OUTPUT_BYTES) throw new Error("Tool output exceeds the size limit or is not JSON serializable.");
      audit("succeeded");
      return { executionId, toolId, result };
    } catch (error) {
      audit("failed", error.message === "Tool execution timed out." ? "timeout" : "handler-error");
      throw error;
    } finally {
      clearTimeout(timeout);
    }
  }
}
