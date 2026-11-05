import { createHash, randomUUID } from "node:crypto";

const ID_PATTERN = /^[a-z][a-z0-9_]{1,63}$/;
const MAX_TOOLS = 100;
const MAX_INPUT_BYTES = 16 * 1024;
const MAX_OUTPUT_BYTES = 32 * 1024;

function stableJson(value) {
  if (Array.isArray(value)) return "[" + value.map(stableJson).join(",") + "]";
  if (value && typeof value === "object") {
    return "{" + Object.keys(value).sort().map((key) =>
      JSON.stringify(key) + ":" + stableJson(value[key])
    ).join(",") + "}";
  }
  return JSON.stringify(value);
}

function digest(value) {
  return createHash("sha256").update(stableJson(value)).digest("hex");
}

function byteLength(value) {
  return Buffer.byteLength(JSON.stringify(value), "utf8");
}

function validateSchema(value, schema, path = "$") {
  if (!schema || typeof schema !== "object" || Array.isArray(schema)) {
    throw new Error("Tool schema must be an object.");
  }
  const allowedTypes = ["object", "string", "number", "integer", "boolean", "array"];
  if (!allowedTypes.includes(schema.type)) throw new Error("Unsupported schema type.");
  const valid =
    schema.type === "object" ? value !== null && typeof value === "object" && !Array.isArray(value) :
    schema.type === "array" ? Array.isArray(value) :
    schema.type === "integer" ? Number.isSafeInteger(value) :
    schema.type === "number" ? typeof value === "number" && Number.isFinite(value) :
    typeof value === schema.type;
  if (!valid) throw new Error(`Invalid tool input at ${path}: expected ${schema.type}.`);

  if (schema.enum && (!Array.isArray(schema.enum) || !schema.enum.some((item) => item === value))) {
    throw new Error(`Invalid tool input at ${path}: value is not allowed.`);
  }
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
    if (schema.items) value.forEach((item, index) => validateSchema(item, schema.items, `${path}[${index}]`));
  }
  if (schema.type === "object") {
    const properties = schema.properties || {};
    const required = schema.required || [];
    for (const key of required) {
      if (!Object.hasOwn(value, key)) throw new Error(`Invalid tool input at ${path}: missing ${key}.`);
    }
    if (schema.additionalProperties !== true) {
      for (const key of Object.keys(value)) {
        if (!Object.hasOwn(properties, key)) throw new Error(`Invalid tool input at ${path}: unexpected property ${key}.`);
      }
    }
    for (const [key, childSchema] of Object.entries(properties)) {
      if (Object.hasOwn(value, key)) validateSchema(value[key], childSchema, `${path}.${key}`);
    }
  }
}

export class ToolRegistry {
  #tools = new Map();
  #active = 0;
  #audit;
  #maxConcurrent;

  constructor({ audit = () => {}, maxConcurrent = 3 } = {}) {
    if (typeof audit !== "function") throw new TypeError("audit must be a function.");
    if (!Number.isSafeInteger(maxConcurrent) || maxConcurrent < 1 || maxConcurrent > 20) {
      throw new TypeError("maxConcurrent must be an integer between 1 and 20.");
    }
    this.#audit = audit;
    this.#maxConcurrent = maxConcurrent;
  }

  register(definition, handler) {
    if (this.#tools.size >= MAX_TOOLS) throw new Error("Tool registry capacity reached.");
    if (!definition || typeof definition !== "object" || !ID_PATTERN.test(definition.id || "")) {
      throw new Error("Tool id must be a lowercase identifier.");
    }
    if (this.#tools.has(definition.id)) throw new Error("Tool id is already registered.");
    if (typeof definition.description !== "string" || !definition.description.trim() || definition.description.length > 500) {
      throw new Error("Tool description must contain 1 to 500 characters.");
    }
    if (!["read", "write", "destructive"].includes(definition.risk)) {
      throw new Error("Tool risk must be read, write or destructive.");
    }
    if (typeof definition.permission !== "string" || !/^[a-z][a-z0-9:_-]{1,79}$/.test(definition.permission)) {
      throw new Error("Tool permission is invalid.");
    }
    if (!Number.isSafeInteger(definition.timeoutMs) || definition.timeoutMs < 100 || definition.timeoutMs > 30000) {
      throw new Error("Tool timeout must be between 100 and 30000 ms.");
    }
    if (typeof handler !== "function") throw new TypeError("Tool handler must be a function.");
    validateSchema({}, definition.inputSchema);
    this.#tools.set(definition.id, Object.freeze({
      id: definition.id,
      description: definition.description,
      risk: definition.risk,
      permission: definition.permission,
      timeoutMs: definition.timeoutMs,
      inputSchema: structuredClone(definition.inputSchema)
    }));
  }

  list() {
    return [...this.#tools.values()].map(({ inputSchema, ...tool }) => ({ ...tool }));
  }

  approvalDigest(toolId, input) {
    if (!this.#tools.has(toolId)) throw new Error("Unknown tool.");
    return digest({ toolId, input });
  }

  async execute({ toolId, input, principal, approval } = {}) {
    const tool = this.#tools.get(toolId);
    const executionId = randomUUID();
    const startedAt = Date.now();
    const record = (status, extra = {}) => {
      try {
        this.#audit({
          executionId, toolId, status, at: new Date().toISOString(),
          durationMs: Date.now() - startedAt, ...extra
        });
      } catch {
        // Audit sinks must not leak exceptions into tool execution.
      }
    };

    if (!tool) {
      record("denied", { reason: "unknown-tool" });
      throw new Error("Tool is not registered.");
    }
    if (!principal || typeof principal.id !== "string" || !Array.isArray(principal.permissions)) {
      record("denied", { reason: "missing-principal" });
      throw new Error("An authenticated principal is required.");
    }
    if (!principal.permissions.includes(tool.permission)) {
      record("denied", { reason: "permission-denied" });
      throw new Error("Principal is not authorized for this tool.");
    }
    try {
      validateSchema(input, tool.inputSchema);
      if (byteLength(input) > MAX_INPUT_BYTES) throw new Error("Tool input exceeds the size limit.");
    } catch (error) {
      record("rejected", { reason: "invalid-input" });
      throw error;
    }

    if (tool.risk !== "read") {
      const expected = this.approvalDigest(toolId, input);
      if (approval?.confirmed !== true || approval?.inputSha256 !== expected) {
        record("approval-required", { reason: "missing-or-stale-approval" });
        throw new Error("This action requires explicit approval for these exact inputs.");
      }
    }
    if (this.#active >= this.#maxConcurrent) {
      record("rejected", { reason: "capacity" });
      throw new Error("Tool execution capacity reached. Retry shortly.");
    }

    this.#active += 1;
    const controller = new AbortController();
    let timeout;
    try {
      const result = await Promise.race([
        this.#handlers.get(toolId)({ input: structuredClone(input), principal: { id: principal.id }, signal: controller.signal, executionId }),
        new Promise((_, reject) => {
          timeout = setTimeout(() => {
            controller.abort();
            reject(new Error("Tool execution timed out."));
          }, tool.timeoutMs);
        })
      ]);
      if (byteLength(result) > MAX_OUTPUT_BYTES) throw new Error("Tool output exceeds the size limit.");
      record("succeeded");
      return { executionId, toolId, result };
    } catch (error) {
      record("failed", { reason: error.message === "Tool execution timed out." ? "timeout" : "handler-error" });
      throw error;
    } finally {
      clearTimeout(timeout);
      this.#active -= 1;
    }
  }

  #handlers = new Map();

  register(definition, handler) {
    if (this.#tools.size >= MAX_TOOLS) throw new Error("Tool registry capacity reached.");
    if (!definition || typeof definition !== "object" || !ID_PATTERN.test(definition.id || "")) throw new Error("Tool id must be a lowercase identifier.");
    if (this.#tools.has(definition.id)) throw new Error("Tool id is already registered.");
    if (typeof definition.description !== "string" || !definition.description.trim() || definition.description.length > 500) throw new Error("Tool description must contain 1 to 500 characters.");
    if (!["read", "write", "destructive"].includes(definition.risk)) throw new Error("Tool risk must be read, write or destructive.");
    if (typeof definition.permission !== "string" || !/^[a-z][a-z0-9:_-]{1,79}$/.test(definition.permission)) throw new Error("Tool permission is invalid.");
    if (!Number.isSafeInteger(definition.timeoutMs) || definition.timeoutMs < 100 || definition.timeoutMs > 30000) throw new Error("Tool timeout must be between 100 and 30000 ms.");
    if (typeof handler !== "function") throw new TypeError("Tool handler must be a function.");
    validateSchema({}, definition.inputSchema);
    this.#tools.set(definition.id, Object.freeze({ id: definition.id, description: definition.description, risk: definition.risk, permission: definition.permission, timeoutMs: definition.timeoutMs, inputSchema: structuredClone(definition.inputSchema) }));
    this.#handlers.set(definition.id, handler);
  }
}
