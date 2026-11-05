import { ToolRegistry } from "./tool-registry.js";
import { searchPublicKnowledge } from "./research.js";
import { searchMultiSourceResearch } from "./research-multisource.js";

const PUBLIC_RESEARCH_PRINCIPAL = Object.freeze({
  id: "public-research-route",
  permissions: Object.freeze(["research:public"])
});

function auditToolExecution(event) {
  // ToolRegistry emits metadata only; never log query text or tool inputs.
  console.info("DREAM tool audit", JSON.stringify(event));
}

export function createDreamToolRegistry({
  fetchImpl = fetch,
  audit = auditToolExecution,
  maxConcurrent = 2
} = {}) {
  const registry = new ToolRegistry({ audit, maxConcurrent });
  registry.register({
    id: "public_knowledge_search",
    description: "Search Wikipedia's public API and return up to five source-linked snippets.",
    risk: "read",
    permission: "research:public",
    timeoutMs: 8000,
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", minLength: 1, maxLength: 200 }
      },
      required: ["query"],
      additionalProperties: false
    }
  }, async ({ input }) => searchPublicKnowledge(input.query, fetchImpl));
  registry.register({
    id: "multi_source_scholarly_research",
    description: "Search Crossref and OpenAlex for scholarly publication metadata, with provider provenance and publication freshness labels.",
    risk: "read",
    permission: "research:public",
    timeoutMs: 15000,
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", minLength: 1, maxLength: 200 }
      },
      required: ["query"],
      additionalProperties: false
    }
  }, async ({ input }) => searchMultiSourceResearch(input.query, fetchImpl));
  return registry;
}

export const dreamToolRegistry = createDreamToolRegistry();

export async function runPublicResearch(query) {
  const execution = await dreamToolRegistry.execute({
    toolId: "public_knowledge_search",
    input: { query },
    principal: PUBLIC_RESEARCH_PRINCIPAL
  });
  return execution.result;
}

export function listDreamTools() {
  return dreamToolRegistry.list();
}
