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

export async function runPublicResearch(query, registry = dreamToolRegistry) {
  const tasks = await Promise.allSettled([
    registry.execute({
      toolId: "public_knowledge_search",
      input: { query },
      principal: PUBLIC_RESEARCH_PRINCIPAL
    }),
    registry.execute({
      toolId: "multi_source_scholarly_research",
      input: { query },
      principal: PUBLIC_RESEARCH_PRINCIPAL
    })
  ]);
  const encyclopedia = tasks[0].status === "fulfilled" ? tasks[0].value.result : null;
  const scholarlyResearch = tasks[1].status === "fulfilled" ? tasks[1].value.result : null;
  if (!encyclopedia && !scholarlyResearch) {
    throw new Error("All public research providers are unavailable.");
  }
  return {
    ...(encyclopedia ?? {
      provider: "Wikipedia",
      query,
      evidenceLevel: "unavailable",
      results: []
    }),
    scholarlyResearch,
    providerAvailability: {
      encyclopedia: encyclopedia ? "available" : "unavailable",
      scholarly: scholarlyResearch ? "available" : "unavailable"
    }
  };
}

export function listDreamTools() {
  return dreamToolRegistry.list();
}
