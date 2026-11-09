import { ToolRegistry } from "./tool-registry.js";
import { searchPublicKnowledge } from "./research.js";
import { searchMultiSourceResearch } from "./research-multisource.js";
import { searchGlobalPublicSources } from "./research-global.js";
import { searchPlatformSources } from "./research-platforms.js";

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
  env = process.env,
  audit = auditToolExecution,
  maxConcurrent = 4
} = {}) {
  const registry = new ToolRegistry({ audit, maxConcurrent });
  registry.register({
    id: "public_knowledge_search",
    description: "Search Wikipedia's public API and return source-linked encyclopedia excerpts.",
    risk: "read",
    permission: "research:public",
    timeoutMs: 8000,
    inputSchema: {
      type: "object",
      properties: { query: { type: "string", minLength: 1, maxLength: 200 } },
      required: ["query"],
      additionalProperties: false
    }
  }, async ({ input }) => searchPublicKnowledge(input.query, fetchImpl));
  registry.register({
    id: "global_public_source_search",
    description: "Search Wikidata public entity records, GDELT news from the last seven days, and optionally a configured Brave web index.",
    risk: "read",
    permission: "research:public",
    timeoutMs: 22000,
    inputSchema: {
      type: "object",
      properties: { query: { type: "string", minLength: 1, maxLength: 200 } },
      required: ["query"],
      additionalProperties: false
    }
  }, async ({ input }) => searchGlobalPublicSources(input.query, fetchImpl, env));
  registry.register({
    id: "platform_public_source_search",
    description: "Search Google News RSS, public Stack Overflow Q&A, public YouTube video metadata, public Bluesky posts and optional Google-grounded web and X public-post results. Each provider reports its own availability and keyless services use anonymous read-only access.",
    risk: "read",
    permission: "research:public",
    timeoutMs: 26000,
    inputSchema: {
      type: "object",
      properties: { query: { type: "string", minLength: 1, maxLength: 200 } },
      required: ["query"],
      additionalProperties: false
    }
  }, async ({ input }) => searchPlatformSources(input.query, fetchImpl, env));
  registry.register({
    id: "multi_source_scholarly_research",
    description: "Search Crossref, OpenAlex, arXiv and Semantic Scholar for public scholarly metadata and available abstracts, with provider provenance and publication freshness labels.",
    risk: "read",
    permission: "research:public",
    timeoutMs: 22000,
    inputSchema: {
      type: "object",
      properties: { query: { type: "string", minLength: 1, maxLength: 200 } },
      required: ["query"],
      additionalProperties: false
    }
  }, async ({ input }) => searchMultiSourceResearch(input.query, fetchImpl));
  return registry;
}

export const dreamToolRegistry = createDreamToolRegistry();

async function executePublicTool(toolId, query, registry) {
  const execution = await registry.execute({
    toolId,
    input: { query },
    principal: PUBLIC_RESEARCH_PRINCIPAL
  });
  return execution.result;
}

export async function runPublicKnowledgeSearch(query, registry = dreamToolRegistry) {
  return executePublicTool("public_knowledge_search", query, registry);
}

export async function runPublicResearch(query, registry = dreamToolRegistry) {
  const tasks = await Promise.allSettled([
    executePublicTool("public_knowledge_search", query, registry),
    executePublicTool("global_public_source_search", query, registry),
    executePublicTool("multi_source_scholarly_research", query, registry),
    executePublicTool("platform_public_source_search", query, registry)
  ]);
  const encyclopedia = tasks[0].status === "fulfilled" ? tasks[0].value : null;
  const globalResearch = tasks[1].status === "fulfilled" ? tasks[1].value : null;
  const scholarlyResearch = tasks[2].status === "fulfilled" ? tasks[2].value : null;
  const platformResearch = tasks[3].status === "fulfilled" ? tasks[3].value : null;
  if (!encyclopedia && !globalResearch && !scholarlyResearch && !platformResearch) {
    throw new Error("All public research providers are unavailable.");
  }
  return {
    ...(encyclopedia ?? {
      provider: "Wikipedia",
      query,
      evidenceLevel: "unavailable",
      results: []
    }),
    query,
    globalResearch,
    scholarlyResearch,
    platformResearch,
    providerAvailability: {
      encyclopedia: encyclopedia ? "available" : "unavailable",
      globalPublicSources: globalResearch ? "available" : "unavailable",
      scholarly: scholarlyResearch ? "available" : "unavailable",
      platforms: platformResearch ? "available" : "unavailable"
    }
  };
}

export function listDreamTools() {
  return dreamToolRegistry.list();
}
