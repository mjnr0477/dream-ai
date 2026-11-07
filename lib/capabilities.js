const TOOL_DEFINITIONS = [
  {
    id: "public_knowledge_search",
    label: "Public knowledge search",
    status: "available",
    description: "Wikipedia public API for encyclopedia search and short article introductions.",
    requires: []
  },
  {
    id: "global_public_source_search",
    label: "Global public-source discovery",
    status: "available",
    description: "Wikidata public entity records and GDELT news headlines from the last seven days; a configured Brave Search key additionally enables general web-index results.",
    requires: []
  },
  {
    id: "multi_source_scholarly_research",
    label: "Scholarly research discovery",
    status: "available",
    description: "Crossref and OpenAlex publication metadata; metadata does not prove paper findings or include full text.",
    requires: []
  },
  {
    id: "web_research",
    label: "General web-index search",
    status: "not-configured",
    description: "Broad web-index results require DREAM_BRAVE_SEARCH_API_KEY. Current public-source research still works through Wikipedia, Wikidata, GDELT, Crossref and OpenAlex.",
    requires: ["A Brave Search API key stored as a server-side Render environment variable"]
  },
  {
    id: "file_analysis",
    label: "File analysis",
    status: "not-implemented",
    description: "The application does not yet accept and analyze uploaded documents.",
    requires: ["Secure upload storage", "File-type validation", "Isolated parsers and retention controls"]
  },
  {
    id: "code_execution",
    label: "Code execution",
    status: "not-implemented",
    description: "DREAM can discuss code, but it cannot execute user code in a sandbox yet.",
    requires: ["Ephemeral isolated execution", "CPU, memory and time limits", "Network-deny-by-default policy"]
  },
  {
    id: "computer_control",
    label: "Computer control",
    status: "not-configured",
    description: "No computer-use agent or controlled desktop session is connected.",
    requires: ["A dedicated remote session", "User-visible action approvals", "Audit and emergency-stop controls"]
  },
  {
    id: "external_actions",
    label: "External actions",
    status: "not-configured",
    description: "No external accounts or action connectors are connected to DREAM.",
    requires: ["Explicitly connected services", "Per-action authorization", "Scoped credentials and audit logs"]
  }
];

export function getDreamCapabilities(runtime = {}) {
  const ready = runtime.ready === true;
  const configured = runtime.configured === true;

  return {
    product: "DREAM AI",
    contractVersion: 2,
    model: {
      id: runtime.modelName || "dream-native",
      type: !configured
        ? "not-configured"
        : runtime.provider === "native"
          ? "independent-native-transformer"
          : runtime.provider === "openai-compatible"
            ? "compatible-upstream-model"
            : "unverified",
      status: ready ? "ready" : configured ? "unavailable" : "not-configured",
      completedTrainingSteps: ready ? runtime.completedTrainingSteps : null,
      checkpointSha256: ready ? runtime.checkpointSha256 : null
    },
    application: {
      chat: "available",
      modes: ["general", "coding", "learning", "research"],
      codingArtifacts: "validated-multi-file-text",
      researchScope: "multi-provider-public-source-discovery",
      conversationHistory: "browser-local",
      export: "available"
    },
    tools: TOOL_DEFINITIONS.map((tool) => ({
      ...tool,
      requires: [...tool.requires]
    })),
    safety: {
      toolExecutionDefault: "read-only-public-sources",
      externalActionsRequireApproval: true,
      unrestrictedAccess: false,
      personalDataPolicy: "public-interest and consent-based research only; no private dossiers or sensitive personal data exposure"
    }
  };
}
