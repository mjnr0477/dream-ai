const TOOL_DEFINITIONS = [
  {
    id: "web_research",
    label: "Web research",
    status: "not-configured",
    description: "Live web search and source retrieval are not connected to the DREAM runtime.",
    requires: ["A trusted search connector", "Server-side credentials or an approved public search endpoint"]
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
    contractVersion: 1,
    model: {
      id: runtime.modelName || "dream-native",
      type: "independent-native-transformer",
      status: ready ? "ready" : configured ? "unavailable" : "not-configured",
      completedTrainingSteps: ready ? runtime.completedTrainingSteps : null,
      checkpointSha256: ready ? runtime.checkpointSha256 : null
    },
    application: {
      chat: "available",
      modes: ["general", "coding", "learning"],
      conversationHistory: "browser-local",
      export: "available"
    },
    tools: TOOL_DEFINITIONS.map((tool) => ({
      ...tool,
      requires: [...tool.requires]
    })),
    safety: {
      toolExecutionDefault: "disabled",
      externalActionsRequireApproval: true,
      unrestrictedAccess: false
    }
  };
}
