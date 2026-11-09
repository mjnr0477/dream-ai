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
    description: "Crossref and OpenAlex publication metadata plus Europe PMC biomedical publication records and abstracts where supplied.",
    requires: []
  },
  {
    id: "web_research",
    label: "General web-index search",
    status: "not-configured",
    description: "Extracted page context from Brave's LLM Context API requires DREAM_BRAVE_SEARCH_API_KEY. Current no-key public-source research still works through Wikipedia, Wikidata, GDELT, Crossref and OpenAlex.",
    requires: ["A Brave Search API key stored as a server-side Render environment variable"]
  },
  {
    id: "platform_public_source_search",
    label: "Search engines, news and social sources",
    status: "available",
    description: "Keyless World Bank indicator data, public Bluesky search and Google News RSS are available; Google Search grounding, YouTube Data API and X API activate only with their own server-side credentials.",
    requires: []
  },
  {
    id: "google_search_grounding",
    label: "Google Search grounding",
    status: "not-configured",
    description: "Google-grounded public web research through the Gemini API; this is a retrieval service and does not train or replace the DREAM model.",
    requires: ["DREAM_GOOGLE_GEMINI_API_KEY stored as a private server-side environment variable"]
  },
  {
    id: "youtube_public_search",
    label: "YouTube public video search",
    status: "not-configured",
    description: "Public video titles, descriptions, channel names and publication dates through YouTube Data API v3; no private account access or transcript guarantee.",
    requires: ["DREAM_YOUTUBE_API_KEY stored server-side"]
  },
  {
    id: "x_public_search",
    label: "X public conversation search",
    status: "not-configured",
    description: "Recent public posts through the official X API; access, pricing and quotas depend on the developer account.",
    requires: ["DREAM_X_BEARER_TOKEN stored server-side and an eligible X API plan"]
  },
  {
    id: "amazon_product_catalog",
    label: "Amazon product catalogue",
    status: "not-implemented",
    description: "Amazon's Creators API is a product-catalogue and shopping source, not a general web-knowledge search engine.",
    requires: ["Amazon Associates enrolment for a supported marketplace", "Creators API approval and credentials", "At least 10 qualifying sales in the previous 30 days under current access requirements"]
  },
  {
    id: "restricted_social_research",
    label: "Restricted social-network research",
    status: "not-implemented",
    description: "Meta, TikTok, LinkedIn and Reddit require platform-specific permissions, review or commercial agreements. DREAM does not scrape restricted accounts or bypass platform controls.",
    requires: ["Approved platform applications and scopes", "Terms-compliant access and data retention", "Separate permission for any AI-training use of platform content"]
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

export function getDreamCapabilities(runtime = {}, env = process.env) {
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
      researchScope: [
        "multi-provider-public-source-discovery",
        "world-bank-indicators",
        ...(env.DREAM_BRAVE_SEARCH_API_KEY ? ["brave-web-context"] : []),
        ...(env.DREAM_GOOGLE_GEMINI_API_KEY ? ["google-search-grounding"] : []),
        ...(env.DREAM_YOUTUBE_API_KEY ? ["youtube-public-search"] : []),
        ...(env.DREAM_X_BEARER_TOKEN ? ["x-public-search"] : [])
      ],
      conversationHistory: "browser-local",
      export: "available"
    },
    tools: TOOL_DEFINITIONS.map((tool) => {
      if (tool.id === "google_search_grounding") {
        const configured = Boolean(env.DREAM_GOOGLE_GEMINI_API_KEY);
        return { ...tool, status: configured ? "available" : "not-configured", requires: configured ? [] : [...tool.requires] };
      }
      if (tool.id === "youtube_public_search") {
        const configured = Boolean(env.DREAM_YOUTUBE_API_KEY);
        return { ...tool, status: configured ? "available" : "not-configured", requires: configured ? [] : [...tool.requires] };
      }
      if (tool.id === "x_public_search") {
        const configured = Boolean(env.DREAM_X_BEARER_TOKEN);
        return { ...tool, status: configured ? "available" : "not-configured", requires: configured ? [] : [...tool.requires] };
      }
      if (tool.id !== "web_research") {
        return { ...tool, requires: [...tool.requires] };
      }
      const configured = Boolean(env.DREAM_BRAVE_SEARCH_API_KEY);
      return {
        ...tool,
        status: configured ? "available" : "not-configured",
        description: configured
          ? "AI-focused extracted web-page context is connected through Brave LLM Context. The provider supplies indexed page passages and source metadata; results still require evidence checks."
          : tool.description,
        requires: configured ? [] : [...tool.requires]
      };
    }),
    safety: {
      toolExecutionDefault: "read-only-public-sources",
      externalActionsRequireApproval: true,
      unrestrictedAccess: false,
      personalDataPolicy: "public-interest and consent-based research only; no private dossiers or sensitive personal data exposure"
    }
  };
}
