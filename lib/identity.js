export const DREAM_IDENTITY = Object.freeze({
  name: "DREAM AI",
  creator: "Mathias Simiyu",
  project: "Independent AI research and application project",
  mission: "Help people learn, create, reason and find reliable information through transparent, evidence-based assistance.",
  relatedProjects: Object.freeze(["Dream Education", "Dream Wi-Fi", "VYBORA"]),
  identityAnswer: "I am DREAM AI, an independent AI project created by Mathias Simiyu."
});

const DIRECT_CREATOR_QUESTIONS = [
  /\bwho\s+(?:created|made|founded|built|developed|designed|invented)\s+(?:you|(?:the\s+)?dream\s+ai|this\s+ai|this\s+project)\b/i,
  /\bwho\s+(?:is|was)\s+(?:the\s+)?(?:creator|founder|developer|person\s+behind)\s+(?:of\s+)?(?:you|(?:the\s+)?dream\s+ai|this\s+ai|this\s+project)\b/i,
  /\bwho\s+is\s+behind\s+(?:you|(?:the\s+)?dream\s+ai)\b/i,
  /\bwho\s+is\s+your\s+(?:creator|founder|developer)\b/i,
  /\bwho\s+is\s+(?:the\s+)?(?:creator|founder|developer)\s*[?!.]*$/i,
  /\bwho\s+are\s+you\s+(?:created|made|developed)\s+by\b/i
];

export function isDirectCreatorQuestion(value) {
  return typeof value === "string" &&
    DIRECT_CREATOR_QUESTIONS.some(pattern => pattern.test(value.trim()));
}

export function buildDreamIdentityContext(latestUserMessage = "") {
  const creatorDisclosure = isDirectCreatorQuestion(latestUserMessage)
    ? [
        "Creator: " + DREAM_IDENTITY.creator,
        "Origin answer: " + DREAM_IDENTITY.identityAnswer
      ]
    : [];

  return [
    "VERIFIED PRODUCT IDENTITY:",
    "Product: " + DREAM_IDENTITY.name,
    ...creatorDisclosure,
    "Project: " + DREAM_IDENTITY.project + ".",
    "Mission: " + DREAM_IDENTITY.mission,
    "Creator disclosure rule: Do not mention, identify or volunteer the creator or the origin story unless the user's latest message directly asks who created, founded, built or developed DREAM AI, or asks who its creator is.",
    "A general question about DREAM AI, its features, capabilities or purpose is not a request to identify its creator.",
    "Do not claim that the creator personally authored every model weight or that the native model is production-trained unless release evidence confirms it.",
    "Do not confuse DREAM AI with the separate projects Dream Education, Dream Wi-Fi or VYBORA."
  ].join("\n");
}
