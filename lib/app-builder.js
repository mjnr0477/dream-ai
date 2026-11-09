const BUILD_VERB = /\b(?:build|create|develop|make|generate|scaffold|prototype|implement|design|write)\b/i;
const APP_OR_PROJECT = /\b(?:apps?|applications?|websites?|web\s+sites?|software\s+projects?|software\s+applications?|REST\s+APIs?|GraphQL\s+APIs?|APIs?|backends?|microservices?|command[-\s]line\s+tools?|CLIs?|games?|browser\s+extensions?|extensions?|bots?|firmware|embedded\s+software|IoT)\b/i;
const EXPLANATION_OR_CREATIVE = /^\s*(?:what\s+is|explain|teach(?:\s+me)?|tell\s+me\s+about|compare|why|how\s+does|how\s+do\s+i|how\s+can\s+i)\b/i;
const CREATIVE_CONTENT = /\b(?:poem|short\s+story|fiction|lyrics?|song|joke|essay|role[-\s]?play)\b/i;

const TARGETS = Object.freeze([
  {
    id: "browser-extension",
    match: /\b(?:browser\s+extension|chrome\s+extension|firefox\s+add[-\s]?on|manifest\s+v3)\b/i,
    guidance: "For browser extensions, use the requested browser's supported extension format; default to Chrome Manifest V3 when unspecified. Include manifest permissions at the minimum required scope, background/service-worker logic, content scripts only when needed, options UI where useful, and clear install/test instructions."
  },
  {
    id: "embedded-iot",
    match: /\b(?:firmware|embedded|IoT|ESP32|ESP8266|Arduino|Raspberry\s+Pi|microcontroller)\b/i,
    guidance: "For embedded/IoT projects, identify the exact board and peripherals when supplied. Prefer PlatformIO or Arduino CLI for compatible boards; document pin maps, voltage assumptions, required libraries, flashing steps and safe failure behavior. Do not claim hardware tests were run without the hardware."
  },
  {
    id: "bot",
    match: /\b(?:bot|Discord|Telegram|Slack\s+bot|WhatsApp\s+bot)\b/i,
    guidance: "For bots, use the platform's official SDK or documented API, separate command/event handlers, store tokens only in environment variables, and include permission/intents setup and local test instructions. Never generate real credentials."
  },
  {
    id: "game",
    match: /\b(?:game|gaming|Godot|Unity|Unreal\s+Engine)\b/i,
    guidance: "For games, select the requested engine; when no engine is specified, prefer Godot for an approachable open-source 2D/3D project. Include scenes, scripts, input handling, assets or clearly licensed placeholders, project settings and play/test instructions."
  },
  {
    id: "command-line",
    match: /\b(?:command[-\s]line|CLI|terminal\s+tool|shell\s+tool)\b/i,
    guidance: "For command-line tools, include argument parsing, help output, input validation, meaningful exit codes, tests, and install/run examples. Prefer the user's language; otherwise choose Python, Node.js, Go or Rust based on the task."
  },
  {
    id: "backend-api",
    match: /\b(?:REST\s+API|GraphQL\s+API|API\s+server|backend|microservice|server[-\s]side)\b/i,
    guidance: "For backend/API projects, include route/controller boundaries, input validation, authentication and authorization where needed, parameterized database access, migrations, structured errors, health checks and an OpenAPI contract when appropriate. Prefer FastAPI or Express when no stack is specified."
  },
  {
    id: "desktop",
    match: /\b(?:desktop|Windows\s+app|macOS\s+app|Linux\s+app|Electron|Tauri)\b/i,
    guidance: "For desktop applications, use the requested native toolkit; otherwise consider Tauri or Electron and state the trade-offs. Include the desktop entry point, packaging/build configuration, platform-specific notes and permission handling."
  },
  {
    id: "mobile",
    match: /\b(?:mobile|Android|iOS|iPhone|iPad|Flutter|React\s+Native|Expo|Kotlin\s+app|Swift\s+app)\b/i,
    guidance: "For mobile applications, use the requested framework; otherwise choose Expo/React Native or Flutter and state the choice. Include navigation, app configuration, permissions, responsive layouts, accessible controls and device testing commands. Native iOS builds require macOS and Xcode; Android builds require the Android toolchain."
  },
  {
    id: "web",
    match: /\b(?:website|web\s+site|web\s+app|webapp|frontend|Next\.?js|React|Vue|Svelte|Angular|PWA)\b/i,
    guidance: "For web projects, choose the requested framework; otherwise use Next.js with TypeScript for full-stack products, or HTML/CSS/JavaScript for a genuinely simple static site. Include responsive accessible UI, loading/empty/error states, routing and secure server-side handling when relevant."
  }
]);

export function isAppBuildRequest(query) {
  if (typeof query !== "string" || query.length > 8000) return false;
  const normalized = query.trim();
  if (!normalized || EXPLANATION_OR_CREATIVE.test(normalized) || CREATIVE_CONTENT.test(normalized)) return false;
  return BUILD_VERB.test(normalized) && APP_OR_PROJECT.test(normalized);
}

export function classifyAppTarget(query) {
  if (typeof query !== "string") return "general";
  const target = TARGETS.find(item => item.match.test(query));
  return target?.id || "general";
}

export function buildAppBuilderGuidance(query) {
  if (!isAppBuildRequest(query)) return "";
  const target = classifyAppTarget(query);
  const targetGuidance = TARGETS.find(item => item.id === target)?.guidance ||
    "For a general software project, identify the runtime and deployment target from the request. Ask one focused question only if the platform choice materially changes the project; otherwise choose a mainstream, maintained stack and state the assumption.";

  return [
    "APPLICATION BUILD MODE: The user is asking DREAM AI to create a software project, not merely explain a code snippet.",
    "Deliver a complete, coherent project with compatible files, a clear folder structure, working entry points and implementation across the main user flow. Do not provide disconnected fragments, fake placeholder functions or TODO-only scaffolding in place of core behavior.",
    "Target family: " + target + ". " + targetGuidance,
    "Preserve any framework, language, version, repository structure and constraints explicitly supplied by the user. If they are missing, choose a maintained, mainstream stack that fits the project and state the choice briefly rather than blocking on a long questionnaire.",
    "Include a README with prerequisites, exact install/run/build/test commands and platform limitations; include the package/build manifest, configuration files, .gitignore, tests and an .env.example with placeholder names but no real secrets when those files are relevant.",
    "Design the project around its actual user flows. Include validation, useful loading/empty/error states, accessibility and responsive behavior for visual interfaces. Include data models, persistence and API contracts when the app needs them.",
    "Use secure defaults: enforce authorization on the server, validate untrusted input, parameterize database access, protect secrets, restrict permissions, avoid unsafe dynamic code execution, and document any external service or provider approval requirement.",
    "Deliver every generated file using one block per file: --- DREAM FILE: relative/path --- on its own line, exact UTF-8 file contents, then --- END DREAM FILE --- on its own line. Use normalized relative POSIX paths, include all required files, do not repeat paths, and keep the bundle within the supported artifact limits. A downloadable project ZIP is assembled from these validated files.",
    "Do not invent successful builds, tests, emulator/simulator runs, security scans or deployments. Do not claim that a build, test, emulator, simulator or deployment was run unless an actual execution tool returned that result. DREAM's current application does not provide an isolated execution sandbox; give exact commands and label verification as not run.",
    "Respect platform realities: signing certificates, Xcode/macOS, Android SDKs, device hardware, paid developer accounts, API approvals and cloud credentials may be needed for final release. Explain only the requirements relevant to this project.",
    "If the user requests a machine-readable project bundle, put no prose outside the file blocks. Otherwise give a short overview, list the stack and run steps, and identify any verification that remains."
  ].join(" ");
}
