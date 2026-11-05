import http from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { generateDreamReply, getDreamRuntimeStatus } from "./lib/dream-engine.js";
import { getDreamCapabilities } from "./lib/capabilities.js";

const ROOT = fileURLToPath(new URL(".", import.meta.url));
const PUBLIC_DIR = join(ROOT, "public");
const PORT = Number(process.env.PORT || 3000);
const MAX_BODY_BYTES = 32 * 1024;
const MAX_MESSAGE_CHARS = 8000;
const MAX_HISTORY_ITEMS = 12;
const MAX_ACTIVE_CHATS = 2;
let activeChats = 0;
const MODES = ["general", "coding", "learning"];
const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon"
};

const SECURITY_HEADERS = {
  "x-content-type-options": "nosniff",
  "x-frame-options": "DENY",
  "referrer-policy": "strict-origin-when-cross-origin",
  "permissions-policy": "camera=(), microphone=(), geolocation=()"
};
const CONTENT_SECURITY_POLICY = [
  "default-src 'self'",
  "base-uri 'self'",
  "object-src 'none'",
  "frame-ancestors 'none'",
  "form-action 'self'",
  "script-src 'self'",
  "style-src 'self' https://fonts.googleapis.com",
  "style-src-attr 'unsafe-inline'",
  "font-src 'self' https://fonts.gstatic.com",
  "img-src 'self' data:"
].join("; ");

function json(res, status, payload) {
  res.writeHead(status, {
    ...SECURITY_HEADERS,
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store"
  });
  res.end(JSON.stringify(payload));
}

function validateBrowserRequest(req) {
  const contentType = req.headers["content-type"] || "";
  if (!/^application\/json(?:\s*;|$)/i.test(contentType)) {
    const error = new Error("Content-Type must be application/json.");
    error.status = 415;
    throw error;
  }

  // Browser metadata is an additional CSRF signal, not an authentication system.
  const fetchSite = req.headers["sec-fetch-site"];
  if (fetchSite && !["same-origin", "none"].includes(fetchSite.toLowerCase())) {
    const error = new Error("Cross-origin chat requests are not allowed.");
    error.status = 403;
    throw error;
  }

  // Origin is optional for non-browser clients, but when supplied it must match
  // the request Host exactly. Never trust forwarded-host headers for this check.
  const origin = req.headers.origin;
  if (origin) {
    let parsedOrigin;
    try {
      parsedOrigin = new URL(origin);
    } catch {
      const error = new Error("Invalid request origin.");
      error.status = 403;
      throw error;
    }
    const requestHost = String(req.headers.host || "").toLowerCase();
    if (
      !["http:", "https:"].includes(parsedOrigin.protocol) ||
      parsedOrigin.host.toLowerCase() !== requestHost ||
      parsedOrigin.username ||
      parsedOrigin.password
    ) {
      const error = new Error("Cross-origin chat requests are not allowed.");
      error.status = 403;
      throw error;
    }
  }
}

async function readJson(req) {
  let size = 0;
  const chunks = [];
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BODY_BYTES) {
      const error = new Error("Request body is too large.");
      error.status = 413;
      throw error;
    }
    chunks.push(chunk);
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    const error = new Error("Request body must be valid JSON.");
    error.status = 400;
    throw error;
  }
}

async function handleChat(req, res) {
  validateBrowserRequest(req);
  const body = await readJson(req);
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return json(res, 400, { error: "Request body must be a JSON object." });
  }
  const mode = body.mode ?? "general";
  if (!MODES.includes(mode)) {
    return json(res, 400, { error: "Mode must be general, coding or learning." });
  }
  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    return json(res, 400, { error: "Send a non-empty messages array." });
  }
  const messages = body.messages.slice(-MAX_HISTORY_ITEMS);
  for (const item of messages) {
    if (
      !item ||
      !["user", "assistant"].includes(item.role) ||
      typeof item.content !== "string" ||
      !item.content.trim() ||
      item.content.length > MAX_MESSAGE_CHARS
    ) {
      return json(res, 400, {
        error: "Each message needs a valid user/assistant role and text under 8,000 characters."
      });
    }
  }
  if (messages.at(-1).role !== "user") {
    return json(res, 400, { error: "The last message must be from the user." });
  }

  if (activeChats >= MAX_ACTIVE_CHATS) {
    return json(res, 503, {
      error: "DREAM AI is handling other requests. Please retry in a few seconds."
    });
  }

  activeChats += 1;
  try {
    const result = await generateDreamReply(messages, mode);
    return json(res, 200, {
      reply: result.reply,
      mode,
      source: result.source
    });
  } catch (error) {
    console.error("DREAM AI request failed:", error.message);
    return json(res, 502, { error: error.message || "DREAM AI runtime unavailable." });
  } finally {
    activeChats -= 1;
  }
}

async function serveStatic(encodedPathname, res) {
  let pathname;
  try {
    pathname = decodeURIComponent(encodedPathname);
  } catch {
    res.writeHead(400, { ...SECURITY_HEADERS, "content-type": "text/plain; charset=utf-8" });
    res.end("Invalid URL path");
    return;
  }
  const requested = pathname === "/" ? "/index.html" : pathname;
  const filePath = resolve(PUBLIC_DIR, "." + requested);
  const pathFromPublic = relative(PUBLIC_DIR, filePath);
  if (
    pathFromPublic === ".." ||
    pathFromPublic.startsWith(".." + sep) ||
    resolve(filePath) === resolve(PUBLIC_DIR)
  ) {
    res.writeHead(403, { ...SECURITY_HEADERS, "content-type": "text/plain; charset=utf-8" });
    res.end("Forbidden");
    return;
  }
  try {
    const content = await readFile(filePath);
    res.writeHead(200, {
      ...SECURITY_HEADERS,
      "content-type": MIME[extname(filePath)] || "application/octet-stream",
      "content-security-policy": CONTENT_SECURITY_POLICY,
      "cache-control": extname(filePath) === ".html" ? "no-cache" : "public, max-age=300"
    });
    res.end(content);
  } catch {
    res.writeHead(404, { ...SECURITY_HEADERS, "content-type": "text/plain; charset=utf-8" });
    res.end("Not found");
  }
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  if (req.method === "GET" && url.pathname === "/api/health/live") {
    return json(res, 200, { ok: true, name: "DREAM AI", status: "live" });
  }
  if (req.method === "GET" && url.pathname === "/api/health") {
    const runtime = await getDreamRuntimeStatus();
    // Public health is deliberately coarse; detailed checkpoint and hardware
    // diagnostics belong in authenticated operator tooling, not a public route.
    return json(res, 200, {
      ok: true,
      name: "DREAM AI",
      version: "0.3.0",
      modelConfigured: runtime.configured,
      modelReady: runtime.ready
    });
  }
  if (req.method === "GET" && url.pathname === "/api/capabilities") {
    const runtime = await getDreamRuntimeStatus();
    const capabilities = getDreamCapabilities(runtime);
    if (capabilities.model) {
      delete capabilities.model.completedTrainingSteps;
      delete capabilities.model.checkpointSha256;
    }
    return json(res, 200, capabilities);
  }
  if (req.method === "POST" && url.pathname === "/api/chat") {
    try {
      return await handleChat(req, res);
    } catch (error) {
      return json(res, error.status || 500, {
        error: error.status ? error.message : "An unexpected server error occurred."
      });
    }
  }
  if (req.method === "GET") return serveStatic(url.pathname, res);
  res.writeHead(405, { ...SECURITY_HEADERS, allow: "GET, POST" }).end("Method not allowed");
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`DREAM AI listening on port ${PORT}`);
});

function shutdown() {
  server.close(() => process.exit(0));
}
process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);
