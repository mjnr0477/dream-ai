import http from "node:http";
import { isIP } from "node:net";
import { readFile } from "node:fs/promises";
import { extname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { generateDreamReply, getDreamRuntimeStatus } from "./lib/dream-engine.js";
import { getDreamCapabilities } from "./lib/capabilities.js";
import { runPublicResearch } from "./lib/dream-tools.js";
import { formatCodeArtifacts, parseCodeArtifacts } from "./lib/code-artifacts.js";

const ROOT = fileURLToPath(new URL(".", import.meta.url));
const PUBLIC_DIR = join(ROOT, "public");
const PORT = Number(process.env.PORT || 3000);
const MAX_BODY_BYTES = 32 * 1024;
const MAX_MESSAGE_CHARS = 8000;
const MAX_HISTORY_ITEMS = 12;
const MAX_ACTIVE_CHATS = 2;
const configuredRateLimit = Number(process.env.DREAM_API_RATE_LIMIT);
const RATE_LIMIT_CAPACITY = Number.isInteger(configuredRateLimit) && configuredRateLimit > 0
  ? Math.min(configuredRateLimit, 120)
  : 20;
const RATE_LIMIT_REFILL_PER_MS = RATE_LIMIT_CAPACITY / 60000;
const RATE_LIMIT_IDLE_MS = 10 * 60 * 1000;
const MAX_RATE_LIMIT_BUCKETS = 5000;
const rateLimitBuckets = new Map();
let activeChats = 0;
let activeResearch = 0;
const MODES = ["general", "coding", "learning"];
const SECURITY_HEADERS = {
  "x-content-type-options": "nosniff",
  "x-frame-options": "DENY",
  "referrer-policy": "strict-origin-when-cross-origin",
  "permissions-policy": "camera=(), microphone=(), geolocation=()"
};
const CONTENT_SECURITY_POLICY = [
  "default-src 'self'", "base-uri 'self'", "object-src 'none'",
  "frame-ancestors 'none'", "form-action 'self'", "script-src 'self'",
  "style-src 'self' https://fonts.googleapis.com", "style-src-attr 'unsafe-inline'",
  "font-src 'self' https://fonts.gstatic.com", "img-src 'self' data:"
].join("; ");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon"
};

function json(res, status, payload, extraHeaders = {}) {
  res.writeHead(status, { ...SECURITY_HEADERS, ...extraHeaders,
    "content-type": "application/json; charset=utf-8", "cache-control": "no-store"
  });
  res.end(JSON.stringify(payload));
}

function validateBrowserRequest(req) {
  const contentType = req.headers["content-type"] || "";
  if (!/^application\/json(?:\s*;|$)/i.test(contentType)) {
    const error = new Error("Content-Type must be application/json."); error.status = 415; throw error;
  }
  const fetchSite = req.headers["sec-fetch-site"];
  if (fetchSite && !["same-origin", "none"].includes(fetchSite.toLowerCase())) {
    const error = new Error("Cross-origin requests are not allowed."); error.status = 403; throw error;
  }
  const origin = req.headers.origin;
  if (origin) {
    let parsedOrigin;
    try { parsedOrigin = new URL(origin); }
    catch { const error = new Error("Invalid request origin."); error.status = 403; throw error; }
    const requestHost = String(req.headers.host || "").trim().toLowerCase();
    const isLoopbackHost = /^(localhost|127\.0\.0\.1|\[::1\])(?::\d+)?$/.test(requestHost);
    const expectedOrigin = process.env.DREAM_PUBLIC_ORIGIN
      ? new URL(process.env.DREAM_PUBLIC_ORIGIN).origin
      : (isLoopbackHost ? "http://" : "https://") + requestHost;
    if (
      parsedOrigin.origin !== expectedOrigin ||
      parsedOrigin.pathname !== "/" ||
      parsedOrigin.search ||
      parsedOrigin.hash ||
      parsedOrigin.username ||
      parsedOrigin.password
    ) {
      const error = new Error("Cross-origin requests are not allowed."); error.status = 403; throw error;
    }
  }
}


function getClientIp(req) {
  // Render documents X-Forwarded-For as the source of the real client IP.
  // Use only its first address; never trust later proxy-supplied entries.
  const forwarded = req.headers["x-forwarded-for"];
  const first = typeof forwarded === "string" ? forwarded.split(",")[0].trim() : "";
  if (first && isIP(first)) return first;
  const remote = req.socket?.remoteAddress;
  return remote && isIP(remote) ? remote : "unknown";
}

function consumeApiRateLimit(req) {
  const now = Date.now();
  const ip = getClientIp(req);
  if (ip === "127.0.0.1" || ip === "::1" || ip === "::ffff:127.0.0.1") return { allowed: true };
  let bucket = rateLimitBuckets.get(ip);

  if (!bucket) {
    if (rateLimitBuckets.size >= MAX_RATE_LIMIT_BUCKETS) {
      for (const [key, entry] of rateLimitBuckets) {
        if (now - entry.lastSeenAt > RATE_LIMIT_IDLE_MS) rateLimitBuckets.delete(key);
      }
      if (rateLimitBuckets.size >= MAX_RATE_LIMIT_BUCKETS) {
        return { allowed: false, retryAfter: 60 };
      }
    }
    bucket = { tokens: RATE_LIMIT_CAPACITY, updatedAt: now, lastSeenAt: now };
  }

  const elapsed = Math.max(0, now - bucket.updatedAt);
  bucket.tokens = Math.min(RATE_LIMIT_CAPACITY, bucket.tokens + elapsed * RATE_LIMIT_REFILL_PER_MS);
  bucket.updatedAt = now;
  bucket.lastSeenAt = now;

  if (bucket.tokens < 1) {
    rateLimitBuckets.set(ip, bucket);
    const retryAfter = Math.max(1, Math.ceil((1 - bucket.tokens) / RATE_LIMIT_REFILL_PER_MS / 1000));
    return { allowed: false, retryAfter };
  }

  bucket.tokens -= 1;
  rateLimitBuckets.set(ip, bucket);
  return { allowed: true };
}

function enforceApiRateLimit(req, res) {
  const result = consumeApiRateLimit(req);
  if (result.allowed) return true;
  json(res, 429, { error: "Too many requests. Please retry shortly." }, {
    "retry-after": String(result.retryAfter)
  });
  return false;
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

async function handleResearch(req, res) {
  validateBrowserRequest(req);
  if (!enforceApiRateLimit(req, res)) return;
  const body = await readJson(req);
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return json(res, 400, { error: "Request body must be a JSON object." });
  }
  if (typeof body.query !== "string" || !body.query.trim() || body.query.length > 200) {
    return json(res, 400, { error: "Research query must contain 1 to 200 characters." });
  }
  if (activeResearch >= 2) {
    res.writeHead(503, { ...SECURITY_HEADERS,
      "retry-after": "3", "content-type": "application/json; charset=utf-8", "cache-control": "no-store"
    });
    return res.end(JSON.stringify({ error: "DREAM research is busy. Please retry in a few seconds." }));
  }
  activeResearch += 1;
  try {
    return json(res, 200, await runPublicResearch(body.query));
  } catch (error) {
    console.error("DREAM public research failed:", error.message);
    return json(res, 502, { error: "Public research is temporarily unavailable. Please try again." });
  } finally {
    activeResearch -= 1;
  }
}

async function handleChat(req, res) {
  validateBrowserRequest(req);
  if (!enforceApiRateLimit(req, res)) return;
  const body = await readJson(req);
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return json(res, 400, { error: "Request body must be a JSON object." });
  }
  const mode = body.mode ?? "general";
  const memoryContext = body.memoryContext ?? "";
  if (typeof memoryContext !== "string" || memoryContext.length > 3000) {
    return json(res, 400, { error: "Memory context must be text no longer than 3,000 characters." });
  }
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
    res.writeHead(503, { ...SECURITY_HEADERS,
      "retry-after": "3", "content-type": "application/json; charset=utf-8", "cache-control": "no-store"
    });
    return res.end(JSON.stringify({
      error: "DREAM AI is handling other requests. Please retry in a few seconds."
    }));
  }

  activeChats += 1;
  try {
    const result = await generateDreamReply(messages, mode, process.env, memoryContext);
    let reply = result.reply;
    if (mode === "coding" && reply.includes("--- DREAM FILE:")) {
      try {
        reply = formatCodeArtifacts(parseCodeArtifacts(reply));
      } catch (error) {
        console.warn("DREAM coding artifact format was invalid:", error.message);
      }
    }
    return json(res, 200, {
      reply,
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
    res.writeHead(200, { ...SECURITY_HEADERS,
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
  if (req.method === "POST" && url.pathname === "/api/research") {
    try {
      return await handleResearch(req, res);
    } catch (error) {
      return json(res, error.status || 500, { error: error.status ? error.message : "An unexpected server error occurred." });
    }
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
