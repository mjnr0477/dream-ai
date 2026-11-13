import http from "node:http";
import { isIP } from "node:net";
import { readFile } from "node:fs/promises";
import { extname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { generateDreamReply, getDreamRuntimeStatus, isDreamEngineConfigured } from "./lib/dream-engine.js";
import { parseCodeArtifactsIfPresent } from "./lib/code-artifacts.js";
import { isAppBuildRequest } from "./lib/app-builder.js";
import { getDreamCapabilities } from "./lib/capabilities.js";
import { getDreamPluginCatalog } from "./lib/plugin-catalog.js";
import { createGoogleAccountService, isGmailRequest } from "./lib/google-account.js";
import { createMetaAccountService, isMetaRequest } from "./lib/meta-account.js";
import { runPublicResearch } from "./lib/dream-tools.js";
import { shouldAutoResearch, buildResearchContext, buildResearchAttemptContext } from "./lib/auto-research.js";
import { ClientConcurrencyLimiter } from "./lib/client-limiter.js";
import { resolveModelConfig } from "./lib/model-provider.js";
import { buildOfflineResearchReply, isOfflineGreetingQuery } from "./lib/offline-research-reply.js";

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
const clientConcurrencyLimiter = new ClientConcurrencyLimiter(1);
const googleAccountService = createGoogleAccountService();
const metaAccountService = createMetaAccountService();
let activeChats = 0;
let activeResearch = 0;
const MODES = ["general", "coding", "learning", "research"];
const OFFLINE_FALLBACK_BLOCKED_ERRORS = new Set([
  "DREAM_MODEL_URL must be a valid URL.",
  "DREAM_MODEL_URL must not contain embedded credentials.",
  "DREAM_MODEL_URL must use HTTPS unless it targets loopback for local development.",
  "Model endpoints must use HTTPS unless they target loopback for local development.",
  "Configured runtime is not the native DREAM transformer. External AI providers are not accepted.",
  "Native DREAM runtime has no verifiable trained-checkpoint identity.",
  "Native DREAM runtime model name does not match the configured model."
]);
const SAFE_CHAT_ERRORS = new Set([
  "DREAM_MODEL_URL must be a valid URL.",
  "DREAM_MODEL_URL must not contain embedded credentials.",
  "DREAM_MODEL_URL must use HTTPS unless it targets loopback for local development.",
  "Model endpoints must use HTTPS unless they target loopback for local development.",
  "Native runtime health check failed.",
  "Configured runtime is not the native DREAM transformer. External AI providers are not accepted.",
  "Native DREAM runtime has no verifiable trained-checkpoint identity.",
  "Native DREAM runtime model name does not match the configured model.",
  "Native DREAM runtime identity check timed out.",
  "DREAM AI's model runtime could not complete the request.",
  "DREAM AI's model runtime returned an empty response.",
  "DREAM AI's model runtime took too long to respond. Please try again."
]);
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
  // Render's Cloudflare edge supplies CF-Connecting-IP with the visitor IP.
  // Do not use X-Forwarded-For for security decisions: clients can send a
  // forged prefix and some proxy chains preserve it when appending addresses.
  const cloudflareClientIp = req.headers["cf-connecting-ip"];
  const candidate = typeof cloudflareClientIp === "string" ? cloudflareClientIp.trim() : "";
  if (candidate && isIP(candidate)) return candidate;
  // Fail closed to the transport peer when the trusted edge header is absent
  // or malformed. This may group clients behind a proxy, but cannot let them
  // choose arbitrary rate-limit/concurrency identities via request headers.
  const remote = req.socket?.remoteAddress;
  return remote && isIP(remote) ? remote : "unknown";
}

async function runWithClientConcurrency(req, res, handler) {
  const clientIp = getClientIp(req);
  // Keep loopback development/test clients outside the production per-client
  // gate; explicit forwarded client addresses remain subject to enforcement.
  if (clientIp === "127.0.0.1" || clientIp === "::1" || clientIp === "::ffff:127.0.0.1") {
    return handler(req, res);
  }
  const slot = clientConcurrencyLimiter.acquire(clientIp);
  if (!slot.allowed) {
    json(res, 429, { error: "This client already has a request in progress. Please retry shortly." }, {
      "retry-after": "3"
    });
    return;
  }
  try {
    return await handler(req, res);
  } finally {
    slot.release();
  }
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
  let tooLarge = false;
  const chunks = [];
  // Drain the request after crossing the limit without retaining additional
  // bytes. Exiting a Node async-iterator early destroys the request stream,
  // which can prevent the server from returning the intended 413 response.
  for await (const chunk of req) {
    if (tooLarge) continue;
    size += chunk.length;
    if (size > MAX_BODY_BYTES) {
      tooLarge = true;
      chunks.length = 0;
      continue;
    }
    chunks.push(chunk);
  }
  if (tooLarge) {
    const error = new Error("Request body is too large.");
    error.status = 413;
    throw error;
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
  const latestQuestion = messages.at(-1).content;
  const requestsGmail = isGmailRequest(latestQuestion);
  const requestsMeta = isMetaRequest(latestQuestion);
  let automaticResearch = null;
  let researchSources = [];
  try {
    let researchContext = "";
    let privateAccountContext = "";
    if (requestsGmail && !isDreamEngineConfigured(process.env)) {
      privateAccountContext = "DREAM's model runtime is not configured. No Gmail messages were retrieved. Explain that private account questions can be answered only after the DREAM model runtime is ready.";
    } else if (requestsGmail) {
      try {
        privateAccountContext = await googleAccountService.gmailContext(req, latestQuestion);
      } catch (error) {
        console.warn("DREAM Gmail retrieval failed:", error.providerStatus || error.name || "error");
        privateAccountContext = "Gmail retrieval failed. No email contents were successfully retrieved. Tell the user that account access failed and do not infer or invent email contents.";
      }
    }
    if (requestsMeta) {
      try {
        const metaContext = await metaAccountService.socialContext(req, latestQuestion);
        privateAccountContext = [privateAccountContext, metaContext].filter(Boolean).join("\n\n");
      } catch (error) {
        console.warn("DREAM Meta account retrieval failed:", error.providerStatus || error.name || "error");
        privateAccountContext = [privateAccountContext, "Meta account retrieval failed. No private social account data was successfully retrieved. Tell the user to reconnect the relevant Meta account; do not infer or invent account contents."].filter(Boolean).join("\n\n");
      }
    }
    if (!requestsGmail && !requestsMeta && !isOfflineGreetingQuery(latestQuestion) && shouldAutoResearch(latestQuestion, mode)) {
      const researchQuery = latestQuestion.trim().replace(/\s+/g, " ").slice(0, 200);
      try {
        // Research current public sources as well as stable reference material.
        // Keep provider diagnostics even when no source survives evidence validation,
        // so the model can explain the actual retrieval result instead of guessing.
        const research = await runPublicResearch(researchQuery);
        const prepared = buildResearchContext(research);
        researchSources = prepared.sources;
        researchContext = [
          prepared.context,
          buildResearchAttemptContext(research, { sourceCount: prepared.sources.length })
        ].filter(Boolean).join("\n\n");
        automaticResearch = {
          attempted: true,
          sourceCount: prepared.sources.length,
          status: prepared.sources.length ? "sources-found" : "no-usable-sources",
          sources: prepared.sources,
          providers: {
            ...(research.providerAvailability || {}),
            globalPublicSources: research.globalResearch?.providers || [],
            platformSources: research.platformResearch?.providers || [],
            scholarlySources: research.scholarlyResearch?.providers || []
          }
        };
      } catch (error) {
        // A failed search must not become a false claim that DREAM has no
        // real-time capability. Preserve a safe, explicit failure state for chat.
        console.info("DREAM automatic research unavailable:", error.message);
        researchContext = buildResearchAttemptContext(null, { failed: true });
        automaticResearch = {
          attempted: true,
          sourceCount: 0,
          status: "unavailable",
          sources: [],
          providers: {}
        };
      }
    }
    const result = await generateDreamReply(
      messages, mode, process.env, memoryContext, researchContext, privateAccountContext
    );
    const offlineSource = result.source === "dream-engine-unconfigured";
    const reply = offlineSource
      ? (requestsGmail || requestsMeta
          ? "DREAM AI's language-model worker is offline. No private account content has been analysed or returned. Reconnect this request when the model runtime is available."
          : buildOfflineResearchReply({ query: latestQuestion, sources: researchSources, mode }))
      : result.reply;
    const responseSource = offlineSource
      ? (researchSources.length ? "public-research-fallback" : "offline-status")
      : result.source;
    let artifacts = null;
    if (!offlineSource && (mode === "coding" || isAppBuildRequest(latestQuestion))) {
      try {
        artifacts = parseCodeArtifactsIfPresent(reply);
      } catch (error) {
        console.warn("DREAM coding artifact output was invalid:", error.message);
      }
    }
    return json(res, 200, {
      reply,
      mode,
      source: responseSource,
      ...(automaticResearch ? { autoResearch: automaticResearch } : {}),
      ...(artifacts ? { artifacts } : {})
    });
  } catch (error) {
    console.error("DREAM AI request failed:", error.message);
    if (OFFLINE_FALLBACK_BLOCKED_ERRORS.has(error.message)) {
      const publicError = SAFE_CHAT_ERRORS.has(error.message) ? error.message : "DREAM AI runtime configuration is invalid.";
      return json(res, 502, { error: publicError });
    }
    const offlineReply = requestsGmail || requestsMeta
      ? "DREAM AI's language-model worker is offline. No private account content has been analysed or returned. Reconnect this request when the model runtime is available."
      : buildOfflineResearchReply({ query: latestQuestion, sources: researchSources, mode });
    return json(res, 200, {
      reply: offlineReply,
      mode,
      source: (requestsGmail || requestsMeta || !researchSources.length) ? "offline-status" : "public-research-fallback",
      ...(automaticResearch ? { autoResearch: automaticResearch } : {})
    });
  } finally {
    activeChats -= 1;
  }
}

async function handleLearningExample(req, res) {
  validateBrowserRequest(req);
  if (!enforceApiRateLimit(req, res)) return;
  const body = await readJson(req);
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return json(res, 400, { error: "Request body must be a JSON object." });
  }
  const mode = body.mode;
  const prompt = typeof body.prompt === "string" ? body.prompt.trim() : "";
  const responseText = typeof body.response === "string" ? body.response.trim() : "";
  const consentVersion = body.consentVersion === undefined
    ? "dream-learning-opt-in-v1"
    : body.consentVersion;
  const sourceGroupId = body.sourceGroupId;
  if (isGmailRequest(prompt) || isMetaRequest(prompt)) {
    return json(res, 403, { error: "Private account questions cannot be submitted to DREAM learning or training collection." });
  }
  const validSourceGroupId = typeof sourceGroupId === "string" &&
    /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(sourceGroupId);
  if (!["general", "coding", "learning"].includes(mode) ||
      !prompt || prompt.length > 8000 ||
      !responseText || responseText.length > 16000 ||
      !["dream-learning-opt-in-v1", "dream-learning-opt-in-v2", "dream-learning-opt-in-v3"].includes(consentVersion) ||
      ((consentVersion === "dream-learning-opt-in-v2" || consentVersion === "dream-learning-opt-in-v3") && !validSourceGroupId)) {
    return json(res, 400, { error: "A valid single question, answer and learning-consent version are required." });
  }
  const endpoint = process.env.DREAM_LEARNING_INGEST_URL;
  const token = process.env.DREAM_LEARNING_INGEST_TOKEN;
  const anonKey = process.env.DREAM_LEARNING_SUPABASE_ANON_KEY;
  if (!endpoint || !token || !anonKey) {
    return json(res, 503, { error: "Dream AI learning collection is not configured." });
  }
  let target;
  try {
    target = new URL(endpoint);
    if (target.protocol !== "https:" || !target.hostname.endsWith(".supabase.co")) throw new Error("invalid");
  } catch {
    return json(res, 503, { error: "Dream AI learning collection is not configured." });
  }
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000);
  try {
    const upstream = await fetch(target, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        apikey: anonKey,
        authorization: `Bearer ${anonKey}`,
        "x-dream-learning-token": token
      },
      body: JSON.stringify({
        mode, prompt, response: responseText, consentVersion,
        ...(consentVersion !== "dream-learning-opt-in-v1" ? { sourceGroupId } : {})
      }),
      redirect: "error",
      signal: controller.signal
    });
    const result = await upstream.json().catch(() => ({}));
    if (!upstream.ok) {
      console.error("DREAM learning ingest failed:", upstream.status);
      return json(res, 502, { error: "Could not submit this example. Please try again." });
    }
    return json(res, upstream.status === 201 ? 201 : 200, {
      accepted: result.accepted === true,
      status: result.status || (result.duplicate ? "already-submitted" : "pending")
    });
  } catch (error) {
    console.error("DREAM learning ingest request failed:", error.name || "error");
    return json(res, 502, { error: "Learning collection is temporarily unavailable." });
  } finally {
    clearTimeout(timeout);
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
      modelReady: runtime.ready,
      provider: runtime.provider || null,
      modelStatus: runtime.status || (runtime.ready ? "ready" : "unavailable")
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
  if (req.method === "GET" && url.pathname === "/api/plugins") {
    return json(res, 200, getDreamPluginCatalog());
  }
  if (req.method === "GET" && url.pathname === "/api/account/meta/status") {
    try {
      return json(res, 200, await metaAccountService.status(req, res));
    } catch {
      return json(res, 503, { metaOAuthConfigured: false, operatorTokenConfigured: false, facebook: { connected: false }, threads: { connected: false } });
    }
  }
  if (req.method === "GET" && url.pathname === "/api/account/meta/facebook/start") {
    try { return metaAccountService.begin(req, res, "facebook"); }
    catch {
      res.writeHead(303, { ...SECURITY_HEADERS, location: "/?meta=setup-required", "cache-control": "no-store" });
      return res.end();
    }
  }
  if (req.method === "GET" && url.pathname === "/api/account/meta/facebook/callback") {
    try { return await metaAccountService.callback(req, res, url, "facebook"); }
    catch {
      res.writeHead(303, { ...SECURITY_HEADERS, location: "/?meta=failed", "cache-control": "no-store" });
      return res.end();
    }
  }
  if (req.method === "GET" && url.pathname === "/api/account/meta/threads/start") {
    try { return metaAccountService.begin(req, res, "threads"); }
    catch {
      res.writeHead(303, { ...SECURITY_HEADERS, location: "/?meta=setup-required", "cache-control": "no-store" });
      return res.end();
    }
  }
  if (req.method === "GET" && url.pathname === "/api/account/meta/threads/callback") {
    try { return await metaAccountService.callback(req, res, url, "threads"); }
    catch {
      res.writeHead(303, { ...SECURITY_HEADERS, location: "/?meta=failed", "cache-control": "no-store" });
      return res.end();
    }
  }
  if (req.method === "POST" && url.pathname === "/api/account/meta/disconnect") {
    try {
      validateBrowserRequest(req);
      if (!enforceApiRateLimit(req, res)) return;
      const body = await readJson(req);
      const provider = body?.provider || "all";
      if (! ["all", "facebook", "threads"].includes(provider)) {
        return json(res, 400, { error: "Choose all, facebook or threads." });
      }
      return json(res, 200, await metaAccountService.disconnect(req, res, provider));
    } catch (error) {
      return json(res, error.status || 500, { error: "Could not safely disconnect Meta. Please retry." });
    }
  }
  if (req.method === "GET" && url.pathname === "/api/account/status") {
    try {
      return json(res, 200, await googleAccountService.status(req, res));
    } catch {
      return json(res, 503, { googleOAuthConfigured: false, gmail: { connected: false, email: null, reason: "account_status_unavailable" } });
    }
  }
  if (req.method === "GET" && url.pathname === "/api/account/google/start") {
    try { return googleAccountService.begin(req, res); }
    catch {
      res.writeHead(303, { ...SECURITY_HEADERS, location: "/?account=setup-required", "cache-control": "no-store" });
      return res.end();
    }
  }
  if (req.method === "GET" && url.pathname === "/api/account/google/callback") {
    try { return await googleAccountService.callback(req, res, url); }
    catch {
      res.writeHead(303, { ...SECURITY_HEADERS, location: "/?account=failed", "cache-control": "no-store" });
      return res.end();
    }
  }
  if (req.method === "POST" && url.pathname === "/api/account/google/disconnect") {
    try {
      validateBrowserRequest(req);
      if (!enforceApiRateLimit(req, res)) return;
      await readJson(req);
      return json(res, 200, await googleAccountService.disconnect(req, res));
    } catch (error) {
      return json(res, error.status || 500, { error: "Could not safely disconnect Gmail. Please retry." });
    }
  }
  if (req.method === "POST" && url.pathname === "/api/learning/examples") {
    try {
      return await runWithClientConcurrency(req, res, handleLearningExample);
    } catch (error) {
      return json(res, error.status || 500, {
        error: error.status ? error.message : "An unexpected server error occurred."
      });
    }
  }
  if (req.method === "POST" && url.pathname === "/api/research") {
    try {
      return await runWithClientConcurrency(req, res, handleResearch);
    } catch (error) {
      return json(res, error.status || 500, { error: error.status ? error.message : "An unexpected server error occurred." });
    }
  }
  if (req.method === "POST" && url.pathname === "/api/chat") {
    try {
      return await runWithClientConcurrency(req, res, handleChat);
    } catch (error) {
      return json(res, error.status || 500, {
        error: error.status ? error.message : "An unexpected server error occurred."
      });
    }
  }
  if (req.method === "GET") return serveStatic(url.pathname, res);
  res.writeHead(405, { ...SECURITY_HEADERS, allow: "GET, POST" }).end("Method not allowed");
});

async function runStartupModelSmokeTest() {
  try {
    const result = await generateDreamReply(
      [{ role: "user", content: "Reply with exactly: DREAM AI MODEL CONNECTION OK" }],
      "general"
    );
    const reply = String(result?.reply || "").replace(/[\\r\\n\\t]+/g, " ").trim();
    console.info("DREAM_MODEL_STARTUP_SMOKE_TEST completed:", JSON.stringify({
      source: result?.source || "unknown",
      replyLength: reply.length,
      replyPreview: reply.slice(0, 120)
    }));
  } catch (error) {
    console.error("DREAM_MODEL_STARTUP_SMOKE_TEST failed:", error?.message || "unknown model error");
  }
}

server.listen(PORT, "0.0.0.0", () => {
  console.log(`DREAM AI listening on port ${PORT}`);
  try {
    const modelConfig = resolveModelConfig();
    console.info("DREAM_MODEL_CONFIGURATION", JSON.stringify({
      provider: modelConfig.provider,
      model: modelConfig.modelName,
      endpointHost: new URL(modelConfig.url).host,
      tokenConfigured: Boolean(modelConfig.token)
    }));
  } catch (error) {
    console.error("DREAM_MODEL_CONFIGURATION invalid:", error?.message || "unknown configuration error");
  }
  if (process.env.DREAM_MODEL_STARTUP_SMOKE_TEST === "true") {
    console.log("DREAM_MODEL_STARTUP_SMOKE_TEST starting one controlled inference check.");
    setTimeout(() => { void runStartupModelSmokeTest(); }, 1500);
  }
});

function shutdown() {
  server.close(() => process.exit(0));
}
process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);
