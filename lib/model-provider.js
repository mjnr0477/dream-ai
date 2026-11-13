const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "::1"]);
const MAX_PENDING_MODEL_REQUESTS = 2;
let pendingModelRequests = 0;
let modelRequestQueue = Promise.resolve();

function runSerializedModelRequest(operation) {
  if (pendingModelRequests >= MAX_PENDING_MODEL_REQUESTS) {
    throw new Error("DREAM AI is handling other requests. Please retry shortly.");
  }

  pendingModelRequests += 1;
  const currentRequest = modelRequestQueue.then(operation, operation);
  modelRequestQueue = currentRequest.then(() => undefined, () => undefined);

  return currentRequest.finally(() => {
    pendingModelRequests -= 1;
  });
}

function parseModelUrl(value, provider) {
  let url;
  try { url = new URL(value); }
  catch { throw new Error("DREAM_MODEL_URL must be a valid URL."); }
  if (url.username || url.password) throw new Error("DREAM_MODEL_URL must not contain embedded credentials.");
  const hostname = url.hostname.toLowerCase().replace(/^\[|\]$/g, "");
  const loopback = LOOPBACK_HOSTS.has(hostname);
  if (url.protocol !== "https:" && !(url.protocol === "http:" && loopback)) {
    throw new Error("DREAM_MODEL_URL must use HTTPS unless it targets loopback for local development.");
  }
  if (provider === "native" && url.protocol !== "https:" && !loopback) {
    throw new Error("The native model endpoint must use HTTPS unless it targets loopback.");
  }
  return url;
}

export function resolveModelConfig(env = process.env) {
  const provider = String(env.DREAM_MODEL_PROVIDER || "native").trim().toLowerCase();
  if (!["native", "openai-compatible", "cloudflare-workers-ai"].includes(provider)) {
    throw new Error("DREAM_MODEL_PROVIDER must be native, openai-compatible or cloudflare-workers-ai.");
  }

  const timeoutValue = env.DREAM_MODEL_TIMEOUT_MS === undefined ? 60000 : Number(env.DREAM_MODEL_TIMEOUT_MS);
  if (!Number.isSafeInteger(timeoutValue) || timeoutValue < 1000 || timeoutValue > 120000) {
    throw new Error("DREAM_MODEL_TIMEOUT_MS must be an integer from 1000 to 120000.");
  }

  if (provider === "cloudflare-workers-ai") {
    const accountId = typeof env.CLOUDFLARE_ACCOUNT_ID === "string" ? env.CLOUDFLARE_ACCOUNT_ID.trim() : "";
    const token = typeof env.CLOUDFLARE_API_TOKEN === "string" ? env.CLOUDFLARE_API_TOKEN.trim() : "";
    if (!/^[a-f0-9]{32}$/i.test(accountId)) {
      throw new Error("CLOUDFLARE_ACCOUNT_ID must be a 32-character Cloudflare account ID.");
    }
    if (!token) throw new Error("CLOUDFLARE_API_TOKEN is required.");
    const modelName = typeof env.DREAM_MODEL_NAME === "string" && env.DREAM_MODEL_NAME.trim()
      ? env.DREAM_MODEL_NAME.trim()
      : "@cf/meta/llama-3.1-8b-instruct";
    if (modelName.length > 200) throw new Error("DREAM_MODEL_NAME must contain at most 200 characters.");
    const url = "https://api.cloudflare.com/client/v4/accounts/" + accountId + "/ai/v1/chat/completions";
    return Object.freeze({ provider, modelName, url, token, timeoutMs: timeoutValue });
  }

  const modelName = typeof env.DREAM_MODEL_NAME === "string" ? env.DREAM_MODEL_NAME.trim() : "";
  if (!modelName || modelName.length > 200) throw new Error("DREAM_MODEL_NAME must contain 1 to 200 characters.");
  if (typeof env.DREAM_MODEL_URL !== "string" || !env.DREAM_MODEL_URL.trim()) {
    throw new Error("DREAM_MODEL_URL is required.");
  }
  const url = parseModelUrl(env.DREAM_MODEL_URL.trim(), provider);
  const token = typeof env.DREAM_MODEL_TOKEN === "string" ? env.DREAM_MODEL_TOKEN : "";
  const path = url.pathname.replace(/\/+$/, "");
  const apiPath = path.endsWith("/chat/completions") ? path : path.endsWith("/v1") ? path + "/chat/completions" : path + "/v1/chat/completions";
  url.pathname = apiPath;
  url.search = "";
  url.hash = "";
  return Object.freeze({ provider, modelName, url: url.href, token, timeoutMs: timeoutValue });
}
const RETRYABLE_MODEL_STATUSES = new Set([408, 425, 429, 500, 502, 503, 504]);
const MODEL_RETRY_DELAYS_MS = [500, 1200];

function getModelRetryDelay(response, attempt) {
  const fallback = MODEL_RETRY_DELAYS_MS[attempt] ?? MODEL_RETRY_DELAYS_MS.at(-1);
  let retryAfter = "";
  try { retryAfter = response.headers?.get?.("retry-after") || ""; } catch {}
  if (!retryAfter) return fallback;

  const seconds = Number(retryAfter);
  const requestedMs = Number.isFinite(seconds)
    ? seconds * 1000
    : Date.parse(retryAfter) - Date.now();
  if (!Number.isFinite(requestedMs) || requestedMs <= 0) return fallback;
  return Math.min(5000, Math.max(250, requestedMs));
}

const waitForModelRetry = ms => new Promise(resolve => setTimeout(resolve, ms));

async function readSafeProviderError(response, token = "") {
  let body = "";
  try {
    if (typeof response.text === "function") body = await response.text();
  } catch {
    return "";
  }
  if (typeof body !== "string" || !body.trim()) return "";

  let safe = body
    .replace(/Bearer\s+[^\s"'<>]+/gi, "Bearer [redacted]")
    .replace(/hf_[A-Za-z0-9]{10,}/g, "[redacted]")
    .replace(/(["']?(?:access_token|api_key|authorization|token)["']?\s*[:=]\s*["']?)[^\s,"'}]+/gi, "$1[redacted]");
  if (token) safe = safe.split(token).join("[redacted]");
  return safe.replace(/\s+/g, " ").trim().slice(0, 400);
}

export async function requestChatCompletion({
  config, messages, temperature, fetchImpl = fetch, sleepImpl = waitForModelRetry
}) {
  return runSerializedModelRequest(async () => {
    for (let attempt = 0; attempt <= MODEL_RETRY_DELAYS_MS.length; attempt += 1) {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), config.timeoutMs);
      try {
        const headers = { "content-type": "application/json", accept: "application/json" };
        if (config.token) headers.authorization = "Bearer " + config.token;
        const response = await fetchImpl(config.url, {
          method: "POST",
          headers,
          redirect: "error",
          body: JSON.stringify({ model: config.modelName, messages, temperature }),
          signal: controller.signal
        });
        if (!response.ok) {
          if (RETRYABLE_MODEL_STATUSES.has(response.status) && attempt < MODEL_RETRY_DELAYS_MS.length) {
            await sleepImpl(getModelRetryDelay(response, attempt));
            continue;
          }
          const detail = await readSafeProviderError(response, config.token);
          throw new Error(
            "Model provider returned HTTP " + response.status +
            (detail ? ". Upstream detail: " + detail : ".")
          );
        }
        let data;
        try { data = await response.json(); }
        catch { throw new Error("Model provider returned invalid JSON."); }
        const reply = data?.choices?.[0]?.message?.content;
        if (typeof reply !== "string" || !reply.trim()) throw new Error("Model provider returned an empty response.");
        return reply.trim();
      } catch (error) {
        if (error?.name === "AbortError") throw new Error("Model inference timed out. Please try again.");
        if (error?.name === "TypeError") throw new Error("Model provider request failed. Check endpoint availability and network configuration.");
        throw error;
      } finally {
        clearTimeout(timeout);
      }
    }
    throw new Error("Model provider request failed after retries.");
  });
}
