const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "::1"]);
const GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models";
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

function timeoutFromEnv(env) {
  const timeoutValue = env.DREAM_MODEL_TIMEOUT_MS === undefined ? 60000 : Number(env.DREAM_MODEL_TIMEOUT_MS);
  if (!Number.isSafeInteger(timeoutValue) || timeoutValue < 1000 || timeoutValue > 120000) {
    throw new Error("DREAM_MODEL_TIMEOUT_MS must be an integer from 1000 to 120000.");
  }
  return timeoutValue;
}

export function resolveModelConfig(env = process.env) {
  const configuredProvider = typeof env.DREAM_MODEL_PROVIDER === "string"
    ? env.DREAM_MODEL_PROVIDER.trim().toLowerCase()
    : "";
  const hasEndpoint = Boolean(
    typeof env.DREAM_MODEL_URL === "string" && env.DREAM_MODEL_URL.trim() &&
    typeof env.DREAM_MODEL_NAME === "string" && env.DREAM_MODEL_NAME.trim()
  );
  // Keep DREAM's own runtime first. Use Gemini only as an explicit-credential
  // bridge when no native/compatible model endpoint has been configured.
  const provider = configuredProvider || (
    hasEndpoint ? "native" : env.DREAM_GOOGLE_GEMINI_API_KEY ? "gemini" : "native"
  );
  if (!["native", "openai-compatible", "gemini"].includes(provider)) {
    throw new Error("DREAM_MODEL_PROVIDER must be native, openai-compatible or gemini.");
  }

  const timeoutMs = timeoutFromEnv(env);
  if (provider === "gemini") {
    const modelName = String(env.DREAM_GEMINI_MODEL_NAME || "gemini-3.8-flash").trim();
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$/.test(modelName)) {
      throw new Error("DREAM_GEMINI_MODEL_NAME must be a valid Gemini model ID.");
    }
    const token = typeof env.DREAM_GOOGLE_GEMINI_API_KEY === "string" && env.DREAM_GOOGLE_GEMINI_API_KEY.trim()
      ? env.DREAM_GOOGLE_GEMINI_API_KEY.trim()
      : typeof env.DREAM_MODEL_TOKEN === "string" ? env.DREAM_MODEL_TOKEN.trim() : "";
    if (!token) throw new Error("DREAM_GOOGLE_GEMINI_API_KEY is required for the Gemini provider.");
    return Object.freeze({ provider, modelName, url: null, token, timeoutMs });
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
  return Object.freeze({ provider, modelName, url: url.href, token, timeoutMs });
}

async function requestGeminiCompletion({ config, messages, temperature, fetchImpl }) {
  const systemText = messages
    .filter(message => message.role === "system")
    .map(message => message.content)
    .join("\n\n")
    .trim();
  const contents = [];
  for (const message of messages) {
    if (message.role !== "user" && message.role !== "assistant") continue;
    const role = message.role === "assistant" ? "model" : "user";
    const previous = contents.at(-1);
    if (previous?.role === role) {
      previous.parts.push({ text: message.content });
    } else {
      contents.push({ role, parts: [{ text: message.content }] });
    }
  }
  if (!contents.length || contents.at(-1).role !== "user") {
    throw new Error("Gemini requests must end with a user message.");
  }

  const endpoint = new URL(
    GEMINI_MODELS_URL + "/" + encodeURIComponent(config.modelName) + ":generateContent"
  );
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.timeoutMs);
  try {
    const response = await fetchImpl(endpoint, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        accept: "application/json",
        "x-goog-api-key": config.token
      },
      redirect: "error",
      body: JSON.stringify({
        ...(systemText ? { systemInstruction: { parts: [{ text: systemText }] } } : {}),
        contents,
        generationConfig: { temperature }
      }),
      signal: controller.signal
    });
    if (!response.ok) throw new Error("Gemini provider returned HTTP " + response.status + ".");
    let data;
    try { data = await response.json(); }
    catch { throw new Error("Gemini provider returned invalid JSON."); }
    const parts = data?.candidates?.[0]?.content?.parts;
    const reply = Array.isArray(parts)
      ? parts.filter(part => part?.thought !== true && typeof part?.text === "string").map(part => part.text).join("\n")
      : "";
    if (!reply.trim()) throw new Error("Gemini provider returned an empty response.");
    return reply.trim();
  } catch (error) {
    if (error?.name === "AbortError") throw new Error("Gemini model inference timed out. Please try again.");
    if (error?.name === "TypeError") throw new Error("Gemini provider request failed. Check API availability and key configuration.");
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

export async function verifyGeminiModel(config, fetchImpl = fetch) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);
  try {
    const endpoint = new URL(GEMINI_MODELS_URL + "/" + encodeURIComponent(config.modelName));
    const response = await fetchImpl(endpoint, {
      method: "GET",
      headers: { accept: "application/json", "x-goog-api-key": config.token },
      redirect: "error",
      signal: controller.signal
    });
    if (!response.ok) throw new Error("Gemini model verification returned HTTP " + response.status + ".");
    const model = await response.json();
    const supported = Array.isArray(model?.supportedGenerationMethods)
      ? model.supportedGenerationMethods
      : [];
    if (!supported.includes("generateContent")) {
      throw new Error("Configured Gemini model does not support content generation.");
    }
    return model;
  } catch (error) {
    if (error?.name === "AbortError") throw new Error("Gemini model verification timed out.");
    if (error?.name === "TypeError") throw new Error("Gemini model verification request failed.");
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

export async function requestChatCompletion({ config, messages, temperature, fetchImpl = fetch }) {
  return runSerializedModelRequest(async () => {
  if (config.provider === "gemini") {
    return requestGeminiCompletion({ config, messages, temperature, fetchImpl });
  }

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
    if (!response.ok) throw new Error("Model provider returned HTTP " + response.status + ".");
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
  });
}