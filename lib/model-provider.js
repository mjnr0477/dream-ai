const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "::1"]);

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
  if (!["native", "openai-compatible"].includes(provider)) {
    throw new Error("DREAM_MODEL_PROVIDER must be native or openai-compatible.");
  }
  const modelName = typeof env.DREAM_MODEL_NAME === "string" ? env.DREAM_MODEL_NAME.trim() : "";
  if (!modelName || modelName.length > 200) throw new Error("DREAM_MODEL_NAME must contain 1 to 200 characters.");
  if (typeof env.DREAM_MODEL_URL !== "string" || !env.DREAM_MODEL_URL.trim()) {
    throw new Error("DREAM_MODEL_URL is required.");
  }
  const url = parseModelUrl(env.DREAM_MODEL_URL.trim(), provider);
  const timeoutValue = env.DREAM_MODEL_TIMEOUT_MS === undefined ? 60000 : Number(env.DREAM_MODEL_TIMEOUT_MS);
  if (!Number.isSafeInteger(timeoutValue) || timeoutValue < 1000 || timeoutValue > 120000) {
    throw new Error("DREAM_MODEL_TIMEOUT_MS must be an integer from 1000 to 120000.");
  }
  const token = typeof env.DREAM_MODEL_TOKEN === "string" ? env.DREAM_MODEL_TOKEN : "";
  const path = url.pathname.replace(/\/+$/, "");
  const apiPath = path.endsWith("/chat/completions") ? path : path.endsWith("/v1") ? path + "/chat/completions" : path + "/v1/chat/completions";
  url.pathname = apiPath;
  url.search = "";
  url.hash = "";
  return Object.freeze({ provider, modelName, url: url.href, token, timeoutMs: timeoutValue });
}

export async function requestChatCompletion({ config, messages, temperature, fetchImpl = fetch }) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.timeoutMs);
  try {
    const headers = { "content-type": "application/json", accept: "application/json" };
    if (config.token) headers.authorization = `Bearer ${config.token}`;
    const response = await fetchImpl(config.url, {
      method: "POST",
      headers,
      redirect: "error",
      body: JSON.stringify({ model: config.modelName, messages, temperature }),
      signal: controller.signal
    });
    if (!response.ok) throw new Error(`Model provider returned HTTP ${response.status}.`);
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
