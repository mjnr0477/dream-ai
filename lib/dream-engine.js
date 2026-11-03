import { isIP } from "node:net";

const MODES = {
  general: "Answer clearly and helpfully. Be honest about uncertainty and distinguish verified facts from assumptions.",
  coding: "Act as a senior software engineer. Produce complete, runnable code when requested; explain assumptions, include tests, consider security, and never claim code was executed unless it was.",
  learning: "Act as a patient expert tutor. Explain concepts step by step, adapt to the learner, use examples, and check understanding when useful."
};

export function isDreamEngineConfigured(env = process.env) {
  return Boolean(env.DREAM_MODEL_URL && env.DREAM_MODEL_NAME);
}

export function validateRuntimeUrl(value) {
  let endpoint;
  try {
    endpoint = new URL(value);
  } catch {
    throw new Error("DREAM_MODEL_URL must be a valid URL.");
  }
  if (!["https:", "http:"].includes(endpoint.protocol)) {
    throw new Error("DREAM_MODEL_URL must use HTTP or HTTPS.");
  }

  const hostname = endpoint.hostname.toLowerCase().replace("[", "").replace("]", "");
  const ipv4Loopback = isIP(hostname) === 4 && Number(hostname.split(".")[0]) === 127;
  const loopback = hostname === "localhost" || hostname.endsWith(".localhost")
    || hostname === "::1" || ipv4Loopback;
  if (endpoint.protocol === "http:" && !loopback) {
    throw new Error("DREAM_MODEL_URL must use HTTPS unless the model runtime is on loopback.");
  }
  return endpoint;
}

export async function generateDreamReply(messages, mode = "general", env = process.env) {
  if (!isDreamEngineConfigured(env)) {
    return {
      reply: "DREAM AI's application is online, but its own model runtime is not connected yet. The chat interface is not a trained AI model. Connect a self-hosted DREAM model runtime to enable model-generated answers.",
      source: "dream-engine-unconfigured"
    };
  }

  const endpoint = validateRuntimeUrl(env.DREAM_MODEL_URL);
  const base = env.DREAM_MODEL_URL.replace(/\\/+$/, "");
  const url = base.endsWith("/chat/completions")
    ? base
    : `${base}/chat/completions`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);

  try {
    const headers = { "content-type": "application/json" };
    if (env.DREAM_MODEL_TOKEN) {
      headers.authorization = `Bearer ${env.DREAM_MODEL_TOKEN}`;
    }

    const response = await fetch(url, {
      method: "POST",
      redirect: "error",
      headers,
      body: JSON.stringify({
        model: env.DREAM_MODEL_NAME,
        messages: [
          {
            role: "system",
            content: `You are DREAM AI, an independent AI assistant. Be helpful, accurate and transparent. Never claim to have performed actions you have not performed. ${MODES[mode] || MODES.general}`
          },
          ...messages
        ],
        temperature: 0.7
      }),
      signal: controller.signal
    });

    if (!response.ok) {
      const detail = (await response.text()).slice(0, 400);
      console.error("DREAM model runtime returned", response.status, detail);
      throw new Error("DREAM AI's model runtime could not complete the request.");
    }

    const data = await response.json();
    const reply = data?.choices?.[0]?.message?.content;
    if (typeof reply !== "string" || !reply.trim()) {
      throw new Error("DREAM AI's model runtime returned an empty response.");
    }

    return { reply: reply.trim(), source: "dream-model" };
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("DREAM AI's model runtime took too long to respond. Please try again.");
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}
