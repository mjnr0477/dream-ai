const MODES = {
  general: "Answer clearly and helpfully. Be honest about uncertainty and distinguish verified facts from assumptions.",
  coding: "Act as a senior software engineer. Produce complete, runnable code when requested; explain assumptions, include tests, consider security, and never claim code was executed unless it was.",
  learning: "Act as a patient expert tutor. Explain concepts step by step, adapt to the learner, use examples, and check understanding when useful."
};

export function isDreamEngineConfigured(env = process.env) {
  return Boolean(env.DREAM_MODEL_URL && env.DREAM_MODEL_NAME);
}

function nativeHealthUrl(modelUrl) {
  const health = new URL(modelUrl);
  const path = health.pathname.replace(/\/+$/, "");
  const apiRoot = path
    .replace(/\/v1(?:\/chat\/completions)?$/, "")
    .replace(/\/chat\/completions$/, "");
  health.pathname = apiRoot + "/health";
  health.search = "";
  health.hash = "";
  return health;
}

async function verifyNativeRuntime(modelUrl) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);
  try {
    const response = await fetch(nativeHealthUrl(modelUrl), {
      method: "GET",
      headers: { accept: "application/json" },
      signal: controller.signal
    });
    if (!response.ok) {
      throw new Error("Native runtime health check failed.");
    }
    const health = await response.json();
    if (health?.ok !== true || health?.engine !== "dream-native-transformer") {
      throw new Error(
        "Configured runtime is not the native DREAM transformer. External AI providers are not accepted."
      );
    }
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("Native DREAM runtime identity check timed out.");
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

export async function generateDreamReply(messages, mode = "general", env = process.env) {
  if (!isDreamEngineConfigured(env)) {
    return {
      reply: "DREAM AI's application is online, but its own model runtime is not connected yet. The chat interface is not a trained AI model. Connect a self-hosted DREAM model runtime to enable model-generated answers.",
      source: "dream-engine-unconfigured"
    };
  }

  let endpoint;
  try {
    endpoint = new URL(env.DREAM_MODEL_URL);
  } catch {
    throw new Error("DREAM_MODEL_URL must be a valid URL.");
  }
  if (!["https:", "http:"].includes(endpoint.protocol)) {
    throw new Error("DREAM_MODEL_URL must use HTTP or HTTPS.");
  }

  const base = env.DREAM_MODEL_URL.replace(/\/+$/, "");
  const url = base.endsWith("/chat/completions")
    ? base
    : `${base}/chat/completions`;
  await verifyNativeRuntime(env.DREAM_MODEL_URL);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);

  try {
    const headers = { "content-type": "application/json" };
    if (env.DREAM_MODEL_TOKEN) {
      headers.authorization = `Bearer ${env.DREAM_MODEL_TOKEN}`;
    }

    const response = await fetch(url, {
      method: "POST",
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
