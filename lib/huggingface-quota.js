export function getHuggingFaceToken(env = process.env) {
  const value = typeof env.HF_TOKEN === "string" ? env.HF_TOKEN.trim() : "";
  return value;
}

export async function logHuggingFaceZeroGpuQuota({
  env = process.env,
  fetchImpl = fetch,
  logImpl = console.info
} = {}) {
  try {
    const token = getHuggingFaceToken(env);
    if (!token) {
      logImpl("DREAM_HF_AUTH_CHECK", JSON.stringify({ tokenConfigured: false }));
      return;
    }
    const headers = { authorization: "Bearer " + token, accept: "application/json" };
    const identity = await fetchImpl("https://huggingface.co/api/whoami-v2", {
      headers, redirect: "error", signal: AbortSignal.timeout(8000)
    });
    logImpl("DREAM_HF_AUTH_CHECK", JSON.stringify({
      tokenConfigured: true, authenticated: identity.ok, status: identity.status
    }));
    if (!identity.ok) return;

    const quotaResponse = await fetchImpl("https://huggingface.co/api/spaces/zero-gpu/quota", {
      headers, signal: AbortSignal.timeout(8000)
    });
    if (!quotaResponse.ok) {
      logImpl("DREAM_HF_ZEROGPU_QUOTA", JSON.stringify({ available: false, status: quotaResponse.status }));
      return;
    }
    const quota = await quotaResponse.json();
    logImpl("DREAM_HF_ZEROGPU_QUOTA", JSON.stringify({
      available: true,
      baseGpuSeconds: Number.isFinite(Number(quota?.base)) ? Number(quota.base) : null,
      remainingGpuSeconds: Number.isFinite(Number(quota?.remaining)) ? Number(quota.remaining) : null,
      resetsAt: typeof quota?.resets_at === "string" ? quota.resets_at : null
    }));
  } catch (error) {
    logImpl("DREAM_HF_AUTH_CHECK unavailable:", error?.name || "network_error");
  }
}
