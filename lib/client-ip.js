import { isIP } from "node:net";

/**
 * Render's default onrender.com ingress is Cloudflare-backed. Cloudflare
 * supplies CF-Connecting-IP on edge-to-origin traffic. Do not use XFF for
 * security decisions: Render may preserve caller-supplied XFF entries.
 * If CF-Connecting-IP is absent, fall back to the socket peer (which may be
 * a shared proxy address and therefore less useful for per-client quotas).
 */
export function getClientIp(req) {
  const cloudflare = req?.headers?.["cf-connecting-ip"];
  const client = typeof cloudflare === "string" ? cloudflare.trim() : "";
  if (client && isIP(client)) return client;
  const remote = req?.socket?.remoteAddress;
  return typeof remote === "string" && isIP(remote) ? remote : "unknown";
}
