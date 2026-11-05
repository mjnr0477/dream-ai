import { createHash, randomBytes } from "node:crypto";
import { isIP } from "node:net";

const CF_RAY_PATTERN = /^[a-f0-9]{16,32}-[a-z]{3}$/i;
const JSON_CONTENT_TYPE = /^application\/json(?:\s*;|$)/i;

function trustedClientAddress(req) {
  const cfRay = req.headers?.["cf-ray"];
  const cfIp = req.headers?.["cf-connecting-ip"];
  if (
    typeof cfRay === "string" &&
    CF_RAY_PATTERN.test(cfRay.trim()) &&
    typeof cfIp === "string" &&
    isIP(cfIp.trim())
  ) {
    return cfIp.trim();
  }
  // Do not trust X-Forwarded-For: clients can supply forged values.
  return typeof req.socket?.remoteAddress === "string" && req.socket.remoteAddress
    ? req.socket.remoteAddress
    : "unknown";
}

function expectedOrigin(req, env = process.env) {
  const configured = env.PUBLIC_ORIGIN || env.RENDER_EXTERNAL_URL;
  if (configured) {
    try {
      return new URL(configured).origin;
    } catch {
      return null;
    }
  }
  const host = req.headers?.host;
  if (typeof host !== "string" || !host.trim()) return null;
  const forwardedProto = req.headers?.["x-forwarded-proto"];
  const firstProto = typeof forwardedProto === "string"
    ? forwardedProto.split(",")[0].trim().toLowerCase()
    : "";
  const proto = firstProto === "http" || firstProto === "https"
    ? firstProto
    : /^(localhost|127\.0\.0\.1|\[::1\])(?::\d+)?$/i.test(host)
      ? "http"
      : "https";
  try {
    return new URL(`${proto}://${host}`).origin;
  } catch {
    return null;
  }
}

export function validateApiRequest(req, env = process.env) {
  const contentType = req.headers?.["content-type"];
  if (typeof contentType !== "string" || !JSON_CONTENT_TYPE.test(contentType.trim())) {
    return { status: 415, message: "API requests must use Content-Type: application/json." };
  }

  const fetchSite = req.headers?.["sec-fetch-site"];
  if (typeof fetchSite === "string" && fetchSite.trim().toLowerCase() === "cross-site") {
    return { status: 403, message: "Cross-site API requests are not allowed." };
  }

  const origin = req.headers?.origin;
  if (origin !== undefined) {
    if (typeof origin !== "string") {
      return { status: 403, message: "Invalid request origin." };
    }
    let actualOrigin;
    try {
      actualOrigin = new URL(origin).origin;
    } catch {
      return { status: 403, message: "Invalid request origin." };
    }
    const allowedOrigin = expectedOrigin(req, env);
    if (!allowedOrigin || actualOrigin !== allowedOrigin) {
      return { status: 403, message: "Cross-origin API requests are not allowed." };
    }
  }
  return null;
}

export class RequestRateLimiter {
  #buckets = new Map();
  #active = new Map();
  #salt;

  constructor({ limit = 20, windowMs = 60_000, maxBuckets = 10_000, maxConcurrentPerClient = 1, salt = randomBytes(32) } = {}) {
    if (!Number.isInteger(limit) || limit < 1) throw new TypeError("limit must be a positive integer");
    if (!Number.isInteger(windowMs) || windowMs < 1) throw new TypeError("windowMs must be a positive integer");
    if (!Number.isInteger(maxBuckets) || maxBuckets < 1) throw new TypeError("maxBuckets must be a positive integer");
    if (!Number.isInteger(maxConcurrentPerClient) || maxConcurrentPerClient < 1) throw new TypeError("maxConcurrentPerClient must be a positive integer");
    if (!Buffer.isBuffer(salt) || salt.length < 16) throw new TypeError("salt must contain at least 16 bytes");
    this.limit = limit;
    this.windowMs = windowMs;
    this.maxBuckets = maxBuckets;
    this.maxConcurrentPerClient = maxConcurrentPerClient;
    this.#salt = salt;
  }

  acquire(req, now = Date.now()) {
    const address = trustedClientAddress(req);
    const key = createHash("sha256").update(this.#salt).update(address).digest("hex");
    const budget = this.consume(req, now);
    if (!budget.allowed) return budget;

    const active = this.#active.get(key) || 0;
    if (active >= this.maxConcurrentPerClient) {
      return { allowed: false, remaining: budget.remaining, retryAfterSeconds: 3, reason: "concurrency" };
    }
    this.#active.set(key, active + 1);
    let released = false;
    return {
      ...budget,
      release: () => {
        if (released) return;
        released = true;
        const current = this.#active.get(key) || 0;
        if (current <= 1) this.#active.delete(key);
        else this.#active.set(key, current - 1);
      }
    };
  }

  consume(req, now = Date.now()) {
    const address = trustedClientAddress(req);
    const key = createHash("sha256").update(this.#salt).update(address).digest("hex");
    let bucket = this.#buckets.get(key);

    if (!bucket || now - bucket.windowStart >= this.windowMs) {
      if (!bucket && this.#buckets.size >= this.maxBuckets) {
        for (const [storedKey, storedBucket] of this.#buckets) {
          if (now - storedBucket.windowStart >= this.windowMs) this.#buckets.delete(storedKey);
        }
        if (this.#buckets.size >= this.maxBuckets) {
          return { allowed: false, remaining: 0, retryAfterSeconds: Math.ceil(this.windowMs / 1000) };
        }
      }
      bucket = { windowStart: now, count: 1 };
      this.#buckets.set(key, bucket);
      return { allowed: true, remaining: this.limit - 1, retryAfterSeconds: 0 };
    }

    if (bucket.count >= this.limit) {
      return {
        allowed: false,
        remaining: 0,
        retryAfterSeconds: Math.max(1, Math.ceil((this.windowMs - (now - bucket.windowStart)) / 1000))
      };
    }
    bucket.count += 1;
    return { allowed: true, remaining: this.limit - bucket.count, retryAfterSeconds: 0 };
  }
}
