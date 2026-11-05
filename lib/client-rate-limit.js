import { createHash, randomBytes } from "node:crypto";
import { isIP } from "node:net";

const CF_RAY_PATTERN = /^[a-f0-9]{16,32}-[a-z]{3}$/i;
const LOOPBACK = new Set(["127.0.0.1", "::1", "::ffff:127.0.0.1"]);

function clientAddress(req) {
  const ray = req.headers?.["cf-ray"];
  const cfIp = req.headers?.["cf-connecting-ip"];
  if (
    typeof ray === "string" &&
    CF_RAY_PATTERN.test(ray.trim()) &&
    typeof cfIp === "string" &&
    isIP(cfIp.trim())
  ) {
    return cfIp.trim();
  }
  const remote = req.socket?.remoteAddress;
  return typeof remote === "string" && isIP(remote) ? remote : "unknown";
}

export class ClientRateLimiter {
  #buckets = new Map();
  #active = new Map();
  #salt;

  constructor({
    capacity = 20,
    refillPerMinute = 20,
    idleMs = 10 * 60 * 1000,
    maxBuckets = 5000,
    maxConcurrentPerClient = 1,
    salt = randomBytes(32)
  } = {}) {
    if (!Number.isFinite(capacity) || capacity < 1) throw new TypeError("capacity must be positive");
    if (!Number.isFinite(refillPerMinute) || refillPerMinute <= 0) throw new TypeError("refillPerMinute must be positive");
    if (!Number.isInteger(idleMs) || idleMs < 1) throw new TypeError("idleMs must be a positive integer");
    if (!Number.isInteger(maxBuckets) || maxBuckets < 1) throw new TypeError("maxBuckets must be a positive integer");
    if (!Number.isInteger(maxConcurrentPerClient) || maxConcurrentPerClient < 1) {
      throw new TypeError("maxConcurrentPerClient must be a positive integer");
    }
    if (!Buffer.isBuffer(salt) || salt.length < 16) throw new TypeError("salt must contain at least 16 bytes");
    this.capacity = capacity;
    this.refillPerMinute = refillPerMinute;
    this.refillPerMs = refillPerMinute / 60_000;
    this.idleMs = idleMs;
    this.maxBuckets = maxBuckets;
    this.maxConcurrentPerClient = maxConcurrentPerClient;
    this.#salt = salt;
  }

  acquire(req, now = Date.now()) {
    const address = clientAddress(req);
    if (LOOPBACK.has(address)) {
      return { allowed: true, retryAfterSeconds: 0, release() {} };
    }

    const key = createHash("sha256").update(this.#salt).update(address).digest("hex");
    const active = this.#active.get(key) || 0;
    if (active >= this.maxConcurrentPerClient) {
      return { allowed: false, retryAfterSeconds: 3, reason: "concurrency" };
    }

    let bucket = this.#buckets.get(key);
    if (!bucket) {
      if (this.#buckets.size >= this.maxBuckets) {
        for (const [storedKey, stored] of this.#buckets) {
          if (now - stored.lastSeenAt > this.idleMs) this.#buckets.delete(storedKey);
        }
        if (this.#buckets.size >= this.maxBuckets) {
          return { allowed: false, retryAfterSeconds: Math.max(1, Math.ceil(this.idleMs / 1000)), reason: "capacity" };
        }
      }
      bucket = { tokens: this.capacity, updatedAt: now, lastSeenAt: now };
    }

    const elapsed = Math.max(0, now - bucket.updatedAt);
    bucket.tokens = Math.min(this.capacity, bucket.tokens + elapsed * this.refillPerMs);
    bucket.updatedAt = now;
    bucket.lastSeenAt = now;
    if (bucket.tokens < 1) {
      this.#buckets.set(key, bucket);
      return {
        allowed: false,
        retryAfterSeconds: Math.max(1, Math.ceil((1 - bucket.tokens) / this.refillPerMs / 1000)),
        reason: "rate"
      };
    }

    bucket.tokens -= 1;
    this.#buckets.set(key, bucket);
    this.#active.set(key, active + 1);
    let released = false;
    return {
      allowed: true,
      retryAfterSeconds: 0,
      release: () => {
        if (released) return;
        released = true;
        const current = this.#active.get(key) || 0;
        if (current <= 1) this.#active.delete(key);
        else this.#active.set(key, current - 1);
      }
    };
  }
}
