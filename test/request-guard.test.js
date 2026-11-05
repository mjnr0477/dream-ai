import test from "node:test";
import assert from "node:assert/strict";
import { RequestRateLimiter, validateApiRequest } from "../lib/request-guard.js";

function request(headers = {}, remoteAddress = "127.0.0.1") {
  return { headers, socket: { remoteAddress } };
}

test("requires application/json for API requests", () => {
  assert.equal(validateApiRequest(request({ "content-type": "text/plain" })).status, 415);
  assert.equal(validateApiRequest(request({})).status, 415);
  assert.equal(validateApiRequest(request({ "content-type": "application/jsonp" })).status, 415);
  assert.equal(validateApiRequest(request({ "content-type": "application/json; charset=utf-8" })), null);
});

test("rejects cross-site fetch metadata and hostile origins", () => {
  const env = { PUBLIC_ORIGIN: "https://dream-ai.example" };
  assert.equal(
    validateApiRequest(request({
      "content-type": "application/json",
      "sec-fetch-site": "cross-site"
    }), env).status,
    403
  );
  assert.equal(
    validateApiRequest(request({
      "content-type": "application/json",
      origin: "https://evil.example"
    }), env).status,
    403
  );
});

test("allows same-origin browser requests and originless non-browser JSON clients", () => {
  const env = { PUBLIC_ORIGIN: "https://dream-ai.example" };
  assert.equal(
    validateApiRequest(request({
      "content-type": "application/json",
      origin: "https://dream-ai.example",
      "sec-fetch-site": "same-origin"
    }), env),
    null
  );
  assert.equal(validateApiRequest(request({ "content-type": "application/json" }), env), null);
});

test("derives local development origin without requiring a public origin setting", () => {
  assert.equal(
    validateApiRequest(request({
      host: "127.0.0.1:3000",
      "content-type": "application/json",
      origin: "http://127.0.0.1:3000"
    }), {}),
    null
  );
});

test("rate limits by Cloudflare-verified client IP and ignores forged X-Forwarded-For", () => {
  const limiter = new RequestRateLimiter({ limit: 2, windowMs: 60_000, salt: Buffer.alloc(32, 7) });
  const first = request({
    "cf-ray": "0123456789abcdef-LHR",
    "cf-connecting-ip": "203.0.113.10",
    "x-forwarded-for": "198.51.100.99"
  });
  const sameClientDifferentForwarding = request({
    "cf-ray": "fedcba9876543210-LHR",
    "cf-connecting-ip": "203.0.113.10",
    "x-forwarded-for": "192.0.2.44"
  });
  const differentClient = request({
    "cf-ray": "aaaaaaaaaaaaaaaa-LHR",
    "cf-connecting-ip": "203.0.113.11",
    "x-forwarded-for": "203.0.113.10"
  });

  assert.equal(limiter.consume(first, 1_000).allowed, true);
  assert.equal(limiter.consume(sameClientDifferentForwarding, 1_001).allowed, true);
  const blocked = limiter.consume(first, 1_002);
  assert.equal(blocked.allowed, false);
  assert.ok(blocked.retryAfterSeconds > 0);
  assert.equal(limiter.consume(differentClient, 1_003).allowed, true);
});

test("uses the socket address when Cloudflare identity headers are absent or invalid", () => {
  const limiter = new RequestRateLimiter({ limit: 1, salt: Buffer.alloc(32, 9) });
  const direct = request({ "x-forwarded-for": "203.0.113.77" }, "10.0.0.1");
  const forgedCf = request({
    "cf-ray": "not-a-valid-ray",
    "cf-connecting-ip": "203.0.113.77"
  }, "10.0.0.1");
  assert.equal(limiter.consume(direct, 1_000).allowed, true);
  assert.equal(limiter.consume(forgedCf, 1_001).allowed, false);
});

test("rate-limit windows reset and bucket storage remains bounded", () => {
  const limiter = new RequestRateLimiter({
    limit: 1,
    windowMs: 1_000,
    maxBuckets: 1,
    salt: Buffer.alloc(32, 3)
  });
  assert.equal(limiter.consume(request({}, "10.0.0.1"), 1_000).allowed, true);
  assert.equal(limiter.consume(request({}, "10.0.0.2"), 1_001).allowed, false);
  assert.equal(limiter.consume(request({}, "10.0.0.1"), 2_001).allowed, true);
  assert.equal(limiter.consume(request({}, "10.0.0.2"), 2_002).allowed, true);
});

test("per-client concurrency allows only one active request and releases its slot", () => {
  const limiter = new RequestRateLimiter({
    limit: 5,
    maxConcurrentPerClient: 1,
    salt: Buffer.alloc(32, 4)
  });
  const client = request({
    "cf-ray": "0123456789abcdef-LHR",
    "cf-connecting-ip": "203.0.113.20"
  });
  const first = limiter.acquire(client, 1_000);
  assert.equal(first.allowed, true);
  const second = limiter.acquire(client, 1_001);
  assert.equal(second.allowed, false);
  assert.equal(second.reason, "concurrency");
  first.release();
  const third = limiter.acquire(client, 1_002);
  assert.equal(third.allowed, true);
  third.release();
  third.release();
});
