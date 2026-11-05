import test from "node:test";
import assert from "node:assert/strict";
import { ClientRateLimiter } from "../lib/client-rate-limit.js";

function request(headers = {}, remoteAddress = "10.0.0.1") {
  return { headers, socket: { remoteAddress } };
}

test("Cloudflare client identity wins and forged X-Forwarded-For is ignored", () => {
  const limiter = new ClientRateLimiter({ capacity: 1, refillPerMinute: 1, salt: Buffer.alloc(32, 1) });
  const trusted = request({
    "cf-ray": "0123456789abcdef-LHR",
    "cf-connecting-ip": "198.51.100.20",
    "x-forwarded-for": "203.0.113.99"
  });
  const sameClient = request({
    "cf-ray": "fedcba9876543210-LHR",
    "cf-connecting-ip": "198.51.100.20",
    "x-forwarded-for": "198.51.100.21"
  });
  const otherClient = request({
    "cf-ray": "aaaaaaaaaaaaaaaa-LHR",
    "cf-connecting-ip": "198.51.100.21",
    "x-forwarded-for": "198.51.100.20"
  });
  const first = limiter.acquire(trusted, 1_000);
  assert.equal(first.allowed, true);
  first.release();
  assert.equal(limiter.acquire(sameClient, 1_001).allowed, false);
  assert.equal(limiter.acquire(otherClient, 1_002).allowed, true);
});

test("invalid or absent Cloudflare identity falls back to the socket, not X-Forwarded-For", () => {
  const limiter = new ClientRateLimiter({ capacity: 1, refillPerMinute: 1, salt: Buffer.alloc(32, 2) });
  const direct = request({ "x-forwarded-for": "198.51.100.20" }, "10.0.0.7");
  const forged = request({
    "cf-ray": "invalid",
    "cf-connecting-ip": "198.51.100.20",
    "x-forwarded-for": "203.0.113.77"
  }, "10.0.0.7");
  const first = limiter.acquire(direct, 1_000);
  assert.equal(first.allowed, true);
  first.release();
  assert.equal(limiter.acquire(forged, 1_001).allowed, false);
});

test("one client cannot occupy multiple global API slots", () => {
  const limiter = new ClientRateLimiter({ capacity: 5, salt: Buffer.alloc(32, 3) });
  const client = request({
    "cf-ray": "0123456789abcdef-LHR",
    "cf-connecting-ip": "198.51.100.30"
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

test("token bucket refills gradually and reports retry time", () => {
  const limiter = new ClientRateLimiter({
    capacity: 2,
    refillPerMinute: 2,
    salt: Buffer.alloc(32, 4)
  });
  const client = request({
    "cf-ray": "0123456789abcdef-LHR",
    "cf-connecting-ip": "198.51.100.40"
  });
  const first = limiter.acquire(client, 1_000);
  first.release();
  const second = limiter.acquire(client, 1_001);
  second.release();
  const blocked = limiter.acquire(client, 1_002);
  assert.equal(blocked.allowed, false);
  assert.equal(blocked.retryAfterSeconds, 30);
  const refilled = limiter.acquire(client, 31_002);
  assert.equal(refilled.allowed, true);
  refilled.release();
});

test("bucket storage stays bounded and evicts idle client identities", () => {
  const limiter = new ClientRateLimiter({
    capacity: 1,
    refillPerMinute: 1,
    idleMs: 1_000,
    maxBuckets: 1,
    salt: Buffer.alloc(32, 5)
  });
  const one = request({ "cf-ray": "0123456789abcdef-LHR", "cf-connecting-ip": "198.51.100.50" });
  const two = request({ "cf-ray": "fedcba9876543210-LHR", "cf-connecting-ip": "198.51.100.51" });
  const first = limiter.acquire(one, 1_000);
  assert.equal(first.allowed, true);
  first.release();
  assert.equal(limiter.acquire(two, 1_001).allowed, false);
  const next = limiter.acquire(two, 2_002);
  assert.equal(next.allowed, true);
  next.release();
});

test("loopback development requests bypass the public quota", () => {
  const limiter = new ClientRateLimiter({ capacity: 1, salt: Buffer.alloc(32, 6) });
  assert.equal(limiter.acquire(request({}, "127.0.0.1"), 1_000).allowed, true);
  assert.equal(limiter.acquire(request({}, "::1"), 1_001).allowed, true);
});
