import test from "node:test";
import assert from "node:assert/strict";
import { getClientIp } from "../lib/client-ip.js";

test("uses the single Cloudflare client IP and ignores caller-controlled X-Forwarded-For", () => {
  assert.equal(getClientIp({
    headers: {
      "cf-connecting-ip": "198.51.100.10",
      "x-forwarded-for": "203.0.113.1, 203.0.113.2"
    },
    socket: { remoteAddress: "10.0.0.4" }
  }), "198.51.100.10");
});

test("falls back to the socket peer rather than an untrusted X-Forwarded-For value", () => {
  assert.equal(getClientIp({
    headers: { "x-forwarded-for": "203.0.113.9" },
    socket: { remoteAddress: "198.51.100.20" }
  }), "198.51.100.20");
});

test("rejects malformed Cloudflare and socket addresses", () => {
  assert.equal(getClientIp({
    headers: { "cf-connecting-ip": "not-an-ip", "x-forwarded-for": "203.0.113.9" },
    socket: { remoteAddress: "invalid" }
  }), "unknown");
});
