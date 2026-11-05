import test from "node:test";
import assert from "node:assert/strict";
import { ClientConcurrencyLimiter } from "../lib/client-limiter.js";

test("one client cannot occupy multiple concurrent inference slots", () => {
  const limiter = new ClientConcurrencyLimiter(1);
  const first = limiter.acquire("198.51.100.10");
  assert.equal(first.allowed, true);
  assert.equal(limiter.acquire("198.51.100.10").allowed, false);
  first.release();
  assert.equal(limiter.acquire("198.51.100.10").allowed, true);
});

test("different clients can make progress independently", () => {
  const limiter = new ClientConcurrencyLimiter(1);
  const first = limiter.acquire("198.51.100.10");
  const second = limiter.acquire("198.51.100.11");
  assert.equal(first.allowed, true);
  assert.equal(second.allowed, true);
  first.release();
  second.release();
});

test("release is idempotent and unknown identity does not lock all callers", () => {
  const limiter = new ClientConcurrencyLimiter(1);
  const slot = limiter.acquire("198.51.100.10");
  slot.release();
  slot.release();
  assert.equal(limiter.acquire("198.51.100.10").allowed, true);
  assert.equal(limiter.acquire("unknown").allowed, true);
  assert.equal(limiter.acquire("unknown").allowed, true);
});

test("limiter rejects invalid capacity configuration", () => {
  for (const value of (0, -1, 1.5, true)) {
    assert.throws(() => new ClientConcurrencyLimiter(value), /positive integer/);
  }
});
