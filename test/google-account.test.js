import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createGoogleAccountService, isGmailRequest } from "../lib/google-account.js";

function responseRecorder() {
  const headers = new Map();
  return {
    statusCode: 200,
    headers,
    getHeader(name) { return headers.get(String(name).toLowerCase()); },
    setHeader(name, value) { headers.set(String(name).toLowerCase(), value); },
    writeHead(status, values = {}) {
      this.statusCode = status;
      for (const [name, value] of Object.entries(values)) this.setHeader(name, value);
      return this;
    },
    end(body = "") { this.body = body; this.writableEnded = true; }
  };
}

function cookieValue(setCookie, name) {
  const values = Array.isArray(setCookie) ? setCookie : [setCookie];
  const line = values.find(value => String(value).startsWith(name + "="));
  return line ? String(line).split(";")[0] : "";
}

function jsonResponse(value, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json" }
  });
}

test("Gmail connector remains disabled without verified persistent encrypted storage", async () => {
  const directory = await mkdtemp(join(tmpdir(), "dream-account-gate-"));
  try {
    const service = createGoogleAccountService({
      env: {
        DREAM_GOOGLE_OAUTH_CLIENT_ID: "client-id",
        DREAM_GOOGLE_OAUTH_CLIENT_SECRET: "client-secret",
        DREAM_GOOGLE_OAUTH_REDIRECT_URI: "https://dream.example/api/account/google/callback",
        DREAM_ACCOUNT_ENCRYPTION_KEY: "a".repeat(64),
        DREAM_ACCOUNT_STORE_PATH: join(directory, "accounts.json"),
        DREAM_ACCOUNT_STORE_PERSISTENT: "false"
      }
    });
    const req = { headers: {}, socket: { encrypted: true } };
    const res = responseRecorder();
    const status = await service.status(req, res);
    assert.equal(status.googleOAuthConfigured, false);
    assert.equal(status.persistentStorageReady, false);
    assert.equal(status.gmail.connected, false);
    assert.match(res.getHeader("set-cookie").join(";"), /HttpOnly/);
    assert.match(res.getHeader("set-cookie").join(";"), /SameSite=Lax/);
    assert.throws(() => service.begin(req, res), /not configured by the DREAM AI operator/);
    assert.match(await service.gmailContext(req, "Search my Gmail for invoices"), /not configured/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("Gmail connector uses OAuth PKCE, encrypts tokens at rest and supports read-only search and disconnect", async () => {
  const directory = await mkdtemp(join(tmpdir(), "dream-account-roundtrip-"));
  const storePath = join(directory, "accounts.json");
  const calls = [];
  const fetchImpl = async (url, options = {}) => {
    const parsed = new URL(url);
    calls.push({ url: parsed.href, method: options.method || "GET" });
    if (parsed.hostname === "oauth2.googleapis.com" && parsed.pathname === "/token") {
      return jsonResponse({
        access_token: "ACCESS_TOKEN_SECRET",
        refresh_token: "REFRESH_TOKEN_SECRET",
        expires_in: 3600,
        scope: "openid email https://www.googleapis.com/auth/gmail.readonly"
      });
    }
    if (parsed.hostname === "openidconnect.googleapis.com") {
      return jsonResponse({ sub: "google-user-123", email: "owner@example.com", email_verified: true });
    }
    if (parsed.hostname === "gmail.googleapis.com" && parsed.pathname.endsWith("/messages")) {
      assert.equal(parsed.searchParams.get("q"), "project meeting");
      return jsonResponse({ messages: [{ id: "message-123" }] });
    }
    if (parsed.hostname === "gmail.googleapis.com" && parsed.pathname.endsWith("/messages/message-123")) {
      const encoded = Buffer.from("The project meeting is scheduled for Thursday at 14:00.").toString("base64url");
      return jsonResponse({
        snippet: "Project meeting scheduled for Thursday.",
        payload: {
          mimeType: "text/plain",
          headers: [
            { name: "From", value: "team@example.com" },
            { name: "To", value: "owner@example.com" },
            { name: "Date", value: "Mon, 9 Nov 2026 08:00:00 +0000" },
            { name: "Subject", value: "Project meeting" }
          ],
          body: { data: encoded }
        }
      });
    }
    if (parsed.hostname === "oauth2.googleapis.com" && parsed.pathname === "/revoke") return new Response("", { status: 200 });
    throw new Error("Unexpected provider request: " + parsed.href);
  };
  try {
    const service = createGoogleAccountService({
      fetchImpl,
      env: {
        DREAM_GOOGLE_OAUTH_CLIENT_ID: "test-client.apps.googleusercontent.com",
        DREAM_GOOGLE_OAUTH_CLIENT_SECRET: "operator-secret",
        DREAM_GOOGLE_OAUTH_REDIRECT_URI: "https://dream.example/api/account/google/callback",
        DREAM_ACCOUNT_ENCRYPTION_KEY: "b".repeat(64),
        DREAM_ACCOUNT_STORE_PATH: storePath,
        DREAM_ACCOUNT_STORE_PERSISTENT: "true"
      },
      clock: () => Date.parse("2026-11-09T08:00:00Z")
    });
    const req = { headers: {}, socket: { encrypted: true } };
    const res = responseRecorder();
    const before = await service.status(req, res);
    assert.equal(before.googleOAuthConfigured, true);
    assert.equal(before.gmail.connected, false);
    const session = cookieValue(res.getHeader("set-cookie"), "dream_session");
    assert.match(session, /^dream_session=[A-Za-z0-9_-]{43}$/);

    service.begin(req, res);
    assert.equal(res.statusCode, 302);
    const authorize = new URL(res.getHeader("location"));
    assert.equal(authorize.hostname, "accounts.google.com");
    assert.equal(authorize.searchParams.get("code_challenge_method"), "S256");
    assert.ok(authorize.searchParams.get("code_challenge"));
    assert.deepEqual(authorize.searchParams.get("scope").split(" "), [
      "openid", "email", "https://www.googleapis.com/auth/gmail.readonly"
    ]);
    const cookieHeaders = res.getHeader("set-cookie");
    const stateCookie = cookieValue(cookieHeaders, "dream_google_oauth_state");
    const verifierCookie = cookieValue(cookieHeaders, "dream_google_oauth_verifier");
    const state = stateCookie.split("=")[1];
    const cookies = [
      session,
      stateCookie,
      verifierCookie
    ].join("; ");
    req.headers.cookie = cookies;

    await service.callback(
      req,
      responseRecorder(),
      new URL("https://dream.example/api/account/google/callback?code=authorization-code-123&state=" + encodeURIComponent(state))
    );
    const connected = await service.status(req, responseRecorder());
    assert.equal(connected.gmail.connected, true);
    assert.equal(connected.gmail.email, "owner@example.com");

    const context = await service.gmailContext(req, "Search my Gmail for: project meeting");
    assert.match(context, /owner@example.com/);
    assert.match(context, /Project meeting/);
    assert.match(context, /Thursday at 14:00/);
    assert.match(context, /read-only/);
    assert.equal(calls.filter(call => call.method === "POST").some(call => call.url.includes("/token")), true);

    const stored = await readFile(storePath, "utf8");
    assert.doesNotMatch(stored, /ACCESS_TOKEN_SECRET|REFRESH_TOKEN_SECRET|owner@example.com/);
    assert.match(stored, /aes-256-gcm/);

    const disconnected = await service.disconnect(req, responseRecorder());
    assert.deepEqual(disconnected, { disconnected: true, providerRevoked: true });
    const after = await service.status(req, responseRecorder());
    assert.equal(after.gmail.connected, false);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("Gmail intent detection is narrow and excludes unrelated questions", () => {
  assert.equal(isGmailRequest("Search my Gmail for invoices"), true);
  assert.equal(isGmailRequest("What is in my inbox?"), true);
  assert.equal(isGmailRequest("Tell me about Gmail security"), false);
  assert.equal(isGmailRequest("Explain photosynthesis"), false);
});


test("Google OAuth rejects insecure redirects and mismatched state before exchanging a code", async () => {
  const directory = await mkdtemp(join(tmpdir(), "dream-account-state-"));
  let providerCalls = 0;
  try {
    const service = createGoogleAccountService({
      fetchImpl: async () => { providerCalls += 1; return jsonResponse({}); },
      env: {
        DREAM_GOOGLE_OAUTH_CLIENT_ID: "client-id",
        DREAM_GOOGLE_OAUTH_CLIENT_SECRET: "client-secret",
        DREAM_GOOGLE_OAUTH_REDIRECT_URI: "https://dream.example/api/account/google/callback",
        DREAM_ACCOUNT_ENCRYPTION_KEY: "c".repeat(64),
        DREAM_ACCOUNT_STORE_PATH: join(directory, "accounts.json"),
        DREAM_ACCOUNT_STORE_PERSISTENT: "true"
      }
    });
    const insecureReq = { headers: {}, socket: { encrypted: false } };
    assert.throws(() => service.begin(insecureReq, responseRecorder()), /require HTTPS/);

    const req = { headers: {}, socket: { encrypted: true } };
    const res = responseRecorder();
    await service.status(req, res);
    const session = cookieValue(res.getHeader("set-cookie"), "dream_session");
    service.begin(req, res);
    const cookies = res.getHeader("set-cookie");
    req.headers.cookie = [
      session,
      cookieValue(cookies, "dream_google_oauth_state"),
      cookieValue(cookies, "dream_google_oauth_verifier")
    ].join("; ");
    await assert.rejects(
      service.callback(
        req,
        responseRecorder(),
        new URL("https://dream.example/api/account/google/callback?code=authorization-code-123&state=wrong-state")
      ),
      /state could not be verified/
    );
    assert.equal(providerCalls, 0);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
