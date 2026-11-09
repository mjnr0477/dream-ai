import test from "node:test";
import assert from "node:assert/strict";
import {
  createMetaAccountService,
  isMetaRequest
} from "../lib/meta-account.js";

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

const baseEnv = () => ({
  META_APP_ID: "123456789012345",
  META_APP_SECRET: "test_meta_app_secret_value_1234567890",
  META_FACEBOOK_REDIRECT_URI: "https://dream.example/api/account/meta/facebook/callback",
  META_THREADS_REDIRECT_URI: "https://dream.example/api/account/meta/threads/callback",
  DREAM_PUBLIC_ORIGIN: "https://dream.example"
});

test("Meta sign-in stays disabled without app credentials and never returns the operator token", async () => {
  const env = { META_ACCESS_TOKEN: "operator-secret-token-that-must-not-leak" };
  const service = createMetaAccountService({ env });
  const status = await service.status({ headers: {}, socket: { encrypted: true } }, responseRecorder());
  assert.equal(status.metaOAuthConfigured, false);
  assert.equal(status.operatorTokenConfigured, true);
  assert.equal(status.operatorTokenValid, false);
  assert.equal(JSON.stringify(status).includes(env.META_ACCESS_TOKEN), false);
  assert.throws(
    () => service.begin({ headers: {}, socket: { encrypted: true } }, responseRecorder(), "facebook"),
    /META_APP_ID and META_APP_SECRET/
  );
});

test("Facebook OAuth validates state, exchanges tokens, encrypts the session and lists linked Instagram accounts", async () => {
  const calls = [];
  const fetchImpl = async (url, options = {}) => {
    const parsed = new URL(url);
    calls.push({ url: parsed.href, method: options.method || "GET", body: options.body });
    if (parsed.hostname === "graph.facebook.com" && parsed.pathname.endsWith("/oauth/access_token")) {
      return parsed.searchParams.get("grant_type") === "fb_exchange_token"
        ? jsonResponse({ access_token: "LONG_LIVED_FACEBOOK_TOKEN", expires_in: 5184000 })
        : jsonResponse({ access_token: "SHORT_LIVED_FACEBOOK_TOKEN", expires_in: 3600 });
    }
    if (parsed.hostname === "graph.facebook.com" && parsed.pathname.endsWith("/me/permissions")) {
      return jsonResponse({ data: [
        { permission: "pages_show_list", status: "granted" },
        { permission: "pages_read_engagement", status: "granted" },
        { permission: "instagram_basic", status: "granted" }
      ] });
    }
    if (parsed.hostname === "graph.facebook.com" && parsed.pathname.endsWith("/me/accounts")) {
      return jsonResponse({ data: [{
        id: "page-42",
        name: "Dream Learning Page",
        instagram_business_account: {
          id: "ig-77",
          username: "dream.learning",
          account_type: "BUSINESS",
          media_count: 8
        }
      }] });
    }
    if (parsed.hostname === "graph.facebook.com" && parsed.pathname.endsWith("/me")) {
      return jsonResponse({ id: "fb-user-1", name: "Test Meta User" });
    }
    throw new Error("Unexpected test request: " + parsed.href);
  };
  const service = createMetaAccountService({ env: baseEnv(), fetchImpl });
  const req = { headers: { host: "dream.example", "x-forwarded-proto": "https" }, socket: {} };
  const startResponse = responseRecorder();
  service.begin(req, startResponse, "facebook");
  assert.equal(startResponse.statusCode, 302);
  const authorize = new URL(startResponse.getHeader("location"));
  assert.equal(authorize.hostname, "www.facebook.com");
  assert.equal(authorize.searchParams.get("client_id"), baseEnv().META_APP_ID);
  assert.ok(authorize.searchParams.get("scope").includes("pages_show_list"));
  assert.ok(authorize.searchParams.get("scope").includes("instagram_basic"));
  const stateCookie = cookieValue(startResponse.getHeader("set-cookie"), "dream_meta_facebook_state");
  const state = stateCookie.split("=")[1];
  req.headers.cookie = stateCookie;

  const callbackResponse = responseRecorder();
  await service.callback(
    req,
    callbackResponse,
    new URL("https://dream.example/api/account/meta/facebook/callback?code=meta-auth-code-123&state=" + encodeURIComponent(state)),
    "facebook"
  );
  assert.equal(callbackResponse.statusCode, 303);
  assert.equal(callbackResponse.getHeader("location"), "/?meta=connected");
  const sessionCookie = cookieValue(callbackResponse.getHeader("set-cookie"), "dream_meta_facebook");
  assert.ok(sessionCookie.startsWith("dream_meta_facebook="));
  assert.match(sessionCookie, /HttpOnly/);
  assert.match(sessionCookie, /Secure/);
  assert.doesNotMatch(sessionCookie, /LONG_LIVED_FACEBOOK_TOKEN|SHORT_LIVED_FACEBOOK_TOKEN/);

  req.headers.cookie = sessionCookie;
  const status = await service.status(req, responseRecorder());
  assert.equal(status.facebook.connected, true);
  assert.equal(status.facebook.label, "Test Meta User");
  assert.deepEqual(status.facebook.scopes, ["pages_show_list", "pages_read_engagement", "instagram_basic"]);

  const context = await service.socialContext(req, "Check my Instagram account");
  assert.match(context, /PRIVATE META ACCOUNT CONTEXT/);
  assert.match(context, /@dream\.learning/);
  assert.match(context, /8 media items/);
  assert.equal(JSON.stringify(context).includes("LONG_LIVED_FACEBOOK_TOKEN"), false);
  assert.equal(calls.some(call => call.url.includes("LONG_LIVED_FACEBOOK_TOKEN")), false);
});

test("Threads uses its separate OAuth flow and separate encrypted session cookie", async () => {
  const calls = [];
  const fetchImpl = async (url, options = {}) => {
    const parsed = new URL(url);
    calls.push({ url: parsed.href, method: options.method || "GET", body: options.body });
    if (parsed.hostname === "graph.threads.net" && parsed.pathname === "/oauth/access_token") {
      const form = new URLSearchParams(options.body);
      assert.equal(form.get("grant_type"), "authorization_code");
      return jsonResponse({ access_token: "SHORT_THREADS_SECRET", user_id: "threads-12", expires_in: 3600 });
    }
    if (parsed.hostname === "graph.threads.net" && parsed.pathname === "/access_token") {
      assert.equal(parsed.searchParams.get("grant_type"), "th_exchange_token");
      assert.equal(options.headers?.authorization, "Bearer SHORT_THREADS_SECRET");
      assert.equal(parsed.searchParams.has("access_token"), false);
      return jsonResponse({ access_token: "LONG_THREADS_SECRET", expires_in: 5184000 });
    }
    if (parsed.hostname === "graph.threads.net" && parsed.pathname === "/v1.0/me") {
      return jsonResponse({ id: "threads-12", username: "dream_user" });
    }
    throw new Error("Unexpected test request: " + parsed.href);
  };
  const service = createMetaAccountService({ env: baseEnv(), fetchImpl });
  const req = { headers: { host: "dream.example", "x-forwarded-proto": "https" }, socket: {} };
  const startResponse = responseRecorder();
  service.begin(req, startResponse, "threads");
  const authorize = new URL(startResponse.getHeader("location"));
  assert.equal(authorize.hostname, "threads.net");
  assert.equal(authorize.searchParams.get("scope"), "threads_basic");
  const stateCookie = cookieValue(startResponse.getHeader("set-cookie"), "dream_meta_threads_state");
  const state = stateCookie.split("=")[1];
  req.headers.cookie = stateCookie;

  const callbackResponse = responseRecorder();
  await service.callback(
    req,
    callbackResponse,
    new URL("https://dream.example/api/account/meta/threads/callback?code=threads-auth-code-123&state=" + encodeURIComponent(state)),
    "threads"
  );
  const sessionCookie = cookieValue(callbackResponse.getHeader("set-cookie"), "dream_meta_threads");
  assert.ok(sessionCookie.startsWith("dream_meta_threads="));
  assert.doesNotMatch(sessionCookie, /SHORT_THREADS_SECRET|LONG_THREADS_SECRET/);
  req.headers.cookie = sessionCookie;
  const status = await service.status(req, responseRecorder());
  assert.equal(status.threads.connected, true);
  assert.equal(status.threads.label, "dream_user");
  const context = await service.socialContext(req, "Check my Threads account");
  assert.match(context, /@dream_user/);
  assert.equal(JSON.stringify(context).includes("LONG_THREADS_SECRET"), false);
  assert.equal(calls.some(call => call.url.includes("LONG_THREADS_SECRET")), false);
});

test("Meta OAuth rejects insecure callbacks and mismatched state without calling Meta", async () => {
  let calls = 0;
  const service = createMetaAccountService({
    env: baseEnv(),
    fetchImpl: async () => { calls += 1; return jsonResponse({}); }
  });
  const insecureRequest = { headers: { host: "dream.example" }, socket: {} };
  assert.throws(
    () => service.begin(insecureRequest, responseRecorder(), "facebook"),
    /require HTTPS/
  );
  const request = { headers: { host: "dream.example", "x-forwarded-proto": "https" }, socket: {} };
  await assert.rejects(
    service.callback(
      request,
      responseRecorder(),
      new URL("https://dream.example/api/account/meta/facebook/callback?code=meta-auth-code-123&state=wrong-state"),
      "facebook"
    ),
    /state could not be verified/
  );
  assert.equal(calls, 0);
});

test("Meta intent detection is limited to explicit private-account requests", () => {
  assert.equal(isMetaRequest("Check my Facebook Pages"), true);
  assert.equal(isMetaRequest("Show my Instagram account"), true);
  assert.equal(isMetaRequest("Review my Threads profile"), true);
  assert.equal(isMetaRequest("Explain how Facebook Login works"), false);
  assert.equal(isMetaRequest("What is Instagram?"), false);
});
