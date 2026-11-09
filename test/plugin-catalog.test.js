import test from "node:test";
import assert from "node:assert/strict";
import { getDreamPluginCatalog } from "../lib/plugin-catalog.js";

test("plugin catalog is guest-first and never asks users to supply API keys", () => {
  const catalog = getDreamPluginCatalog();
  assert.equal(catalog.mode, "discovery_catalog");
  assert.equal(catalog.guestAccessSupported, true);
  assert.equal(catalog.userRegistrationRequired, false);
  assert.equal(catalog.userSuppliedApiKeysRequired, false);
  assert.equal(catalog.policy.anonymousPublicReadOnlyByDefault, true);
  assert.equal(catalog.policy.privateAccountAccessRequiresAuthorization, true);
  assert.equal(catalog.policy.writeAndFinancialActionsRequireExplicitConfirmation, true);
  assert.ok(catalog.categories.length >= 8);
  assert.ok(catalog.plugins.length >= 30);
});

test("catalog distinguishes implemented public connectors from gated integrations", () => {
  const catalog = getDreamPluginCatalog();
  const byId = new Map(catalog.plugins.map(plugin => [plugin.id, plugin]));

  assert.equal(byId.get("bluesky").status, "connected_read_only");
  assert.equal(byId.get("wikipedia").apiKeyFree, true);
  assert.equal(byId.get("public_github").status, "catalogued");
  assert.equal(byId.get("facebook_instagram_threads").status, "requires_provider_access");
  assert.equal(byId.get("gmail").status, "requires_user_authorization");
  assert.equal(byId.get("mpesa").status, "requires_provider_access");
  assert.equal(byId.get("facebook_instagram_threads").apiKeyFree, false);
  assert.equal(byId.get("facebook_instagram_threads").access, "approved_meta_api_and_user_authorization");
});

test("catalog responses are fresh copies and cannot mutate the server catalogue", () => {
  const first = getDreamPluginCatalog();
  first.plugins[0].name = "tampered";
  first.plugins[0].capabilities.push("unsafe_action");
  first.categories[0].name = "tampered";

  const second = getDreamPluginCatalog();
  assert.equal(second.plugins[0].name, "Wikipedia");
  assert.equal(second.plugins[0].capabilities.includes("unsafe_action"), false);
  assert.equal(second.categories[0].name, "Research and knowledge");
});

test("every plugin has a valid category, status and non-empty safety note", () => {
  const catalog = getDreamPluginCatalog();
  const categoryIds = new Set(catalog.categories.map(category => category.id));
  const statuses = new Set([
    "connected_read_only", "catalogued", "requires_provider_access",
    "requires_user_authorization", "planned"
  ]);

  for (const plugin of catalog.plugins) {
    assert.ok(categoryIds.has(plugin.category), `unknown category for ${plugin.id}`);
    assert.ok(statuses.has(plugin.status), `unknown status for ${plugin.id}`);
    assert.ok(Array.isArray(plugin.capabilities) && plugin.capabilities.length > 0);
    assert.equal(typeof plugin.note, "string");
    assert.ok(plugin.note.length > 0);
  }
});
