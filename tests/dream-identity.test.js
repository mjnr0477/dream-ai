import test from "node:test";
import assert from "node:assert/strict";
import { DREAM_IDENTITY, buildDreamIdentityContext } from "../lib/identity.js";
import { buildSystemPrompt } from "../lib/dream-engine.js";

test("DREAM AI has a stable, accurate creator identity", () => {
  assert.equal(DREAM_IDENTITY.name, "DREAM AI");
  assert.equal(DREAM_IDENTITY.creator, "Mathias Simiyu");
  assert.match(DREAM_IDENTITY.identityAnswer, /created by Mathias Simiyu/);
});

test("creator identity is included in every system prompt", () => {
  for (const mode of ["general", "coding", "learning", "research"]) {
    const prompt = buildSystemPrompt(mode);
    assert.match(prompt, /Creator: Mathias Simiyu/);
    assert.match(prompt, /I am DREAM AI, an independent AI project created by Mathias Simiyu/);
    assert.match(prompt, /Do not confuse DREAM AI with the separate projects/);
    assert.match(prompt, /fresh retrieved evidence/);
    assert.match(prompt, /private dossiers/);
  }
});

test("identity context uses readable separate lines", () => {
  const context = buildDreamIdentityContext();
  assert.match(context, /VERIFIED PRODUCT IDENTITY:\nProduct: DREAM AI\nCreator: Mathias Simiyu/);
});
