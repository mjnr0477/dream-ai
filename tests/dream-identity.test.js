import test from "node:test";
import assert from "node:assert/strict";
import { DREAM_IDENTITY, buildDreamIdentityContext, isDirectCreatorQuestion } from "../lib/identity.js";
import { buildSystemPrompt } from "../lib/dream-engine.js";

test("DREAM AI retains a stable, accurate creator record", () => {
  assert.equal(DREAM_IDENTITY.name, "DREAM AI");
  assert.equal(DREAM_IDENTITY.creator, "Mathias Simiyu");
  assert.match(DREAM_IDENTITY.identityAnswer, /created by Mathias Simiyu/);
});

test("creator disclosure is not included in ordinary product conversations", () => {
  for (const mode of ["general", "coding", "learning", "research"]) {
    const prompt = buildSystemPrompt(mode, "", "Tell me about DREAM AI and what it can do.");
    assert.doesNotMatch(prompt, /Mathias Simiyu/);
    assert.doesNotMatch(prompt, /Origin answer:/);
    assert.match(prompt, /Do not mention, identify or volunteer the creator/);
    assert.match(prompt, /Do not confuse DREAM AI with the separate projects/);
    assert.match(prompt, /fresh retrieved evidence/);
    assert.match(prompt, /private dossiers/);
  }
});

test("creator identity is included only for a direct creator question", () => {
  for (const query of [
    "Who created you?",
    "Who is the creator of DREAM AI?",
    "Who founded DREAM AI?",
    "Who is your developer?"
  ]) {
    assert.equal(isDirectCreatorQuestion(query), true, query);
    const prompt = buildSystemPrompt("general", "", query);
    assert.match(prompt, /Creator: Mathias Simiyu/);
    assert.match(prompt, /I am DREAM AI, an independent AI project created by Mathias Simiyu/);
  }
});

test("unrelated questions about a person do not trigger creator disclosure", () => {
  assert.equal(isDirectCreatorQuestion("Who is Mathias Simiyu?"), false);
  assert.equal(isDirectCreatorQuestion("What does DREAM AI do?"), false);
  assert.equal(isDirectCreatorQuestion("Who owns the company?"), false);
  const context = buildDreamIdentityContext("What does DREAM AI do?");
  assert.doesNotMatch(context, /Mathias Simiyu|Origin answer:/);
});

test("identity context keeps product identity separate from creator disclosure", () => {
  const context = buildDreamIdentityContext();
  assert.match(context, /VERIFIED PRODUCT IDENTITY:\nProduct: DREAM AI/);
  assert.match(context, /Creator disclosure rule:/);
  assert.doesNotMatch(context, /Creator: Mathias Simiyu/);
});
