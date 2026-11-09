import test from "node:test";
import assert from "node:assert/strict";
import { isAppBuildRequest, classifyAppTarget, buildAppBuilderGuidance } from "./app-builder.js";

test("recognizes explicit app and full-project creation requests", () => {
  for (const query of [
    "Build a complete web app for booking appointments",
    "Create an Android mobile application",
    "Develop a desktop app for managing stock",
    "Generate a REST API for a school platform",
    "Make a browser extension for saving research",
    "Build a 2D game in Godot",
    "Create a Discord bot"
  ]) assert.equal(isAppBuildRequest(query), true, query);
});

test("does not treat app-development explanations as project-generation requests", () => {
  for (const query of [
    "What is application development?",
    "Explain how mobile apps are tested",
    "Which programming language is used for websites?",
    "Tell me about the history of video games"
  ]) assert.equal(isAppBuildRequest(query), false, query);
});

test("classifies common application targets", () => {
  const cases = [
    ["Build a website using Next.js", "web"],
    ["Create a Flutter Android and iOS app", "mobile"],
    ["Develop a macOS desktop application", "desktop"],
    ["Build a REST API and backend service", "backend-api"],
    ["Create a command-line tool in Python", "command-line"],
    ["Make a 2D game", "game"],
    ["Build a Chrome browser extension", "browser-extension"],
    ["Create a Telegram bot", "bot"],
    ["Build firmware for an ESP32 IoT sensor", "embedded-iot"],
    ["Develop an application", "general"]
  ];
  for (const [query, expected] of cases) assert.equal(classifyAppTarget(query), expected, query);
});

test("app guidance requires complete project files and honest verification", () => {
  const guidance = buildAppBuilderGuidance("Build a mobile app");
  for (const phrase of [
    "APPLICATION BUILD MODE",
    "complete, coherent project",
    "README",
    "package/build manifest",
    ".env.example",
    "tests",
    "--- DREAM FILE: relative/path ---",
    "downloadable project ZIP",
    "Do not claim that a build, test, emulator, simulator or deployment was run",
    "isolated execution sandbox"
  ]) assert.ok(guidance.includes(phrase), "Missing instruction: " + phrase);
});

test("guidance is absent for ordinary factual and creative requests", () => {
  assert.equal(buildAppBuilderGuidance("Explain what an API is"), "");
  assert.equal(buildAppBuilderGuidance("Write a poem about computers"), "");
});

test("target guidance names the relevant platform toolchains", () => {
  assert.match(buildAppBuilderGuidance("Create an iOS and Android app"), /Expo|React Native|Flutter/);
  assert.match(buildAppBuilderGuidance("Build a Chrome extension"), /Manifest V3/);
  assert.match(buildAppBuilderGuidance("Develop firmware for ESP32"), /PlatformIO|Arduino/);
});
