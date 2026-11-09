import test from "node:test";
import assert from "node:assert/strict";
import { scoreForecasts } from "./forecast-evaluation.js";

function forecast(overrides = {}) {
  return {
    id: "forecast-1",
    domain: "sports",
    model: "DREAM AI",
    modelVersion: "forecast-engine-v1",
    horizon: "24h",
    issuedAt: "2026-11-08T12:00:00Z",
    resolvedAt: "2026-11-09T12:00:00Z",
    outcomes: ["home", "draw", "away"],
    probabilities: [0.5, 0.25, 0.25],
    actualOutcome: "home",
    ...overrides
  };
}

test("perfect forecasts receive zero Brier and log loss with full top-outcome credit", () => {
  const result = scoreForecasts([
    forecast({ id: "one", probabilities: [1, 0, 0], actualOutcome: "home" }),
    forecast({ id: "two", probabilities: [0, 1, 0], actualOutcome: "draw" })
  ]);
  assert.equal(result.schema, "dream-forecast-evaluation-v1");
  assert.equal(result.overall.samples, 2);
  assert.equal(result.overall.brierScore, 0);
  assert.equal(result.overall.logLoss, 0);
  assert.equal(result.overall.topOutcomeAccuracy, 1);
  assert.equal(result.overall.expectedCalibrationError, 0);
  assert.equal(result.overall.brierSkillScore, null);
});

test("multiclass Brier score and log loss are computed against the resolved outcome", () => {
  const result = scoreForecasts([
    forecast({ probabilities: [0.5, 0.25, 0.25], actualOutcome: "home" })
  ]);
  assert.equal(result.overall.brierScore, 0.375);
  assert.equal(result.overall.logLoss, Number((-Math.log(0.5)).toFixed(8)));
  assert.equal(result.overall.topOutcomeAccuracy, 1);
});

test("ties receive fractional top-outcome credit instead of depending on label order", () => {
  const result = scoreForecasts([
    forecast({ probabilities: [0.5, 0.5, 0], actualOutcome: "draw" })
  ]);
  assert.equal(result.overall.topOutcomeAccuracy, 0.5);
  assert.equal(result.overall.expectedCalibrationError, 0);
});

test("confident wrong predictions are penalized more than uncertain predictions", () => {
  const wrong = scoreForecasts([
    forecast({ probabilities: [0.95, 0.025, 0.025], actualOutcome: "away" })
  ]);
  const uncertain = scoreForecasts([
    forecast({ probabilities: [1 / 3, 1 / 3, 1 / 3], actualOutcome: "away" })
  ]);
  assert.ok(wrong.overall.brierScore > uncertain.overall.brierScore);
  assert.ok(wrong.overall.logLoss > uncertain.overall.logLoss);
});

test("calibration bins, domain and horizon summaries are reported independently", () => {
  const result = scoreForecasts([
    forecast({ id: "weather", domain: "weather", horizon: "48h", outcomes: ["rain", "dry"], probabilities: [0.8, 0.2], actualOutcome: "rain" }),
    forecast({ id: "market", domain: "markets", horizon: "7d", outcomes: ["up", "flat", "down"], probabilities: [0.2, 0.3, 0.5], actualOutcome: "down" })
  ], { bins: 5 });
  assert.deepEqual(Object.keys(result.byDomain), ["markets", "weather"]);
  assert.deepEqual(Object.keys(result.byHorizon), ["48h", "7d"]);
  assert.equal(result.byDomain.weather.samples, 1);
  assert.equal(result.byHorizon["7d"].samples, 1);
  assert.equal(result.overall.calibrationBins.length, 5);
  assert.equal(result.provenance.models[0], "DREAM AI");
});

test("Brier skill score is computed only against a complete supplied baseline", () => {
  const result = scoreForecasts([
    forecast({
      probabilities: [0.8, 0.1, 0.1],
      actualOutcome: "home",
      baselineProbabilities: [0.34, 0.33, 0.33]
    })
  ]);
  assert.ok(result.overall.brierSkillScore > 0);
  assert.ok(result.overall.baselineBrierScore > result.overall.brierScore);
});

test("forecast rows reject malformed identities, timestamps and outcomes", () => {
  assert.throws(() => scoreForecasts([]), /at least one resolved forecast/);
  assert.throws(() => scoreForecasts([forecast(), forecast()]), /Duplicate forecast id/);
  assert.throws(() => scoreForecasts([forecast({ issuedAt: "yesterday" })]), /ISO 8601 timestamp/);
  assert.throws(() => scoreForecasts([forecast({ resolvedAt: "2026-11-08T11:00:00Z" })]), /resolved before it was issued/);
  assert.throws(() => scoreForecasts([forecast({ outcomes: ["home", "home", "away"] })]), /duplicate outcome labels/);
  assert.throws(() => scoreForecasts([forecast({ actualOutcome: "unknown" })]), /not in outcomes/);
  assert.throws(() => scoreForecasts([forecast({ modelVersion: "" })]), /modelVersion/);
});

test("probabilities must be finite, bounded, aligned and sum to one", () => {
  assert.throws(() => scoreForecasts([forecast({ probabilities: [0.5, 0.5] })]), /one probability per outcome/);
  assert.throws(() => scoreForecasts([forecast({ probabilities: [0.5, 0.25, 0.1] })]), /must sum to 1/);
  assert.throws(() => scoreForecasts([forecast({ probabilities: [1.2, -0.1, -0.1] })]), /outside \[0, 1\]/);
  assert.throws(() => scoreForecasts([forecast({ probabilities: [NaN, 0.5, 0.5] })]), /outside \[0, 1\]/);
  assert.throws(() => scoreForecasts([forecast({ probabilities: [0.5, 0.5, 0] , baselineProbabilities:[0.3,0.3,0.4]})]), /must sum to 1/);
});

test("baseline coverage must be complete and bins must be bounded", () => {
  assert.throws(() => scoreForecasts([
    forecast({ id: "one", baselineProbabilities: [0.34, 0.33, 0.33] }),
    forecast({ id: "two" })
  ]), /for every forecast or none/);
  assert.throws(() => scoreForecasts([forecast()], { bins: 1 }), /from 2 to 100/);
  assert.throws(() => scoreForecasts([forecast()], { bins: 101 }), /from 2 to 100/);
});

test("invalid optional provenance and horizon fields are rejected", () => {
  assert.throws(() => scoreForecasts([forecast({ checkpointSha256: "not-a-hash" })]), /64-character SHA-256/);
  assert.throws(() => scoreForecasts([forecast({ horizon: " " })]), /horizon/);
});
