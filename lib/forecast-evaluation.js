const MIN_EPSILON = 1e-15;
const ISO_TIMESTAMP = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?(?:Z|[+-]\d{2}:\d{2})$/;
const SHA256 = /^[a-f0-9]{64}$/i;

function isNonEmptyString(value) {
  return typeof value === "string" && value.trim().length > 0;
}

function parseTimestamp(value, label, id) {
  if (typeof value !== "string" || !ISO_TIMESTAMP.test(value)) {
    throw new Error("Forecast " + id + " needs " + label + " as an ISO 8601 timestamp with a timezone.");
  }
  const parsed = Date.parse(value);
  if (!Number.isFinite(parsed)) throw new Error("Forecast " + id + " has an invalid " + label + ".");
  return parsed;
}

function validateProbabilityVector(probabilities, outcomes, label, id) {
  if (!Array.isArray(probabilities) || probabilities.length !== outcomes.length) {
    throw new Error("Forecast " + id + " needs " + label + " with one probability per outcome.");
  }
  if (probabilities.some(value => typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1)) {
    throw new Error("Forecast " + id + " has a " + label + " value outside [0, 1].");
  }
  const total = probabilities.reduce((sum, value) => sum + value, 0);
  if (Math.abs(total - 1) > 1e-6) {
    throw new Error("Forecast " + id + " " + label + " must sum to 1 within 1e-6; received " + total + ".");
  }
  return probabilities;
}

function validateRecord(record, index) {
  const id = isNonEmptyString(record?.id) ? record.id.trim() : "row-" + (index + 1);
  if (!record || typeof record !== "object" || Array.isArray(record)) {
    throw new Error("Forecast row " + (index + 1) + " must be an object.");
  }
  for (const field of ["id", "domain", "model", "modelVersion", "actualOutcome"]) {
    if (!isNonEmptyString(record[field])) {
      throw new Error("Forecast " + id + " needs a non-empty " + field + ".");
    }
  }
  if (!Array.isArray(record.outcomes) || record.outcomes.length < 2 || record.outcomes.length > 20 ||
      record.outcomes.some(value => !isNonEmptyString(value))) {
    throw new Error("Forecast " + id + " needs 2 to 20 non-empty outcome labels.");
  }
  const outcomes = record.outcomes.map(value => value.trim());
  if (new Set(outcomes).size !== outcomes.length) {
    throw new Error("Forecast " + id + " has duplicate outcome labels.");
  }
  const actualIndex = outcomes.indexOf(record.actualOutcome.trim());
  if (actualIndex < 0) throw new Error("Forecast " + id + " actualOutcome is not in outcomes.");
  const probabilities = validateProbabilityVector(record.probabilities, outcomes, "probabilities", id);
  const issuedAt = parseTimestamp(record.issuedAt, "issuedAt", id);
  const resolvedAt = parseTimestamp(record.resolvedAt, "resolvedAt", id);
  if (resolvedAt < issuedAt) throw new Error("Forecast " + id + " resolved before it was issued.");
  if (record.checkpointSha256 !== undefined &&
      (typeof record.checkpointSha256 !== "string" || !SHA256.test(record.checkpointSha256))) {
    throw new Error("Forecast " + id + " checkpointSha256 must be a 64-character SHA-256 digest when supplied.");
  }
  if (record.horizon !== undefined && !isNonEmptyString(record.horizon)) {
    throw new Error("Forecast " + id + " horizon must be a non-empty string when supplied.");
  }
  const baselineProbabilities = record.baselineProbabilities === undefined
    ? null
    : validateProbabilityVector(record.baselineProbabilities, outcomes, "baselineProbabilities", id);
  const bestProbability = Math.max(...probabilities);
  const bestIndices = probabilities.flatMap((value, outcomeIndex) =>
    Math.abs(value - bestProbability) <= 1e-12 ? [outcomeIndex] : []
  );
  const tieAwareCorrectness = bestIndices.includes(actualIndex) ? 1 / bestIndices.length : 0;
  const actualProb = probabilities[actualIndex];
  const brier = probabilities.reduce((sum, probability, outcomeIndex) =>
    sum + (probability - (outcomeIndex === actualIndex ? 1 : 0)) ** 2, 0
  );
  const baselineBrier = baselineProbabilities
    ? baselineProbabilities.reduce((sum, probability, outcomeIndex) =>
      sum + (probability - (outcomeIndex === actualIndex ? 1 : 0)) ** 2, 0
    )
    : null;
  return {
    id,
    domain: record.domain.trim(),
    model: record.model.trim(),
    modelVersion: record.modelVersion.trim(),
    horizon: record.horizon?.trim() || null,
    issuedAt,
    resolvedAt,
    confidence: bestProbability,
    correctness: tieAwareCorrectness,
    brier,
    logLoss: -Math.log(Math.max(actualProb, MIN_EPSILON)),
    baselineBrier
  };
}

function summarize(records, binCount) {
  const count = records.length;
  const brierScore = records.reduce((sum, row) => sum + row.brier, 0) / count;
  const logLoss = records.reduce((sum, row) => sum + row.logLoss, 0) / count;
  const accuracy = records.reduce((sum, row) => sum + row.correctness, 0) / count;
  const bins = Array.from({ length: binCount }, (_, index) => ({
    lowerBound: index / binCount,
    upperBound: (index + 1) / binCount,
    count: 0,
    meanConfidence: null,
    observedAccuracy: null
  }));
  const totals = bins.map(() => ({ confidence: 0, correctness: 0 }));
  for (const row of records) {
    const index = Math.min(binCount - 1, Math.floor(row.confidence * binCount));
    bins[index].count += 1;
    totals[index].confidence += row.confidence;
    totals[index].correctness += row.correctness;
  }
  let expectedCalibrationError = 0;
  for (let index = 0; index < bins.length; index += 1) {
    const bin = bins[index];
    if (!bin.count) continue;
    bin.meanConfidence = totals[index].confidence / bin.count;
    bin.observedAccuracy = totals[index].correctness / bin.count;
    expectedCalibrationError += (bin.count / count) *
      Math.abs(bin.meanConfidence - bin.observedAccuracy);
  }
  const baselineRows = records.filter(row => row.baselineBrier !== null);
  const baselineComplete = baselineRows.length === count;
  const baselineBrierScore = baselineComplete
    ? baselineRows.reduce((sum, row) => sum + row.baselineBrier, 0) / count
    : null;
  const brierSkillScore = baselineComplete && baselineBrierScore > 0
    ? 1 - brierScore / baselineBrierScore
    : null;
  return {
    samples: count,
    brierScore: Number(brierScore.toFixed(8)),
    logLoss: Number(logLoss.toFixed(8)),
    topOutcomeAccuracy: Number(accuracy.toFixed(8)),
    expectedCalibrationError: Number(expectedCalibrationError.toFixed(8)),
    calibrationBins: bins.map(bin => ({
      ...bin,
      meanConfidence: bin.meanConfidence === null ? null : Number(bin.meanConfidence.toFixed(8)),
      observedAccuracy: bin.observedAccuracy === null ? null : Number(bin.observedAccuracy.toFixed(8))
    })),
    baselineBrierScore: baselineBrierScore === null ? null : Number(baselineBrierScore.toFixed(8)),
    brierSkillScore: brierSkillScore === null ? null : Number(brierSkillScore.toFixed(8))
  };
}

function groupBy(records, key) {
  const groups = new Map();
  for (const record of records) {
    const value = record[key];
    if (!value) continue;
    if (!groups.has(value)) groups.set(value, []);
    groups.get(value).push(record);
  }
  return Object.fromEntries([...groups.entries()].sort(([a], [b]) => a.localeCompare(b)));
}

/**
 * Scores resolved probabilistic forecasts. Lower Brier/log loss/ECE are better.
 * Brier is the mean sum of squared class-probability errors per event (0 to 2).
 * Tied top probabilities receive fractional top-outcome credit; Brier and log
 * loss remain the primary strictly proper scoring rules.
 */
export function scoreForecasts(records, { bins = 10 } = {}) {
  if (!Array.isArray(records) || records.length === 0) {
    throw new Error("Forecast scoring requires at least one resolved forecast record.");
  }
  if (!Number.isSafeInteger(bins) || bins < 2 || bins > 100) {
    throw new Error("Calibration bins must be an integer from 2 to 100.");
  }
  const ids = new Set();
  const normalized = records.map(validateRecord);
  for (const row of normalized) {
    if (ids.has(row.id)) throw new Error("Duplicate forecast id: " + row.id);
    ids.add(row.id);
  }
  const baselineCount = normalized.filter(row => row.baselineBrier !== null).length;
  if (baselineCount > 0 && baselineCount !== normalized.length) {
    throw new Error("Baseline probabilities must be supplied for every forecast or none.");
  }
  return {
    schema: "dream-forecast-evaluation-v1",
    scoring: {
      brier: "mean sum of squared class-probability errors per event; lower is better (range 0 to 2)",
      logLoss: "mean negative natural log of the probability assigned to the actual outcome; lower is better",
      topOutcomeAccuracy: "mean tie-aware correctness of the highest-probability outcome",
      expectedCalibrationError: "bin-weighted absolute difference between mean confidence and observed top-outcome accuracy; lower is better",
      brierSkillScore: "1 - model Brier score / baseline Brier score; positive beats the supplied baseline",
      probabilityTolerance: 1e-6,
      logLossEpsilon: MIN_EPSILON
    },
    overall: summarize(normalized, bins),
    byDomain: Object.fromEntries(Object.entries(groupBy(normalized, "domain"))
      .map(([name, rows]) => [name, summarize(rows, bins)])),
    byHorizon: Object.fromEntries(Object.entries(groupBy(normalized, "horizon"))
      .map(([name, rows]) => [name, summarize(rows, bins)])),
    provenance: {
      models: [...new Set(normalized.map(row => row.model))].sort(),
      modelVersions: [...new Set(normalized.map(row => row.modelVersion))].sort(),
      checkpointSha256Values: [...new Set(records.map(row => row.checkpointSha256).filter(Boolean))].sort(),
      issuedFrom: new Date(Math.min(...normalized.map(row => row.issuedAt))).toISOString(),
      resolvedThrough: new Date(Math.max(...normalized.map(row => row.resolvedAt))).toISOString()
    }
  };
}
