import assert from "node:assert/strict";
import test from "node:test";
import { discoverPublicDatasets } from "../scripts/discover-public-datasets.mjs";

function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json" }
  });
}

test("discovers only public Hugging Face dataset metadata without credentials", async () => {
  let requestedUrl;
  let requestOptions;
  const report = await discoverPublicDatasets("Kiswahili education", {
    now: new Date("2026-11-09T06:00:00.000Z"),
    fetchImpl: async (url, options) => {
      requestedUrl = new URL(url);
      requestOptions = options;
      return jsonResponse([
        {
          id: "open-data/kenya-learning",
          description: "A reviewed candidate dataset.",
          cardData: { license: "cc-by-4.0" },
          tags: ["language:sw", "task_categories:text-generation"],
          downloads: 120,
          likes: 9,
          lastModified: "2026-11-01T00:00:00.000Z",
          private: false,
          gated: false
        },
        {
          id: "community/unknown-license",
          tags: [],
          downloads: 10,
          private: false,
          gated: false
        },
        {
          id: "restricted/gated",
          private: false,
          gated: "manual"
        },
        {
          id: "owner/private",
          private: true,
          gated: false
        },
        {
          id: "owner/string-private",
          private: "true",
          gated: false
        },
        {
          id: "restricted/string-gated",
          private: false,
          gated: "true"
        },
        {
          id: "../path",
          private: false,
          gated: false
        }
      ]);
    }
  });

  assert.equal(requestedUrl.origin, "https://huggingface.co");
  assert.equal(requestedUrl.pathname, "/api/datasets");
  assert.equal(requestedUrl.searchParams.get("search"), "Kiswahili education");
  assert.equal(requestedUrl.searchParams.get("full"), "true");
  assert.equal(requestOptions.headers.authorization, undefined);
  assert.equal(requestOptions.redirect, "error");
  assert.equal(report.candidateCount, 2);
  assert.equal(report.candidates[0].declaredLicense, "cc-by-4.0");
  assert.equal(report.candidates[0].licenseEvidence, "dataset-card-metadata");
  assert.equal(report.candidates[1].declaredLicense, null);
  assert.equal(report.candidates[0].trainingApproved, false);
  assert.deepEqual(report.candidates[0].review, {
    rights: "pending", privacy: "pending", quality: "pending"
  });
  assert.equal(report.trainingStarted, false);
  assert.equal(report.datasetFilesDownloaded, false);
  assert.equal("files" in report.candidates[0], false);
});

test("uses a Hub license tag only as a declaration, never approval", async () => {
  const report = await discoverPublicDatasets("science", {
    fetchImpl: async () => jsonResponse([{
      id: "public/science",
      tags: ["license:mit"],
      private: false,
      gated: false
    }])
  });
  assert.equal(report.candidates[0].declaredLicense, "mit");
  assert.equal(report.candidates[0].licenseEvidence, "hub-license-tag");
  assert.equal(report.candidates[0].trainingApproved, false);
});

test("rejects invalid discovery queries and limits before making requests", async () => {
  let called = false;
  const fetchImpl = async () => {
    called = true;
    return jsonResponse([]);
  };
  await assert.rejects(discoverPublicDatasets(" "), /1 to 120 printable characters/);
  await assert.rejects(discoverPublicDatasets("x".repeat(121)), /1 to 120 printable characters/);
  await assert.rejects(discoverPublicDatasets("science", { fetchImpl, limit: 26 }), /between 1 and 25/);
  assert.equal(called, false);
});

test("fails transparently for malformed and unavailable Hub responses", async () => {
  await assert.rejects(
    discoverPublicDatasets("science", { fetchImpl: async () => jsonResponse({ datasets: [] }) }),
    /invalid response/
  );
  await assert.rejects(
    discoverPublicDatasets("science", { fetchImpl: async () => jsonResponse({}, 503) }),
    /HTTP 503/
  );
});

test("rejects invalid metadata identifiers and caps the result set", async () => {
  const report = await discoverPublicDatasets("education", {
    limit: 2,
    fetchImpl: async () => jsonResponse([
      { id: "org/first", downloads: 4, private: false, gated: false },
      { id: "org/second", downloads: 50, private: false, gated: false },
      { id: "not-a-valid-id", downloads: 1000, private: false, gated: false },
      { id: "org/third", downloads: 1, private: false, gated: false }
    ])
  });
  assert.deepEqual(report.candidates.map(item => item.id), ["org/second", "org/first"]);
  assert.equal(report.candidateCount, 2);
});
