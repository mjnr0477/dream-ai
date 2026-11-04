import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from evaluate_multilingual import evaluate_tokens, load_language_test_corpora


class MeanByteModel:
    def __call__(self, inputs, targets):
        return None, inputs.float().mean()


class MultilingualEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rows = [
            {
                "id": "en-1", "language": "en", "text": "English held out text " * 5,
                "rights_confirmed": True, "privacy_reviewed": True, "quality_reviewed": True,
            },
            {
                "id": "sw-1", "language": "sw", "text": "Kiswahili held out text " * 5,
                "rights_confirmed": True, "privacy_reviewed": True, "quality_reviewed": True,
            },
        ]
        self.records = self.root / "records.jsonl"
        self.records.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in self.rows) + "\n",
            encoding="utf-8",
        )
        self.test_text = "\n\n".join(row["text"].strip() for row in self.rows) + "\n"
        self.test_path = self.root / "test.txt"
        self.test_path.write_text(self.test_text, encoding="utf-8")
        digest = hashlib.sha256(self.test_path.read_bytes()).hexdigest()
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({
            "schema": "dream-corpus-v1",
            "splits": {
                "test": {
                    "sha256": digest,
                    "record_ids": ["en-1", "sw-1"],
                    "language_counts": {"en": 1, "sw": 1},
                }
            },
        }), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_reconstructs_exact_test_and_separates_languages(self):
        corpora, digest, counts = load_language_test_corpora(
            self.records, self.manifest, self.test_path
        )
        self.assertEqual(set(corpora), {"en", "sw"})
        self.assertIn("English", corpora["en"])
        self.assertIn("Kiswahili", corpora["sw"])
        self.assertEqual(counts, {"en": 1, "sw": 1})
        self.assertEqual(digest, hashlib.sha256(self.test_path.read_bytes()).hexdigest())

    def test_rejects_modified_test_file(self):
        self.test_path.write_text("tampered held-out text\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_language_test_corpora(self.records, self.manifest, self.test_path)

    def test_rejects_missing_review_attestation(self):
        self.rows[0]["privacy_reviewed"] = False
        self.records.write_text(
            "\n".join(json.dumps(row) for row in self.rows) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "review attestations"):
            load_language_test_corpora(self.records, self.manifest, self.test_path)

    def test_rejects_incomplete_manifest_record_ids(self):
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        manifest["splits"]["test"]["record_ids"] = ["en-1"]
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reconstruct the exact held-out test file"):
            load_language_test_corpora(self.records, self.manifest, self.test_path)

    def test_evaluation_is_deterministic_and_reports_byte_metrics(self):
        tokens = torch.arange(128, dtype=torch.long)
        first = evaluate_tokens(MeanByteModel(), tokens, 8, 4, 2, 42, "cpu")
        second = evaluate_tokens(MeanByteModel(), tokens, 8, 4, 2, 42, "cpu")
        self.assertEqual(first, second)
        self.assertEqual(first["evaluation_windows"], 8)
        self.assertGreater(first["mean_cross_entropy_nats_per_byte"], 0)
        self.assertGreater(first["bits_per_byte"], 0)
        self.assertGreater(first["perplexity_per_byte"], 1)


if __name__ == "__main__":
    unittest.main()
