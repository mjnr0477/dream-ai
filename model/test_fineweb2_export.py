"""Tests for FineWeb2 streaming export row filtering."""
import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "data" / "export_fineweb2_sample.py"
SPEC = importlib.util.spec_from_file_location("fineweb2_exporter", MODULE_PATH)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def row():
    return {
        "id": "doc-1",
        "text": "A sufficiently long Swahili or English web document body.",
        "url": "https://example.org/article",
        "date": "2024-01-01T12:00:00Z",
        "language": "swa",
        "language_script": "Latn",
        "language_score": 0.99,
        "minhash_cluster_size": 1,
    }


class FineWeb2ExportTests(unittest.TestCase):
    def test_exports_only_required_provenance_fields(self):
        result = exporter.export_row(row())
        self.assertEqual(result["language"], "swa")
        self.assertNotIn("language_score", result)
        self.assertNotIn("minhash_cluster_size", result)
        self.assertEqual(set(result), {
            "id", "text", "url", "date", "language", "language_script"
        })

    def test_rejects_low_language_confidence(self):
        source = row()
        source["language_score"] = 0.80
        self.assertIsNone(exporter.export_row(source))

    def test_rejects_non_finite_language_scores(self):
        source = row()
        source["language_score"] = float("nan")
        self.assertIsNone(exporter.export_row(source))
        source = row()
        source["language_score"] = float("inf")
        self.assertIsNone(exporter.export_row(source))

    def test_rejects_missing_source_provenance(self):
        source = row()
        del source["url"]
        self.assertIsNone(exporter.export_row(source))

    def test_rejects_missing_or_invalid_language_score(self):
        source = row()
        source["language_score"] = None
        self.assertIsNone(exporter.export_row(source))
        source = row()
        source["language_score"] = "not-a-number"
        self.assertIsNone(exporter.export_row(source))

    def test_rejects_non_object_rows(self):
        self.assertIsNone(exporter.export_row("not a row"))


if __name__ == "__main__":
    unittest.main()
