"""Tests for DREAM's exact/normalized benchmark-contamination audit."""
import json
import tempfile
import unittest
from pathlib import Path

from audit_benchmark_contamination import audit, load_benchmark, normalize_text, write_json_atomic


class BenchmarkContaminationAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.benchmark = self.root / "benchmark.jsonl"
        self.benchmark.write_text(json.dumps({
            "id": "task-1",
            "category": "reasoning",
            "prompt": "Explain how a learner can check a long division answer using multiplication.",
            "rubric": ["Explains that multiplying the quotient by the divisor should recover the dividend."],
        }) + "\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_normalizer_matches_nfkc_case_and_whitespace(self):
        self.assertEqual(normalize_text("  ＤＲＥＡＭ   AI\nDATA "), "dream ai data")

    def test_detects_normalized_prompt_overlap_without_disclosing_text(self):
        corpus = self.root / "train.jsonl"
        secret_record = {
            "id": "private-record-123",
            "source_group": "source-group-9",
            "text": "Earlier note: EXPLAIN HOW A LEARNER CAN CHECK A LONG DIVISION ANSWER USING MULTIPLICATION. End.",
        }
        corpus.write_text(json.dumps(secret_record) + "\n", encoding="utf-8")
        report = audit(self.benchmark, [corpus])
        self.assertEqual(report["status"], "overlap_detected")
        self.assertEqual(report["overlap_count"], 1)
        overlap = report["overlaps"][0]
        self.assertEqual(overlap["task_id"], "task-1")
        self.assertEqual(overlap["benchmark_field"], "prompt")
        self.assertEqual(len(overlap["corpus_record_sha256"]), 64)
        rendered = json.dumps(report)
        self.assertNotIn("private-record-123", rendered)
        self.assertNotIn(secret_record["text"], rendered)
        self.assertNotIn("source-group-9", rendered)

    def test_detects_sft_message_content(self):
        corpus = self.root / "sft.jsonl"
        corpus.write_text(json.dumps({
            "id": "dialogue-1",
            "messages": [
                {"role": "user", "content": "Please explain how a learner can check a long division answer using multiplication."},
                {"role": "assistant", "content": "Multiplication can verify division by recovering the dividend."},
            ],
        }) + "\n", encoding="utf-8")
        report = audit(self.benchmark, [corpus])
        self.assertGreaterEqual(report["overlap_count"], 1)
        self.assertIn("source_group_sha256", report["overlaps"][0])

    def test_no_overlap_is_reported_as_a_limited_clean_screen(self):
        corpus = self.root / "unrelated.txt"
        corpus.write_text("Plants use sunlight to make glucose through photosynthesis.", encoding="utf-8")
        report = audit(self.benchmark, [corpus])
        self.assertEqual(report["status"], "no_exact_normalized_overlap_found")
        self.assertEqual(report["overlap_count"], 0)
        self.assertTrue(any("does not detect paraphrases" in item for item in report["limitations"]))

    def test_rejects_duplicate_benchmark_ids(self):
        duplicate = self.root / "duplicate.jsonl"
        task = json.loads(self.benchmark.read_text(encoding="utf-8"))
        duplicate.write_text(json.dumps(task) + "\n" + json.dumps(task) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate benchmark task id"):
            load_benchmark(duplicate)

    def test_rejects_jsonl_corpus_without_text_fields(self):
        corpus = self.root / "metadata-only.jsonl"
        corpus.write_text(json.dumps({"id": "row-1", "source": "catalogue"}) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no recognized text fields"):
            audit(self.benchmark, [corpus])

    def test_writes_report_atomically(self):
        corpus = self.root / "corpus.txt"
        corpus.write_text("An unrelated short document about plants.", encoding="utf-8")
        report = audit(self.benchmark, [corpus])
        output = self.root / "reports" / "audit.json"
        write_json_atomic(output, report)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
        self.assertEqual([item.name for item in output.parent.iterdir()], ["audit.json"])


if __name__ == "__main__":
    unittest.main()
