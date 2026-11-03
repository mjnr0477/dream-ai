import hashlib
import unittest

from benchmark_tokenizer import benchmark


class TokenizerBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.training_text = (
            "DREAM independent intelligence learns from reviewed data. " * 30
            + "akili bandia kutoka Kenya 🇰🇪 " * 10
        )
        self.evaluation_text = (
            "Reviewed data helps DREAM learn independent intelligence. " * 12
            + "akili bandia kutoka Kenya 🇰🇪 " * 5
        )

    def test_reports_lossless_held_out_efficiency_and_provenance(self):
        report, tokenizer = benchmark(
            self.training_text, self.evaluation_text, vocab_size=320
        )
        self.assertEqual(report["schema"], "dream-tokenizer-benchmark-v2")
        self.assertEqual(report["evaluation_scope"], "held-out")
        self.assertTrue(report["round_trip_exact"])
        self.assertEqual(
            report["corpus_utf8_bytes"], len(self.evaluation_text.encode("utf-8"))
        )
        self.assertEqual(
            report["bpe_token_count"], len(tokenizer.encode(self.evaluation_text))
        )
        self.assertEqual(
            report["tokens_saved"],
            report["byte_token_count"] - report["bpe_token_count"]
        )
        self.assertEqual(
            report["training_corpus_sha256"],
            hashlib.sha256(self.training_text.encode("utf-8")).hexdigest()
        )
        self.assertEqual(
            report["evaluation_corpus_sha256"],
            hashlib.sha256(self.evaluation_text.encode("utf-8")).hexdigest()
        )
        self.assertEqual(len(report["tokenizer_sha256"]), 64)
        self.assertGreater(report["token_reduction_fraction"], 0)

    def test_rejects_empty_training_or_evaluation_corpus(self):
        with self.assertRaises(ValueError):
            benchmark("", self.evaluation_text)
        with self.assertRaises(ValueError):
            benchmark(self.training_text, "")

    def test_rejects_reusing_training_text_as_held_out_data(self):
        with self.assertRaises(ValueError):
            benchmark(self.training_text, self.training_text)


if __name__ == "__main__":
    unittest.main()
