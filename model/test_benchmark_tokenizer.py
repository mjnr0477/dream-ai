import unittest

from benchmark_tokenizer import benchmark


class TokenizerBenchmarkTests(unittest.TestCase):
    def test_reports_lossless_round_trip_and_efficiency(self):
        text = ("DREAM independent intelligence learns from reviewed data. " * 30
                + "akili bandia kutoka Kenya 🇰🇪 " * 10)
        report, tokenizer = benchmark(text, vocab_size=320)
        self.assertEqual(report["schema"], "dream-tokenizer-benchmark-v1")
        self.assertTrue(report["round_trip_exact"])
        self.assertEqual(report["corpus_utf8_bytes"], len(text.encode("utf-8")))
        self.assertEqual(report["bpe_token_count"], len(tokenizer.encode(text)))
        self.assertEqual(report["tokens_saved"],
                         report["byte_token_count"] - report["bpe_token_count"])
        self.assertGreater(report["token_reduction_fraction"], 0)

    def test_rejects_empty_corpus(self):
        with self.assertRaises(ValueError):
            benchmark("")


if __name__ == "__main__":
    unittest.main()
