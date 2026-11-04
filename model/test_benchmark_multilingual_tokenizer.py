import unittest

from benchmark_multilingual_tokenizer import benchmark_multilingual


class MultilingualTokenizerBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.train = [
            {"language": "en", "text": "DREAM independent intelligence " * 8},
            {"language": "sw", "text": "DREAM akili bandia kutoka Kenya " * 8},
            {"language": "ar", "text": "الذكاء الاصطناعي المستقل " * 8}
        ]
        self.test = [
            {"language": "en", "text": "DREAM independent intelligence learns safely."},
            {"language": "sw", "text": "Akili bandia inaweza kujifunza kwa usalama."},
            {"language": "ar", "text": "يمكن للذكاء الاصطناعي أن يتعلم بأمان."},
            {"language": "ja", "text": "人工知能は安全に学習できます。"}
        ]

    def test_reports_held_out_efficiency_by_language_and_unicode_round_trip(self):
        report, tokenizer = benchmark_multilingual(
            self.train, self.test, vocab_size=320
        )
        self.assertEqual(report["schema"], "dream-multilingual-tokenizer-benchmark-v1")
        self.assertTrue(report["held_out_text"])
        self.assertTrue(report["overall"]["round_trip_exact"])
        self.assertEqual(report["test_languages"], ["ar", "en", "ja", "sw"])
        self.assertTrue(all(row["round_trip_exact"]
                            for row in report["by_language"].values()))
        for language, row in report["by_language"].items():
            sample = next(item["text"] for item in self.test
                          if item["language"] == language)
            self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)

    def test_supports_bcp47_script_region_and_variant_subtags(self):
        train = self.train + [
            {"language": "zh-hant", "text": "繁體中文訓練資料與人工智慧。 " * 8}
        ]
        test = [
            {"language": "zh-hant-tw", "text": "繁體中文測試資料與智慧系統。"},
            {"language": "sr-latn-rs", "text": "Ovo je nezavisan srpski tekst."},
            {"language": "en-001", "text": "A global English test sentence."},
            {"language": "ca-es-valencia", "text": "Aquest és un text de prova valencià."}
        ]
        report, _ = benchmark_multilingual(train, test, vocab_size=320)
        self.assertEqual(
            report["test_languages"],
            ["ca-ES-valencia", "en-001", "sr-Latn-RS", "zh-Hant-TW"]
        )

    def test_rejects_train_test_text_overlap(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            benchmark_multilingual(
                self.train,
                self.test + [{"language": "en", "text": self.train[0]["text"]}]
            )

    def test_rejects_invalid_language_tag(self):
        with self.assertRaisesRegex(ValueError, "language tag"):
            benchmark_multilingual(
                self.train,
                [{"language": "English", "text": "A sufficiently long sample."}]
            )

    def test_rejects_empty_inputs(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            benchmark_multilingual([], self.test)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            benchmark_multilingual(self.train, [])


if __name__ == "__main__":
    unittest.main()
