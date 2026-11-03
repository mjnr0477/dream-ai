import tempfile
import unittest
from pathlib import Path

from dream_bpe import DreamBPETokenizer


class DreamBPETokenizerTests(unittest.TestCase):
    def test_utf8_round_trip_including_kiswahili_and_emoji(self):
        sample = "DREAM AI — akili huru kutoka Kenya 🇰🇪"
        tokenizer = DreamBPETokenizer.train(sample * 5, vocab_size=320)
        self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)

    def test_training_is_deterministic(self):
        corpus = ("dream ai builds independent intelligence\n" * 20
                  + "akili bandia ya kujitegemea\n" * 10)
        first = DreamBPETokenizer.train(corpus, vocab_size=300)
        second = DreamBPETokenizer.train(corpus, vocab_size=300)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_repeated_text_uses_fewer_tokens_than_utf8_bytes(self):
        sample = "independent intelligence " * 40
        tokenizer = DreamBPETokenizer.train(sample, vocab_size=320)
        self.assertLess(len(tokenizer.encode(sample)), len(sample.encode("utf-8")))
        self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)

    def test_save_load_preserves_vocabulary_and_encoding(self):
        sample = "DREAM AI — Kenya 🇰🇪 " * 8
        tokenizer = DreamBPETokenizer.train(sample, vocab_size=300)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dream-tokenizer.json"
            tokenizer.save(path)
            restored = DreamBPETokenizer.load(path)
        self.assertEqual(restored.to_dict(), tokenizer.to_dict())
        self.assertEqual(restored.encode(sample), tokenizer.encode(sample))
        self.assertEqual(restored.decode(restored.encode(sample)), sample)

    def test_byte_fallback_handles_unseen_unicode(self):
        tokenizer = DreamBPETokenizer.train("hello world " * 10, vocab_size=280)
        sample = "こんにちは — mpya 🇰🇪"
        self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)

    def test_rejects_non_integer_and_boolean_merge_ids(self):
        for merges in ([[True, 1]], [[1.5, 2]], [["1", 2]], "not-a-list"):
            with self.subTest(merges=merges):
                with self.assertRaises(ValueError):
                    DreamBPETokenizer(merges)

    def test_rejects_boolean_saved_vocabulary_size(self):
        payload = DreamBPETokenizer().to_dict()
        payload["vocab_size"] = True
        with self.assertRaises(ValueError):
            DreamBPETokenizer.from_dict(payload)

    def test_rejects_invalid_merge_reference(self):
        with self.assertRaises(ValueError):
            DreamBPETokenizer([(256, 1)])

    def test_rejects_unknown_format(self):
        with self.assertRaises(ValueError):
            DreamBPETokenizer.from_dict({"format": "other", "vocab_size": 256,
                                         "merges": []})

    def test_decode_rejects_negative_and_non_integer_ids(self):
        tokenizer = DreamBPETokenizer()
        for invalid in ([-1], [256], [1.5], [True]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    tokenizer.decode(invalid)

    def test_vocab_size_is_bounded(self):
        with self.assertRaises(ValueError):
            DreamBPETokenizer.train("text", vocab_size=255)


if __name__ == "__main__":
    unittest.main()
