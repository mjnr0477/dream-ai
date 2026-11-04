import unittest
from checkpoint_tokenizer import load_tokenizer, tokenizer_name
from dream_bpe import DreamBPETokenizer
from dream_transformer import DreamByteTokenizer, DreamConfig


class CheckpointTokenizerTests(unittest.TestCase):
    def test_loads_legacy_byte_checkpoint(self):
        tokenizer = load_tokenizer({"tokenizer": "utf8-byte-v1"}, DreamConfig())
        self.assertIsInstance(tokenizer, DreamByteTokenizer)
        self.assertEqual(tokenizer_name("utf8-byte-v1"), "utf8-byte-v1")

    def test_loads_embedded_bpe_checkpoint(self):
        learned = DreamBPETokenizer.train("habari dunia " * 8, vocab_size=280)
        config = DreamConfig(vocab_size=learned.vocab_size)
        tokenizer = load_tokenizer({"tokenizer": learned.to_dict()}, config)
        sample = "habari dunia — Kenya 🇰🇪"
        self.assertIsInstance(tokenizer, DreamBPETokenizer)
        self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)
        self.assertEqual(tokenizer_name(learned.to_dict()), "dream-bpe-v1")

    def test_rejects_model_vocabulary_mismatch(self):
        learned = DreamBPETokenizer.train("hello world " * 8, vocab_size=280)
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_tokenizer({"tokenizer": learned.to_dict()}, DreamConfig(vocab_size=512))

    def test_rejects_unknown_tokenizer_metadata(self):
        with self.assertRaisesRegex(ValueError, "(?i)unsupported"):
            load_tokenizer({"tokenizer": "borrowed-vocabulary"}, DreamConfig())

    def test_rejects_malformed_bpe_metadata(self):
        with self.assertRaises(ValueError):
            load_tokenizer({"tokenizer": {"format": "dream-bpe-v1", "vocab_size": 300,
                                          "merges": [[999, 1]]}},
                           DreamConfig(vocab_size=300))


if __name__ == "__main__":
    unittest.main()
