"""Tests for language-aware pretraining sampling."""
import unittest
from unittest.mock import patch

import torch

from dream_transformer import DreamByteTokenizer
from train import balanced_batches, split_language_streams


class LanguageSamplingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DreamByteTokenizer()
        self.text = (
            "<|language:en|>\n" + ("English training material with enough tokens. " * 5)
            + "<|language:sw|>\n" + ("Maandishi ya Kiswahili yenye maneno ya kutosha. " * 5)
        )

    def test_splits_tagged_documents_into_distinct_language_streams(self):
        streams = split_language_streams(self.text, self.tokenizer, block_size=16)
        self.assertEqual(set(streams), {"en", "sw"})
        self.assertGreaterEqual(len(streams["en"]), 18)
        self.assertGreaterEqual(len(streams["sw"]), 18)

    def test_balanced_batch_has_correct_next_token_targets(self):
        streams = split_language_streams(self.text, self.tokenizer, block_size=16)
        with patch("train.random.choice", side_effect=lambda languages: languages[0]):
            x, y = balanced_batches(streams, batch_size=2, block_size=16, device="cpu")
        self.assertEqual(tuple(x.shape), (2, 16))
        self.assertEqual(tuple(y.shape), (2, 16))
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))

    def test_requires_language_markers(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            split_language_streams("plain untagged corpus " * 20, self.tokenizer, 8)

    def test_rejects_language_stream_shorter_than_context(self):
        text = "<|language:en|>\nshort"
        with self.assertRaisesRegex(ValueError, "block_size"):
            split_language_streams(text, self.tokenizer, 16)

    def test_empty_balanced_streams_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "No language streams"):
            balanced_batches({}, 1, 8, "cpu")


if __name__ == "__main__":
    unittest.main()
