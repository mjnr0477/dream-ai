import tempfile
import unittest
from pathlib import Path

from collections import Counter

from dream_bpe import DreamBPETokenizer


class DreamBPETokenizerTests(unittest.TestCase):
    def test_utf8_round_trip_including_kiswahili_and_emoji(self):
        sample = "DREAM AI — akili huru kutoka Kenya 🇰🇪"
        tokenizer = DreamBPETokenizer.train(sample * 5, vocab_size=320)
        self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)

    def test_incremental_training_matches_full_rescan_reference(self):
        import re

        def reference_train(text, vocab_size, min_frequency):
            frequencies = Counter(re.findall(r"\\s+|[^\\s]+", text))
            sequences = {
                tuple(piece.encode("utf-8")): count
                for piece, count in frequencies.items()
            }
            merges = []
            while 256 + len(merges) < vocab_size:
                pair_counts = Counter()
                for sequence, frequency in sequences.items():
                    for pair in zip(sequence, sequence[1:]):
                        pair_counts[pair] += frequency
                if not pair_counts:
                    break
                best_count = max(pair_counts.values())
                if best_count < min_frequency:
                    break
                best_pair = min(
                    pair for pair, count in pair_counts.items()
                    if count == best_count
                )
                new_id = 256 + len(merges)
                merges.append(best_pair)
                updated = {}
                for sequence, frequency in sequences.items():
                    merged = []
                    index = 0
                    while index < len(sequence):
                        if (index + 1 < len(sequence)
                                and (sequence[index], sequence[index + 1]) == best_pair):
                            merged.append(new_id)
                            index += 2
                        else:
                            merged.append(sequence[index])
                            index += 1
                    key = tuple(merged)
                    updated[key] = updated.get(key, 0) + frequency
                sequences = updated
            return merges

        corpora = [
            "aaaa aaaa aaa ababab ababa\\n" * 5,
            "akili bandia Kenya 🇰🇪 こんにちは مرحبا\\n" * 7,
            "abcabc abcabc abccba xyzxyz\\n" * 4,
            "one two three four five\\n" * 3 + "one one two two\\n" * 5,
        ]
        for corpus in corpora:
            for minimum in (1, 2, 4, 9):
                expected = reference_train(corpus, 310, minimum)
                actual = DreamBPETokenizer.train(
                    corpus, vocab_size=310, min_frequency=minimum
                ).merges
                with self.subTest(min_frequency=minimum, corpus=corpus[:12]):
                    self.assertEqual(actual, expected)

    def test_incremental_training_handles_overlapping_pairs_left_to_right(self):
        corpus = "aaaaa " * 12
        tokenizer = DreamBPETokenizer.train(corpus, vocab_size=270, min_frequency=2)
        self.assertEqual(
            tokenizer.decode(tokenizer.encode(corpus)), corpus
        )

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


    def test_optimized_encoding_matches_sequential_merge_reference(self):
        corpus = ("DREAM AI learns languages safely. akili bandia Kenya 🇰🇪\n" * 30)
        tokenizer = DreamBPETokenizer.train(corpus, vocab_size=340)
        samples = [
            corpus[:120],
            "unseen multilingual text: こんにちは مرحبا Kiswahili",
            "AI AI AI — 🇰🇪",
            "  whitespace\tand\nnewlines  ",
        ]

        def reference_encode(text):
            output = []
            import re
            for piece in re.findall(r"\s+|[^\s]+", text):
                tokens = list(piece.encode("utf-8"))
                for merge_index, (left, right) in enumerate(tokenizer.merges):
                    new_id = tokenizer.base_vocab_size + merge_index
                    merged = []
                    index = 0
                    while index < len(tokens):
                        if (index + 1 < len(tokens)
                                and tokens[index] == left
                                and tokens[index + 1] == right):
                            merged.append(new_id)
                            index += 2
                        else:
                            merged.append(tokens[index])
                            index += 1
                    tokens = merged
                output.extend(tokens)
            return output

        for sample in samples:
            with self.subTest(sample=sample[:24]):
                self.assertEqual(tokenizer.encode(sample), reference_encode(sample))
                self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)

    def test_token_byte_length_matches_merged_utf8_piece_lengths(self):
        sample = "akili bandia Kenya 🇰🇪 " * 8
        tokenizer = DreamBPETokenizer.train(sample, vocab_size=320)
        encoded = tokenizer.encode(sample)
        self.assertEqual(
            sum(tokenizer.token_byte_length(token) for token in encoded),
            len(sample.encode("utf-8"))
        )
        with self.assertRaises(ValueError):
            tokenizer.token_byte_length(tokenizer.vocab_size)

    def test_vocab_size_is_bounded(self):
        with self.assertRaises(ValueError):
            DreamBPETokenizer.train("text", vocab_size=255)


if __name__ == "__main__":
    unittest.main()
