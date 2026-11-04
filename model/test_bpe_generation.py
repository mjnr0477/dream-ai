"""Tests for UTF-8-safe autoregressive generation with byte-fallback BPE."""
import unittest

import torch

from dream_bpe import DreamBPETokenizer
from dream_transformer import DreamConfig, DreamTransformer


class DreamBPEGenerationTests(unittest.TestCase):
    def test_piece_transition_rejects_invalid_utf8_and_finishes_sequences(self):
        config = DreamConfig(vocab_size=259, block_size=8, d_model=16,
                             n_heads=4, n_layers=1, dropout=0.0)
        model = DreamTransformer(config).eval()
        pieces = tuple(bytes([value]) for value in range(256)) + (
            b"\xc3", b"\xa9", b"\xff"
        )
        logits = torch.zeros((1, config.vocab_size))
        logits[0, 258] = 100.0
        logits[0, 256] = 10.0
        token, state = model._sample_next_bpe_token(
            logits, 0, 0, [(0, 0x80, 0xBF)], pieces, 1
        )
        self.assertEqual(token.item(), 256)
        self.assertEqual(state, [(1, 0x80, 0xBF)])
        next_logits = torch.zeros((1, config.vocab_size))
        next_logits[0, 257] = 10.0
        next_logits[0, 258] = 100.0
        token, state = model._sample_next_bpe_token(
            next_logits, 0, 0, state, pieces, 0
        )
        self.assertEqual(token.item(), 257)
        self.assertEqual(state, [(0, 0x80, 0xBF)])

    def test_utf8_transition_rejects_overlong_surrogate_and_out_of_range(self):
        config = DreamConfig(vocab_size=256, block_size=8, d_model=16,
                             n_heads=4, n_layers=1, dropout=0.0)
        model = DreamTransformer(config)
        valid = tuple(bytes.fromhex(value) for value in (
            "e0a080", "ed9fbf", "f0908080", "f48fbfbf"
        ))
        invalid = tuple(bytes.fromhex(value) for value in (
            "c080", "e08080", "eda080", "f4908080", "f5808080"
        ))
        for piece in valid:
            with self.subTest(piece=piece):
                self.assertEqual(
                    model._advance_utf8_piece((0, 0x80, 0xBF), piece),
                    (0, 0x80, 0xBF)
                )
        for piece in invalid:
            with self.subTest(piece=piece):
                self.assertIsNone(model._advance_utf8_piece((0, 0x80, 0xBF), piece))

    def test_generation_emits_valid_utf8_with_cached_and_sliding_paths(self):
        sample = "Habari dunia — akili huru kutoka Kenya 🇰🇪 こんにちは مرحبا"
        tokenizer = DreamBPETokenizer.train(sample * 10, vocab_size=320)
        self.assertEqual(len(tokenizer.token_byte_pieces), tokenizer.vocab_size)
        config = DreamConfig(vocab_size=tokenizer.vocab_size, block_size=16,
                             d_model=16, n_heads=4, n_layers=1, dropout=0.0)
        model = DreamTransformer(config).eval()
        prompt_ids = tokenizer.encode("DREAM:")
        prompt = torch.tensor([prompt_ids], dtype=torch.long)
        for max_new_tokens in (4, 20):
            with self.subTest(max_new_tokens=max_new_tokens):
                torch.manual_seed(23)
                generated = model.generate(
                    prompt.clone(), max_new_tokens=max_new_tokens,
                    temperature=0.8, top_k=12, constrain_utf8=True,
                    token_byte_pieces=tokenizer.token_byte_pieces
                )[0].tolist()
                output_ids = generated[len(prompt_ids):]
                output_bytes = b"".join(
                    tokenizer.token_byte_pieces[token_id] for token_id in output_ids
                )
                decoded = output_bytes.decode("utf-8")
                self.assertNotIn("\ufffd", decoded)


if __name__ == "__main__":
    unittest.main()
