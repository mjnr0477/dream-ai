import unittest
import torch

from dream_transformer import DreamByteTokenizer, DreamConfig, DreamTransformer
from train import batches


class DreamTokenizerTests(unittest.TestCase):
    def test_utf8_round_trip(self):
        tokenizer = DreamByteTokenizer()
        sample = "DREAM AI — Kenya 🇰🇪"
        self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)

    def test_byte_vocabulary_is_fixed(self):
        self.assertEqual(DreamByteTokenizer.vocab_size, 256)

    def test_round_trip_across_diverse_writing_systems(self):
        tokenizer = DreamByteTokenizer()
        samples = [
            "Hello, world!", "Habari za asubuhi — Kiswahili",
            "مرحبا بالعالم", "नमस्ते दुनिया", "你好，世界",
            "こんにちは世界", "안녕하세요 세계", "Bonjour le monde",
            "Hola, mundo", "Ẹ káàbọ̀ sí ayé", "ሰላም ዓለም",
            "স্বাগতম পৃথিবী", "Привет, мир", "🌍🚀"
        ]
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)


class DreamTransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = DreamConfig(
            block_size=16, d_model=32, n_heads=4, n_layers=2, dropout=0.0
        )
        self.model = DreamTransformer(self.config).eval()

    def test_forward_shape_and_loss(self):
        tokens = torch.randint(0, 256, (2, 8))
        logits, loss = self.model(tokens, tokens)
        self.assertEqual(tuple(logits.shape), (2, 8, 256))
        self.assertTrue(torch.isfinite(loss).item())

    def test_attention_is_causal(self):
        first = torch.tensor([[10, 20, 30]])
        changed_future = torch.tensor([[10, 99, 88]])
        with torch.inference_mode():
            first_logits, _ = self.model(first)
            changed_logits, _ = self.model(changed_future)
        self.assertTrue(torch.allclose(
            first_logits[:, 0, :], changed_logits[:, 0, :], atol=1e-6
        ))

    def test_generation_extends_prompt(self):
        prompt = torch.tensor([[65, 66, 67]])
        generated = self.model.generate(prompt, max_new_tokens=4, top_k=10)
        self.assertEqual(tuple(generated.shape), (1, 7))
        self.assertTrue(torch.equal(generated[:, :3], prompt))

    def test_incremental_kv_cache_matches_full_forward_logits(self):
        tokens = torch.tensor([[65, 66, 67, 68, 69]], dtype=torch.long)
        with torch.inference_mode():
            full_logits, _ = self.model(tokens)
            prefix_logits, _, cache = self.model(tokens[:, :-1], use_cache=True)
            cached_logits, _, updated_cache = self.model(
                tokens[:, -1:], past_key_values=cache, use_cache=True
            )
        self.assertEqual(len(updated_cache), self.config.n_layers)
        self.assertEqual(updated_cache[0][0].shape[2], tokens.shape[1])
        self.assertTrue(torch.allclose(
            cached_logits[:, -1, :], full_logits[:, -1, :], atol=1e-5
        ))

    def test_cached_greedy_generation_matches_sliding_window_reference(self):
        prompt = torch.tensor(
            [[65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78]],
            dtype=torch.long
        )
        expected = prompt.clone()
        with torch.inference_mode():
            for _ in range(6):
                context = expected[:, -self.config.block_size:]
                logits, _ = self.model(context)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                expected = torch.cat((expected, next_token), dim=1)
            actual = self.model.generate(
                prompt.clone(), max_new_tokens=6, temperature=0
            )
        self.assertTrue(torch.equal(actual, expected))

    def test_cached_forward_rejects_non_single_token_increment(self):
        prefix = torch.tensor([[65, 66, 67]], dtype=torch.long)
        next_tokens = torch.tensor([[68, 69]], dtype=torch.long)
        with torch.inference_mode():
            _, _, cache = self.model(prefix, use_cache=True)
            with self.assertRaisesRegex(ValueError, "one new token"):
                self.model(next_tokens, past_key_values=cache, use_cache=True)

    def test_greedy_generation_is_deterministic(self):
        prompt = torch.tensor([[65, 66, 67]])
        first = self.model.generate(prompt.clone(), max_new_tokens=5, temperature=0)
        second = self.model.generate(prompt.clone(), max_new_tokens=5, temperature=0)
        self.assertTrue(torch.equal(first, second))

    def test_negative_generation_temperature_is_rejected(self):
        prompt = torch.tensor([[65, 66]], dtype=torch.long)
        with self.assertRaises(ValueError):
            self.model.generate(prompt, temperature=-0.1)

    def test_batches_support_exactly_one_valid_next_token_window(self):
        data = torch.tensor([10, 11, 12, 13, 14], dtype=torch.long)
        x, y = batches(data, batch_size=3, block_size=4, device="cpu")
        self.assertEqual(tuple(x.shape), (3, 4))
        self.assertEqual(tuple(y.shape), (3, 4))
        self.assertTrue(torch.equal(x, torch.tensor([[10, 11, 12, 13]]).repeat(3, 1)))
        self.assertTrue(torch.equal(y, torch.tensor([[11, 12, 13, 14]]).repeat(3, 1)))

    def test_training_step_updates_parameters(self):
        data = torch.randint(0, 256, (128,), dtype=torch.long)
        x, y = batches(data, batch_size=2, block_size=8, device="cpu")
        model = DreamTransformer(DreamConfig(
            block_size=8, d_model=16, n_heads=4, n_layers=1, dropout=0.0
        ))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.token_embedding.weight.detach().clone()
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.token_embedding.weight.detach()))

    def test_invalid_head_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            DreamConfig(d_model=30, n_heads=8)

    def test_zero_heads_are_rejected_cleanly(self):
        with self.assertRaises(ValueError):
            DreamConfig(n_heads=0)

    def test_extended_vocabulary_is_supported_for_bpe(self):
        config = DreamConfig(vocab_size=512)
        model = DreamTransformer(config)
        tokens = torch.randint(0, 512, (2, 8))
        logits, loss = model(tokens, tokens)
        self.assertEqual(tuple(logits.shape), (2, 8, 512))
        self.assertTrue(torch.isfinite(loss).item())

    def test_vocabulary_below_byte_fallback_is_rejected(self):
        with self.assertRaises(ValueError):
            DreamConfig(vocab_size=255)

    def test_invalid_dropout_is_rejected(self):
        with self.assertRaises(ValueError):
            DreamConfig(dropout=1.0)

    def test_invalid_generation_temperature_is_rejected(self):
        prompt = torch.tensor([[65, 66]], dtype=torch.long)
        with self.assertRaises(ValueError):
            self.model.generate(prompt, temperature=float("nan"))

    def test_invalid_target_shape_is_rejected(self):
        prompt = torch.tensor([[65, 66]], dtype=torch.long)
        targets = torch.tensor([[66]], dtype=torch.long)
        with self.assertRaises(ValueError):
            self.model(prompt, targets)

    def test_out_of_range_input_token_ids_are_rejected(self):
        for value in (-1, 256):
            with self.subTest(value=value):
                tokens = torch.tensor([[value]], dtype=torch.long)
                with self.assertRaisesRegex(ValueError, "outside the configured vocabulary"):
                    self.model(tokens)

    def test_out_of_range_target_token_ids_are_rejected(self):
        tokens = torch.tensor([[65, 66]], dtype=torch.long)
        targets = torch.tensor([[66, 256]], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "targets contain IDs"):
            self.model(tokens, targets)

    def test_ignored_target_labels_are_allowed_for_sft(self):
        tokens = torch.tensor([[65, 66, 67]], dtype=torch.long)
        targets = torch.tensor([[-100, -100, 67]], dtype=torch.long)
        _, loss = self.model(tokens, targets)
        self.assertTrue(torch.isfinite(loss).item())

    def test_all_ignored_targets_are_rejected(self):
        tokens = torch.tensor([[65, 66]], dtype=torch.long)
        targets = torch.tensor([[-100, -100]], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "at least one supervised token"):
            self.model(tokens, targets)

    def test_boolean_generation_temperature_is_rejected(self):
        prompt = torch.tensor([[65]], dtype=torch.long)
        with self.assertRaises(ValueError):
            self.model.generate(prompt, temperature=True)

    def test_empty_batches_are_rejected(self):
        tokens = torch.empty((0, 2), dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.model(tokens)

    def test_boolean_dimensions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive integers"):
            DreamConfig(n_layers=True)

    def test_boolean_generation_limits_are_rejected(self):
        prompt = torch.tensor([[65]], dtype=torch.long)
        with self.assertRaises(ValueError):
            self.model.generate(prompt, max_new_tokens=True)
        with self.assertRaises(ValueError):
            self.model.generate(prompt, top_k=True)

    def test_boolean_and_nonfinite_dropout_are_rejected(self):
        for dropout in (True, float("nan"), float("inf")):
            with self.subTest(dropout=dropout):
                with self.assertRaisesRegex(ValueError, "dropout"):
                    DreamConfig(dropout=dropout)


if __name__ == "__main__":
    unittest.main()
