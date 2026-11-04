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

    def test_decode_rejects_out_of_range_and_non_integer_ids(self):
        tokenizer = DreamByteTokenizer()
        for invalid in ([-1], [256], [65.5], [True]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    tokenizer.decode(invalid)

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

    def test_greedy_generation_is_deterministic(self):
        prompt = torch.tensor([[65, 66, 67]])
        first = self.model.generate(prompt.clone(), max_new_tokens=5, temperature=0)
        second = self.model.generate(prompt.clone(), max_new_tokens=5, temperature=0)
        self.assertTrue(torch.equal(first, second))

    def test_cached_prefill_and_incremental_logits_match_full_forward(self):
        prompt = torch.tensor([[65, 66, 67, 68]])
        with torch.inference_mode():
            cached_logits, cache = self.model.forward_cached(prompt)
            full_logits, _ = self.model(prompt)
            self.assertTrue(torch.allclose(cached_logits, full_logits, atol=1e-6, rtol=1e-5))
            history = prompt
            for token in (torch.tensor([[69]]), torch.tensor([[70]]), torch.tensor([[71]])):
                cached_logits, cache = self.model.forward_cached(token, cache)
                history = torch.cat((history, token), dim=1)
                full_logits, _ = self.model(history)
                self.assertTrue(torch.allclose(
                    cached_logits[:, -1, :], full_logits[:, -1, :],
                    atol=1e-6, rtol=1e-5
                ))

    def test_cached_greedy_generation_matches_full_recomputation(self):
        prompt = torch.tensor([[65, 66, 67]])
        expected = prompt.clone()
        with torch.inference_mode():
            for _ in range(6):
                logits, _ = self.model(expected[:, -self.config.block_size:])
                expected = torch.cat((
                    expected, torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                ), dim=1)
            actual = self.model.generate(prompt.clone(), max_new_tokens=6, temperature=0)
        self.assertTrue(torch.equal(actual, expected))

    def test_generation_falls_back_to_sliding_window_when_cache_would_overflow(self):
        prompt = torch.tensor([[65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78]])
        expected = prompt.clone()
        with torch.inference_mode():
            for _ in range(5):
                logits, _ = self.model(expected[:, -self.config.block_size:])
                expected = torch.cat((
                    expected, torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                ), dim=1)
            actual = self.model.generate(prompt.clone(), max_new_tokens=5, temperature=0)
        self.assertTrue(torch.equal(actual, expected))

    def test_generated_bytes_are_valid_utf8_for_batched_cached_and_sliding_paths(self):
        tokenizer = DreamByteTokenizer()
        prompt = torch.tensor([[65, 66, 67], [72, 73, 74]], dtype=torch.long)
        cached = self.model.generate(
            prompt.clone(), max_new_tokens=6, temperature=0.9, top_k=256
        )
        sliding = self.model.generate(
            prompt.clone(), max_new_tokens=40, temperature=0.9, top_k=256
        )
        for generated in (cached, sliding):
            for row in generated[:, prompt.shape[1]:].tolist():
                bytes(row).decode("utf-8", errors="strict")

    def test_utf8_masks_respect_lead_byte_budget_and_unicode_scalar_ranges(self):
        pending = torch.zeros(1, dtype=torch.long)
        minimum = torch.full((1,), 0x80, dtype=torch.long)
        maximum = torch.full((1,), 0xBF, dtype=torch.long)

        last_step = self.model._utf8_next_token_mask(pending, minimum, maximum, 0)
        self.assertTrue(last_step[0, :0x80].all().item())
        self.assertFalse(last_step[0, 0xC2:].any().item())

        two_steps_left = self.model._utf8_next_token_mask(pending, minimum, maximum, 1)
        self.assertTrue(two_steps_left[0, 0xC2:0xE0].all().item())
        self.assertFalse(two_steps_left[0, 0xE0:].any().item())

        for lead, expected_min, expected_max, remaining_after in (
            (0xE0, 0xA0, 0xBF, 1), (0xED, 0x80, 0x9F, 1),
            (0xF0, 0x90, 0xBF, 2), (0xF4, 0x80, 0x8F, 2),
        ):
            state = self.model._advance_utf8_state(
                pending, minimum, maximum, torch.tensor([lead])
            )
            allowed = self.model._utf8_next_token_mask(*state, remaining_after)
            self.assertTrue(allowed[0, expected_min:expected_max + 1].all().item())
            self.assertEqual(int(allowed[0].sum().item()), expected_max - expected_min + 1)

    def test_cached_forward_requires_eval_mode_and_valid_cache(self):
        prompt = torch.tensor([[65, 66]])
        self.model.train()
        with self.assertRaisesRegex(RuntimeError, "model.eval"):
            self.model.forward_cached(prompt)
        self.model.eval()
        with torch.inference_mode():
            _, cache = self.model.forward_cached(prompt)
            with self.assertRaisesRegex(ValueError, "exactly one new token"):
                self.model.forward_cached(torch.tensor([[67, 68]]), cache)
            with self.assertRaisesRegex(ValueError, "one key/value pair per layer"):
                self.model.forward_cached(torch.tensor([[67]]), cache[:1])

    def test_negative_generation_temperature_is_rejected(self):
        prompt = torch.tensor([[65, 66]], dtype=torch.long)
        with self.assertRaises(ValueError):
            self.model.generate(prompt, temperature=-0.1)

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

    def test_extended_vocabulary_supports_bpe_token_ids(self):
        config = DreamConfig(
            vocab_size=512, block_size=8, d_model=16,
            n_heads=4, n_layers=1, dropout=0.0
        )
        model = DreamTransformer(config)
        tokens = torch.randint(0, config.vocab_size, (2, 8))
        logits, loss = model(tokens, tokens)
        self.assertEqual(tuple(logits.shape), (2, 8, 512))
        self.assertTrue(torch.isfinite(loss).item())

    def test_extended_vocabulary_generation_can_emit_bpe_tokens(self):
        config = DreamConfig(
            vocab_size=300, block_size=8, d_model=16,
            n_heads=4, n_layers=1, dropout=0.0
        )
        model = DreamTransformer(config).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.final_norm.bias.fill_(1.0)
            model.token_embedding.weight[270].fill_(1.0)
        prompt = torch.tensor([[65, 66]], dtype=torch.long)
        generated = model.generate(
            prompt, max_new_tokens=1, temperature=0,
            constrain_utf8=False
        )
        self.assertEqual(int(generated[0, -1]), 270)

    def test_vocabulary_below_byte_fallback_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 256"):
            DreamConfig(vocab_size=255)

    def test_invalid_dropout_is_rejected(self):
        with self.assertRaises(ValueError):
            DreamConfig(dropout=1.0)

    def test_invalid_generation_temperature_is_rejected(self):
        prompt = torch.tensor([[65, 66]], dtype=torch.long)
        for temperature in (float("nan"), float("inf"), True, False):
            with self.subTest(temperature=temperature):
                with self.assertRaises(ValueError):
                    self.model.generate(prompt, temperature=temperature)

    def test_generation_integer_arguments_reject_booleans(self):
        prompt = torch.tensor([[65, 66]], dtype=torch.long)
        for kwargs in ({"max_new_tokens": True}, {"top_k": False}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self.model.generate(prompt, **kwargs)

    def test_generation_validates_prompt_even_without_new_tokens(self):
        with self.assertRaises(TypeError):
            self.model.generate(torch.tensor([[65.0]]), max_new_tokens=0)
        with self.assertRaisesRegex(ValueError, "outside the configured vocabulary"):
            self.model.generate(torch.tensor([[256]]), max_new_tokens=0)

    def test_out_of_range_input_tokens_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the configured vocabulary"):
            self.model(torch.tensor([[0, 256]], dtype=torch.long))

    def test_out_of_range_target_tokens_are_rejected(self):
        inputs = torch.tensor([[0, 1]], dtype=torch.long)
        targets = torch.tensor([[1, 256]], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "targets contain IDs"):
            self.model(inputs, targets)

    def test_invalid_target_shape_is_rejected(self):
        prompt = torch.tensor([[65, 66]], dtype=torch.long)
        targets = torch.tensor([[66]], dtype=torch.long)
        with self.assertRaises(ValueError):
            self.model(prompt, targets)


if __name__ == "__main__":
    unittest.main()
