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

    def test_non_byte_vocabulary_is_rejected(self):
        with self.assertRaises(ValueError):
            DreamConfig(vocab_size=512)

    def test_invalid_dropout_is_rejected(self):
        with self.assertRaises(ValueError):
            DreamConfig(dropout=1.0)

    def test_invalid_generation_temperature_is_rejected(self):
        prompt = torch.tensor([[65, 66]], dtype=torch.long)
        with self.assertRaises(ValueError):
            self.model.generate(prompt, temperature=float("nan"))

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
