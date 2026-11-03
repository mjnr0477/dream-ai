"""Tests for DREAM's native inference performance benchmark."""
import unittest

import torch

from benchmark_inference import benchmark, percentile
from dream_transformer import DreamByteTokenizer, DreamConfig, DreamTransformer


class InferenceBenchmarkTests(unittest.TestCase):
    def test_percentile_uses_nearest_rank(self):
        self.assertEqual(percentile([5, 1, 4, 2, 3], 0.95), 5)
        self.assertEqual(percentile([5, 1, 4, 2, 3], 0.50), 3)

    def setUp(self):
        self.config = DreamConfig(
            vocab_size=256, block_size=32, d_model=16,
            n_heads=4, n_layers=1, dropout=0.0
        )
        self.model = DreamTransformer(self.config).eval()
        self.tokenizer = DreamByteTokenizer()

    def test_benchmark_reports_measured_generation_metrics(self):
        report = benchmark(
            self.model, self.tokenizer, self.config,
            "USER: Hi\nASSISTANT:", "cpu",
            warmup=0, runs=2, max_new_tokens=3, temperature=0
        )
        self.assertEqual(report["schema"], "dream-inference-benchmark-v1")
        self.assertEqual(report["runs"], 2)
        self.assertEqual(report["generated_tokens"], 6)
        self.assertEqual(report["requested_new_tokens_per_run"], 3)
        self.assertGreater(report["mean_latency_seconds"], 0)
        self.assertGreater(report["mean_generated_tokens_per_second"], 0)
        self.assertEqual(report["parameters"], sum(p.numel() for p in self.model.parameters()))

    def test_rejects_invalid_run_counts_and_empty_prompt(self):
        for kwargs in (
            {"runs": 0},
            {"warmup": -1},
            {"max_new_tokens": 0},
            {"temperature": float("nan")},
        ):
            args = {
                "warmup": 0, "runs": 1, "max_new_tokens": 2,
                "temperature": 0.0
            }
            args.update(kwargs)
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    benchmark(
                        self.model, self.tokenizer, self.config,
                        "hello", "cpu", **args
                    )
        with self.assertRaises(ValueError):
            benchmark(
                self.model, self.tokenizer, self.config,
                " ", "cpu", warmup=0, runs=1, max_new_tokens=2
            )


if __name__ == "__main__":
    unittest.main()
