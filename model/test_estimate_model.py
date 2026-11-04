"""Tests for DREAM model parameter and memory budget estimates."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dream_transformer import DreamConfig, DreamTransformer
from estimate_model import estimate_model

SCRIPT = Path(__file__).with_name("estimate_model.py")


class ModelBudgetTests(unittest.TestCase):
    def test_formula_matches_actual_tied_weight_model_parameters(self):
        config = DreamConfig(
            block_size=32, d_model=16, n_heads=4, n_layers=2, dropout=0.0
        )
        report = estimate_model(config)
        actual = sum(parameter.numel() for parameter in DreamTransformer(config).parameters())
        self.assertEqual(report["parameter_count"], actual)

    def test_bpe_vocabulary_changes_exact_parameter_estimate(self):
        config = DreamConfig(
            vocab_size=1024, block_size=32, d_model=16,
            n_heads=4, n_layers=2, dropout=0.0
        )
        report = estimate_model(config)
        actual = sum(parameter.numel() for parameter in DreamTransformer(config).parameters())
        self.assertEqual(report["parameter_count"], actual)
        self.assertEqual(report["config"]["vocab_size"], 1024)

    def test_default_model_matches_documented_parameter_count(self):
        report = estimate_model(DreamConfig())
        self.assertEqual(report["parameter_count"], 4_864_000)

    def test_memory_estimates_are_explicit_lower_bounds(self):
        report = estimate_model(DreamConfig(
            block_size=32, d_model=16, n_heads=4, n_layers=1, dropout=0.0
        ))
        count = report["parameter_count"]
        memory = report["parameter_memory"]
        self.assertEqual(memory["fp32_inference_bytes"], count * 4)
        self.assertEqual(memory["fp16_inference_bytes"], count * 2)
        self.assertEqual(memory["fp32_adam_training_floor_bytes"], count * 16)
        self.assertIn("activations and saved tensors for backpropagation",
                      report["excluded_memory"])
        self.assertIn("lower bounds", report["warning"])

    def test_cli_writes_json_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reports" / "budget.json"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--vocab-size", "320", "--block-size", "32",
                 "--d-model", "16", "--heads", "4", "--layers", "1",
                 "--dropout", "0", "--out", str(output)],
                capture_output=True, text=True, timeout=30
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["parameter_count"], 8_880)
            self.assertEqual(report["config"]["vocab_size"], 320)

    def test_cli_rejects_invalid_head_dimensions(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--d-model", "15", "--heads", "4"],
            capture_output=True, text=True, timeout=30
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("divisible", result.stderr)


if __name__ == "__main__":
    unittest.main()
