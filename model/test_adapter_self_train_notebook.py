import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "training" / "dream_ai_adapter_self_train_kaggle.ipynb"


class DreamAIAdapterSelfTrainingNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = [
            "".join(cell.get("source", []))
            for cell in cls.notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        ]
        cls.joined = "\n".join(cls.code)

    def test_notebook_is_valid_and_has_code_cells(self):
        self.assertEqual(self.notebook.get("nbformat"), 4)
        self.assertGreaterEqual(len(self.code), 7)
        for index, source in enumerate(self.code):
            compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec")

    def test_training_requires_audited_v3_human_approved_examples(self):
        for requirement in (
            "dream-sft-audit-gate-v1",
            "audit_gate.get(\"passed\") is not True",
            "dream-learning-opt-in-v3",
            "rights_confirmed",
            "privacy_reviewed",
            "quality_reviewed",
            "approval_granted",
            "MIN_TOTAL_RECORDS = 32",
            "MIN_SOURCE_GROUPS = 8",
        ):
            self.assertIn(requirement, self.joined)

    def test_before_after_evaluation_surrounds_training(self):
        baseline_index = next(i for i, source in enumerate(self.code) if "baseline_completions" in source)
        training_index = next(i for i, source in enumerate(self.code) if "optimizer = torch.optim.AdamW" in source)
        candidate_index = next(i for i, source in enumerate(self.code) if "candidate_completions" in source)
        self.assertLess(baseline_index, training_index)
        self.assertLess(training_index, candidate_index)
        self.assertIn("candidate_validation_loss > baseline_validation_loss * 1.02", self.joined)

    def test_candidate_upload_is_separate_from_the_live_adapter_root(self):
        self.assertIn('candidate_path = "candidates/" + run_id', self.joined)
        self.assertIn("api.upload_folder(", self.joined)
        self.assertIn("status", self.joined)
        self.assertIn('"candidate-not-active"', self.joined)
        self.assertNotIn('path_in_repo="."', self.joined)


if __name__ == "__main__":
    unittest.main()
