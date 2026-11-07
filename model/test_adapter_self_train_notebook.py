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

    def test_adapter_training_uses_a_four_minute_resumable_budget(self):
        self.assertIn("MAX_TRAINING_SECONDS = 240", self.joined)
        self.assertIn("training_started_at = time.monotonic()", self.joined)
        self.assertIn("if elapsed >= MAX_TRAINING_SECONDS", self.joined)
        self.assertIn("optimizer.load_state_dict(optimizer_state)", self.joined)
        self.assertIn("weights_only=True", self.joined)
        self.assertIn('"optimizer-state.pt"', self.joined)
        self.assertIn('"rng-state.pt"', self.joined)
        self.assertIn("torch.set_rng_state", self.joined)
        self.assertIn("torch.cuda.set_rng_state_all", self.joined)

    def test_resume_progress_is_bound_to_exact_dataset_gate_and_parent_adapter(self):
        for field in (
            '"dataset_sha256": manifest["dataset_sha256"]',
            '"audit_gate_sha256": audit_gate_sha256',
            '"parent_adapter_sha256": parent_adapter_sha256',
            'if progress_meta.get(key) != value',
            'training-progress/',
            'completed_updates',
        ):
            self.assertIn(field, self.joined)
        self.assertIn("api.upload_folder(", self.joined)

    def test_candidate_upload_is_separate_from_the_live_adapter_root(self):
        self.assertIn('candidate_path = "candidates/" + run_id', self.joined)
        self.assertIn("api.upload_folder(", self.joined)
        self.assertIn("status", self.joined)
        self.assertIn('"candidate-not-active"', self.joined)
        self.assertNotIn('path_in_repo="."', self.joined)
        upload_lines = self.joined.splitlines()
        progress_upload_index = next(
            i for i, line in enumerate(upload_lines)
            if line.strip().startswith("progress_uploaded = api.upload_folder(")
        )
        candidate_upload_index = next(
            i for i, line in enumerate(upload_lines)
            if line.strip().startswith("uploaded = api.upload_folder(")
        )
        self.assertLess(progress_upload_index, candidate_upload_index)


if __name__ == "__main__":
    unittest.main()
