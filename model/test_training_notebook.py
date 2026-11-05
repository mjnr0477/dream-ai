"""Guard DREAM's Colab training notebook against losing checkpoints on runtime reset."""
import json
import unittest
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "training"
    / "dream_train_colab.ipynb"
)


class ColabCheckpointPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.document["cells"]
            if cell.get("cell_type") == "code"
        )

    def test_google_drive_persistence_is_enabled_by_default(self):
        self.assertIn("USE_GOOGLE_DRIVE = True", self.code)

    def test_drive_is_mounted_before_checkpoint_directory_is_selected(self):
        mount = self.code.index("drive.mount('/content/drive')")
        checkpoint_dir = self.code.index(
            "CHECKPOINT_DIR = Path('/content/drive/MyDrive/DREAM AI/checkpoints')"
        )
        self.assertLess(mount, checkpoint_dir)

    def test_pretraining_and_sft_checkpoints_share_persistent_directory(self):
        self.assertIn("PRETRAIN_CHECKPOINT = CHECKPOINT_DIR / 'dream-bpe.pt'", self.code)
        self.assertIn("SFT_CHECKPOINT = CHECKPOINT_DIR / 'dream-bpe-instruct.pt'", self.code)
        self.assertIn("train_command += ['--resume', str(PRETRAIN_CHECKPOINT)]", self.code)
        self.assertIn("sft_command += ['--resume', str(SFT_CHECKPOINT)]", self.code)

    def test_manual_candidate_review_is_optional_persistent_and_never_auto_approves(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        all_code = "\\n".join(
            "".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        markdown = "\\n".join(
            "".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        self.assertIn("from data.review_candidates import review_file", all_code)
        self.assertIn("candidate-review-decisions.jsonl", all_code)
        self.assertIn("CHECKPOINT_DIR.parent / 'review'", all_code)
        self.assertIn("limit=25", all_code)
        self.assertIn("never approves automatically", markdown)
        self.assertIn("if not (approved_path.is_file() and approved_path.stat().st_size > 0)", all_code)

    def test_manual_sft_review_creates_source_group_disjoint_splits(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        all_code = "\\n".join(
            "".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        markdown = "\\n".join(
            "".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        self.assertIn("from data.split_sft import split_sft", all_code)
        self.assertIn("approved-sft-records.jsonl", all_code)
        self.assertIn("sft-review-decisions.jsonl", all_code)
        self.assertIn("sft-split-manifest.json", all_code)
        self.assertIn("validation_fraction=0.2", all_code)
        self.assertIn("source_group", markdown)
        self.assertIn("local_sft_available", all_code)
        self.assertIn("data/sft-split-manifest.json", all_code)


if __name__ == "__main__":
    unittest.main()
