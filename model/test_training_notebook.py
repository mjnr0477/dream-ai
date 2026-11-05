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


if __name__ == "__main__":
    unittest.main()
