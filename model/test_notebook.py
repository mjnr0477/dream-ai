"""Validate the checked-in DREAM Colab workflow without executing cloud cells."""
import ast
import json
import unittest
from pathlib import Path

NOTEBOOK = Path(__file__).parents[1] / "training" / "dream_train_colab.ipynb"


class DreamNotebookTests(unittest.TestCase):
    def test_all_code_cells_parse_as_python(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertGreater(len(code_cells), 5)
        for index, cell in enumerate(code_cells):
            with self.subTest(cell=index):
                ast.parse("".join(cell["source"]), filename=f"colab-cell-{index}")

    def test_sft_is_optional_and_downloads_only_if_completed(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        all_code = "\n".join(code)
        self.assertIn("sft_completed = False", all_code)
        self.assertIn("sft_completed = True", all_code)
        self.assertIn("sft_available =", all_code)
        self.assertIn("if sft_available:", all_code)
        self.assertIn("if sft_completed:", all_code)
        self.assertIn("Instruction tuning skipped", all_code)
        self.assertNotIn("raise RuntimeError('Upload both sft-train.jsonl", all_code)

    def test_sft_uses_pretraining_manifest_for_test_isolation(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        all_code = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn("'--pretraining-manifest'", all_code)
        self.assertIn("'data/prepared/manifest.json'", all_code)
        self.assertIn("'--pretraining-test-data'", all_code)
        self.assertIn("'data/prepared/test.txt'", all_code)

    def test_notebook_trains_only_native_random_initialized_dream(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        all_code = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn("model/train.py", all_code)
        self.assertIn("model/sft.py", all_code)
        self.assertNotIn("from_pretrained(", all_code)
        self.assertNotIn("AutoModel", all_code)


if __name__ == "__main__":
    unittest.main()
