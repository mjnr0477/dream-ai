"""Validate the checked-in Colab workflow without running cloud notebook cells."""
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

    def test_instruction_tuning_is_really_optional(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        all_code = "\n".join(code)
        self.assertIn("sft_completed = False", all_code)
        self.assertIn("sft_completed = True", all_code)
        self.assertIn("if sft_completed:", all_code)

    def test_notebook_never_downloads_or_loads_external_model_weights(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        all_code = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertNotIn("from_pretrained(", all_code)
        self.assertNotIn("AutoModel", all_code)
        self.assertIn("model/train.py", all_code)
        self.assertIn("model/sft.py", all_code)


if __name__ == "__main__":
    unittest.main()
