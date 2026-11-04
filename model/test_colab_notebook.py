"""Keep the Colab training workflow aligned with native DREAM BPE support."""
import json
import unittest
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "training" / "dream_train_colab.ipynb"


class ColabNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    def test_notebook_structure(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertTrue(self.notebook["cells"])
        self.assertTrue(all(cell["cell_type"] in ("code", "markdown")
                            for cell in self.notebook["cells"]))

    def test_fresh_training_defaults_to_bpe_and_exposes_legacy_byte_mode(self):
        source = "".join(self.notebook["cells"][6]["source"])
        self.assertIn('TOKENIZER = "bpe"', source)
        self.assertIn('Set to "byte"', source)
        self.assertIn("'--tokenizer', TOKENIZER", source)
        self.assertIn("'--vocab-size', str(BPE_VOCAB_SIZE)", source)

    def test_checkpoint_names_and_evaluation_metrics_match_bpe_workflow(self):
        setup = "".join(self.notebook["cells"][1]["source"])
        evaluation = "".join(self.notebook["cells"][7]["source"])
        report = "".join(self.notebook["cells"][8]["source"])
        self.assertIn("dream-bpe.pt", setup)
        self.assertIn("dream-bpe-instruct.pt", setup)
        self.assertIn("UTF-8 byte", evaluation)
        self.assertIn("dream-bpe.json", report)

    def test_colab_audits_language_balance_and_bundles_the_report(self):
        corpus = "".join(self.notebook["cells"][4]["source"])
        bundle = "".join(self.notebook["cells"][14]["source"])
        self.assertIn("model/audit_language_balance.py", corpus)
        self.assertIn("data/approved-records.jsonl", corpus)
        self.assertIn("language-balance.json", bundle)
        self.assertIn("not an approval", corpus)

    def test_colab_estimates_model_budget_for_selected_tokenizer(self):
        training = "".join(self.notebook["cells"][6]["source"])
        bundle = "".join(self.notebook["cells"][14]["source"])
        self.assertIn("model/estimate_model.py", training)
        self.assertIn("'--vocab-size', str(MODEL_VOCAB_SIZE)", training)
        self.assertIn("model-budget.json", bundle)


if __name__ == "__main__":
    unittest.main()
