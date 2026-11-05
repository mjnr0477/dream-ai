import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLAB_NOTEBOOK = ROOT / "training" / "dream_train_colab.ipynb"
KAGGLE_NOTEBOOK = ROOT / "training" / "dream_train_kaggle.ipynb"


def load_notebook(path):
    return json.loads(path.read_text(encoding="utf-8"))


def find_cell(notebook, fragment, cell_type="code"):
    for cell in notebook["cells"]:
        if cell.get("cell_type") != cell_type:
            continue
        source = "".join(cell.get("source", []))
        if fragment in source:
            return source
    raise AssertionError(f"Could not find {cell_type} notebook cell containing: {fragment}")


class ColabNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.colab = load_notebook(COLAB_NOTEBOOK)
        cls.kaggle = load_notebook(KAGGLE_NOTEBOOK)

    def colab_cell(self, fragment, cell_type="code"):
        return find_cell(self.colab, fragment, cell_type)

    def kaggle_cell(self, fragment, cell_type="code"):
        return find_cell(self.kaggle, fragment, cell_type)

    def test_colab_imports_json_before_corpus_audit(self):
        setup = self.colab_cell("USE_GOOGLE_DRIVE = True")
        corpus = self.colab_cell("model/audit_corpus.py")
        self.assertIn("import json", setup)
        self.assertIn("json.loads", corpus)

    def test_colab_pretraining_audit_fails_closed_before_corpus_preparation(self):
        corpus = self.colab_cell("model/audit_corpus.py")
        self.assertIn("check=True", corpus)
        self.assertIn("possible_pii_findings", corpus)
        self.assertIn("unreviewed_records", corpus)
        self.assertIn("raise RuntimeError", corpus)
        self.assertLess(corpus.index("model/audit_corpus.py"), corpus.index("model/prepare_corpus.py"))

    def test_colab_fresh_training_defaults_to_multilingual_bpe(self):
        training = self.colab_cell('TOKENIZER = "bpe"')
        self.assertIn("BPE_VOCAB_SIZE = 1024", training)
        self.assertIn("TRAIN_STEPS = 2000", training)
        self.assertIn("'--tokenizer', TOKENIZER", training)
        self.assertIn("'--vocab-size', str(BPE_VOCAB_SIZE)", training)

    def test_colab_checkpoint_names_and_evaluation_metrics_are_explicit(self):
        setup = self.colab_cell("PRETRAIN_CHECKPOINT")
        evaluation = self.colab_cell("eval-results/dream-bpe.json")
        self.assertIn("dream-bpe.pt", setup)
        self.assertIn("dream-bpe-instruct.pt", setup)
        self.assertIn("bits_per_byte", evaluation)
        self.assertIn("perplexity_per_byte", evaluation)
        self.assertIn("sample_generation", evaluation)

    def test_colab_language_balance_and_artifact_bundle_are_preserved(self):
        corpus = self.colab_cell("audit_language_balance.py")
        bundle = self.colab_cell("artifact_entries +=")
        self.assertIn("--max-share", corpus)
        self.assertIn("language-balance.json", bundle)
        self.assertIn("SHA256SUMS.txt", bundle)
        self.assertIn("model-budget.json", bundle)

    def test_colab_privacy_provenance_audit_precedes_preparation(self):
        corpus = self.colab_cell("model/audit_corpus.py")
        self.assertIn("model/prepare_corpus.py", corpus)
        self.assertIn("possible_pii_findings", corpus)
        self.assertIn("unreviewed_records", corpus)
        self.assertIn("check=True", corpus)

    def test_colab_audits_both_sft_splits_before_finetuning(self):
        sft = self.colab_cell("sft-{split_name}-corpus-audit.json")
        self.assertIn("for split_name in ('train', 'validation')", sft)
        self.assertIn("model/audit_corpus.py", sft)
        self.assertIn("raise RuntimeError", sft)
        self.assertLess(sft.index("model/audit_corpus.py"), sft.index("model/sft.py"))

    def test_sft_evaluation_removes_stale_report_and_uses_instruction_checkpoint(self):
        evaluation = self.colab_cell("dream-bpe-instruct.json")
        self.assertIn("unlink(missing_ok=True)", evaluation)
        self.assertIn("SFT_CHECKPOINT", evaluation)
        self.assertIn("model/evaluate.py", evaluation)

    def test_sft_checkpoint_has_technical_release_gate_before_runtime_connection(self):
        evaluation = self.colab_cell("model/validate_release.py")
        runtime = self.colab_cell("cloudflared")
        bundle = self.colab_cell("artifact_entries +=")
        self.assertIn("technical_preflight_passed", evaluation)
        self.assertIn("production_release_approved", evaluation)
        self.assertIn("RELEASE_GATE_PATH", runtime)
        self.assertIn("dream-bpe-instruct-release-gate.json", bundle)

    def test_colab_has_authenticated_native_runtime_bridge(self):
        runtime = self.colab_cell("cloudflared")
        self.assertIn("DREAM_RUNTIME_TOKEN", runtime)
        self.assertIn("DREAM_MODEL_URL", runtime)
        self.assertIn("DREAM_MODEL_TOKEN", runtime)
        self.assertIn("TUNNEL_URL + \"/v1\"", runtime)
        self.assertIn("checkpoint_sha256", runtime)

    def test_colab_uses_explicit_render_secret_and_deploy_confirmation(self):
        runtime = self.colab_cell('userdata.get("RENDER_API_KEY")')
        self.assertIn("RENDER_API_KEY", runtime)
        self.assertIn("RENDER_SERVICE_ID", runtime)
        self.assertIn('approval.strip() == "DEPLOY"', runtime)
        self.assertIn("update_render_model_env", runtime)

    def test_colab_estimates_model_budget_before_training(self):
        training = self.colab_cell("model/estimate_model.py")
        bundle = self.colab_cell("artifact_entries +=")
        self.assertIn("parameter_count", training)
        self.assertIn("fp32_adam_training_floor_bytes", training)
        self.assertIn("model-budget.json", bundle)

    def test_colab_prepares_oasst_candidates_without_approving_them(self):
        guidance = self.colab_cell("OASST1 pretraining and instruction candidates", "markdown")
        preparation = self.colab_cell("data/import_oasst1_candidates.py")
        self.assertIn("rights, privacy, quality and training-approval flags false", guidance)
        self.assertIn("data/import_oasst1_sft_candidates.py", preparation)
        self.assertIn("human review", preparation)

    def test_kaggle_audits_pretraining_corpus_before_preparation(self):
        corpus = self.kaggle_cell("model/audit_corpus.py")
        self.assertIn("check=True", corpus)
        self.assertIn("possible_pii_findings", corpus)
        self.assertIn("unreviewed_records", corpus)
        self.assertLess(corpus.index("model/audit_corpus.py"), corpus.index("model/prepare_corpus.py"))

    def test_kaggle_audits_both_sft_splits(self):
        sft = self.kaggle_cell("sft-{split_name}-corpus-audit.json")
        self.assertIn("model/audit_corpus.py", sft)
        self.assertIn("for split_name in ('train', 'validation')", sft)
        self.assertIn("raise RuntimeError", sft)


if __name__ == "__main__":
    unittest.main()
