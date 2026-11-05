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

    def test_colab_imports_json_before_the_first_corpus_audit(self):
        setup = "".join(self.notebook["cells"][1]["source"])
        corpus = "".join(self.notebook["cells"][6]["source"])
        self.assertIn("import os, subprocess, sys, torch, json", setup)
        self.assertLess(setup.index("import os"), corpus.index("json.loads"))


    def test_pretraining_and_sft_audits_fail_closed_without_stale_reports(self):
        corpus = "".join(self.notebook["cells"][6]["source"])
        sft = "".join(self.notebook["cells"][13]["source"])
        self.assertIn("Path('eval-results/corpus-audit.json').unlink(missing_ok=True)", corpus)
        self.assertIn("], capture_output=True, text=True, check=True)", corpus)
        self.assertIn("Path(audit_path).unlink(missing_ok=True)", sft)
        self.assertIn("], check=True, capture_output=True, text=True)", sft)
        self.assertNotIn("check=False", sft)

    def test_fresh_training_defaults_to_bpe_and_exposes_legacy_byte_mode(self):
        source = "".join(self.notebook["cells"][8]["source"])
        self.assertIn('TOKENIZER = "bpe"', source)
        self.assertIn('Set to "byte"', source)
        self.assertIn("'--tokenizer', TOKENIZER", source)
        self.assertIn("'--vocab-size', str(BPE_VOCAB_SIZE)", source)

    def test_checkpoint_names_and_evaluation_metrics_match_bpe_workflow(self):
        setup = "".join(self.notebook["cells"][1]["source"])
        evaluation = "".join(self.notebook["cells"][9]["source"])
        report = "".join(self.notebook["cells"][10]["source"])
        self.assertIn("dream-bpe.pt", setup)
        self.assertIn("dream-bpe-instruct.pt", setup)
        self.assertIn("UTF-8 byte", evaluation)
        self.assertIn("dream-bpe.json", report)

    def test_colab_audits_language_balance_and_bundles_the_report(self):
        corpus = "".join(self.notebook["cells"][6]["source"])
        bundle = "".join(self.notebook["cells"][16]["source"])
        self.assertIn("model/audit_language_balance.py", corpus)
        self.assertIn("data/approved-records.jsonl", corpus)
        self.assertIn("language-balance.json", bundle)
        self.assertIn("not an approval", corpus)

    def test_colab_runs_privacy_and_provenance_audit_before_preparation(self):
        corpus = "".join(self.notebook["cells"][6]["source"])
        bundle = "".join(self.notebook["cells"][16]["source"])
        self.assertLess(
            corpus.index("model/audit_corpus.py"),
            corpus.index("model/prepare_corpus.py")
        )
        self.assertIn("possible_pii_findings", corpus)
        self.assertIn("human rights, privacy and quality review", corpus)

    def test_colab_audits_both_sft_splits_before_instruction_tuning(self):
        sft = "".join(self.notebook["cells"][13]["source"])
        bundle = "".join(self.notebook["cells"][16]["source"])
        self.assertIn("model/audit_corpus.py", sft)
        self.assertIn("sft-{split_name}-corpus-audit.json", sft)
        self.assertLess(sft.index("model/audit_corpus.py"), sft.index("model/sft.py"))
        self.assertIn("sft-train-corpus-audit.json", bundle)
        self.assertIn("sft-validation-corpus-audit.json", bundle)
        self.assertIn("corpus-audit.json", bundle)

    def test_sft_evaluation_removes_stale_report_before_running(self):
        evaluation = "".join(self.notebook["cells"][14]["source"])
        self.assertLess(
            evaluation.index("unlink(missing_ok=True)"),
            evaluation.index("model/evaluate.py")
        )

    def test_sft_checkpoint_must_pass_technical_release_gate_before_runtime(self):
        evaluation = "".join(self.notebook["cells"][14]["source"])
        bundle = "".join(self.notebook["cells"][16]["source"])
        runtime = "".join(self.notebook["cells"][18]["source"])
        self.assertLess(
            evaluation.index("model/audit_checkpoint.py"),
            evaluation.index("model/validate_release.py")
        )
        self.assertIn("dream-bpe-instruct-audit.json", bundle)
        self.assertIn("dream-bpe-instruct-release-gate.json", bundle)
        self.assertIn("production_release_approved", evaluation)
        self.assertIn("RELEASE_GATE_PATH", runtime)
        self.assertIn("RELEASE_GATE.get(\"checkpoint_sha256\")", runtime)
        self.assertIn("not production approval", evaluation.lower())

    def test_colab_has_authenticated_temporary_native_runtime_bridge(self):
        markdown = "".join(self.notebook["cells"][17]["source"])
        runtime = "".join(self.notebook["cells"][18]["source"])
        import ast
        ast.parse(runtime)
        self.assertIn("not sft_completed", runtime)
        self.assertIn("dream-bpe-instruct.json", runtime)
        self.assertIn("DREAM_RUNTIME_TOKEN", runtime)
        self.assertIn("previous_process.terminate()", runtime)
        self.assertIn("cloudflared", runtime)
        self.assertIn(r"\.trycloudflare\.com", runtime)
        self.assertIn("DREAM_MODEL_URL=", runtime)
        self.assertIn("DREAM_MODEL_NAME=", runtime)
        self.assertIn("DREAM_MODEL_TOKEN=", runtime)
        self.assertIn("temporary", markdown.lower())
        self.assertIn("Render service", markdown)


    def test_colab_render_autoconnect_uses_secret_and_requires_explicit_confirmation(self):
        markdown = "".join(self.notebook["cells"][17]["source"])
        runtime = "".join(self.notebook["cells"][18]["source"])
        self.assertIn('userdata.get("RENDER_API_KEY")', runtime)
        self.assertIn('approval.strip() == "DEPLOY"', runtime)
        self.assertIn("update_render_model_env(", runtime)
        self.assertIn("confirm=True", runtime)
        self.assertIn("RENDER_SERVICE_ID = ", runtime)
        self.assertIn("preserving other variables", markdown)
        self.assertIn("RENDER_API_KEY", markdown)
        self.assertIn("not printed", runtime)
        self.assertIn("DREAM_MODEL_TOKEN=" + " + RUNTIME_TOKEN", runtime)
\n    def test_colab_estimates_model_budget_for_selected_tokenizer(self):
        training = "".join(self.notebook["cells"][8]["source"])
        bundle = "".join(self.notebook["cells"][16]["source"])
        self.assertIn("model/estimate_model.py", training)
        self.assertIn("'--vocab-size', str(MODEL_VOCAB_SIZE)", training)
        self.assertIn("model-budget.json", bundle)


    def test_colab_prepares_oasst1_pretraining_and_sft_candidates_without_approval(self):
        guidance = "".join(self.notebook["cells"][3]["source"])
        preparation = "".join(self.notebook["cells"][4]["source"])
        self.assertIn("data/import_oasst1_candidates.py", preparation)
        self.assertIn("data/import_oasst1_sft_candidates.py", preparation)
        self.assertIn("oasst1-sft-candidates.jsonl", preparation)
        self.assertIn("--max-records', '5000", preparation)
        self.assertIn("approval flags false", guidance.lower())
        self.assertIn("human review", guidance.lower())



    def test_kaggle_pretraining_and_sft_audits_fail_closed(self):
        kaggle_path = NOTEBOOK.parents[1] / "training" / "dream_train_kaggle.ipynb"
        kaggle = json.loads(kaggle_path.read_text(encoding="utf-8"))
        pretraining = "".join(kaggle["cells"][4]["source"])
        sft = "".join(kaggle["cells"][11]["source"])
        self.assertIn("corpus-audit.json').unlink(missing_ok=True)", pretraining)
        self.assertIn("language-balance.json').unlink(missing_ok=True)", pretraining)
        self.assertIn("Path(audit_path).unlink(missing_ok=True)", sft)
        self.assertIn("check=True, capture_output=True, text=True", sft)
        self.assertNotIn("check=False", sft)



if __name__ == "__main__":
    unittest.main()
