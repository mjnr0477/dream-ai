"""Structural safety checks for the provider-independent DREAM GPU notebook."""
import ast
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "training" / "dream_native_persistent_notebook.ipynb"
WORKER = ROOT / "dream-host" / "notebook_worker.py"
INSTALLER = ROOT / "scripts" / "install-dream-notebook-worker.sh"
UNIT = ROOT / "deploy" / "systemd" / "dream-notebook-worker.service"
DOC = ROOT / "docs" / "PERMANENT_GPU_WORKER.md"


class PersistentNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code_cells = [
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        cls.code = "\n".join(cls.code_cells)
        cls.markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )

    def test_notebook_json_and_every_code_cell_are_valid(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertTrue(self.notebook["cells"])
        for index, source in enumerate(self.code_cells):
            with self.subTest(cell=index):
                ast.parse(source, filename=f"{NOTEBOOK.name}:cell-{index}")

    def test_native_training_notebook_has_no_kaggle_or_hugging_face_dependency(self):
        source = (self.code + "\n" + self.markdown).casefold()
        for forbidden in (
            "kaggle",
            "/kaggle/",
            "kaggle_secrets",
            "hf_token",
            "huggingface.co",
            "from_pretrained(",
            "autotokenizer",
            "automodelforcausallm",
            "peft",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_native_model_trains_from_its_own_code_and_resumes_from_r2(self):
        self.assertIn("model/train.py", self.code)
        self.assertIn("model/sft.py", self.code)
        self.assertIn("dream-native-pretrain-checkpoint.pt", self.code)
        self.assertIn("dream-native-sft-checkpoint.pt", self.code)
        self.assertIn("r2_restore_latest(", self.code)
        self.assertIn("r2_upload(", self.code)
        self.assertIn("hashlib.sha256", self.code)
        self.assertIn("R2 artifact checksum or size mismatch", self.code)
        self.assertIn("R2 upload verification failed", self.code)

    def test_data_checkpoints_and_reports_use_operator_persistent_paths(self):
        for variable in (
            "DREAM_PREPARED_DIR",
            "DREAM_CHECKPOINT_DIR",
            "DREAM_WORK_DIR",
            "DREAM_APPROVED_RECORDS_PATH",
            "DREAM_SFT_TRAIN_PATH",
            "DREAM_SFT_VALIDATION_PATH",
        ):
            with self.subTest(variable=variable):
                self.assertIn(variable, self.code)

    def test_training_and_sft_retain_existing_data_quality_gates(self):
        self.assertIn("model/audit_corpus.py", self.code)
        self.assertIn("model/prepare_corpus.py", self.code)
        self.assertIn("possible_pii_findings", self.code)
        self.assertIn("unreviewed_records", self.code)
        self.assertIn("data/verify_dream_ai_sft_audits.py", self.code)
        self.assertIn("--pretraining-test-data", self.code)

    def test_host_worker_passes_only_allowlisted_native_configuration(self):
        source = WORKER.read_text(encoding="utf-8")
        for variable in (
            "DREAM_REPO_DIR",
            "DREAM_REPO_BRANCH",
            "DREAM_CHECKPOINT_DIR",
            "DREAM_PREPARED_DIR",
            "DREAM_WORK_DIR",
            "DREAM_APPROVED_RECORDS_PATH",
            "DREAM_SFT_TRAIN_PATH",
            "DREAM_SFT_VALIDATION_PATH",
            "DREAM_NATIVE_TRAIN_STEPS",
            "DREAM_NATIVE_TRAIN_SECONDS",
            "DREAM_NATIVE_SFT_STEPS",
            "DREAM_NATIVE_SFT_SECONDS",
        ):
            with self.subTest(variable=variable):
                self.assertIn('"' + variable + '"', source)
        self.assertIn('environment["HOME"] = str(workspace)', source)

    def test_installer_requires_a_real_nvidia_gpu_and_never_falls_back_to_cpu(self):
        source = INSTALLER.read_text(encoding="utf-8")
        for required in (
            "nvidia-smi",
            "torch.cuda.is_available()",
            "CUDA-enabled PyTorch",
            "DREAM_HOST_NOTEBOOK_WORKER_TOKEN",
            "DREAM_HOST_ARTIFACT_TOKEN",
            "chmod 0600 /etc/dream-ai/notebook-worker.env",
            "systemctl enable --now dream-notebook-worker.service",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)
        self.assertNotIn("rm -rf", source)

    def test_installer_and_systemd_use_restartable_owner_scoped_service(self):
        service = UNIT.read_text(encoding="utf-8")
        self.assertIn("User=dreamai", service)
        self.assertIn("Restart=always", service)
        self.assertIn("EnvironmentFile=/etc/dream-ai/notebook-worker.env", service)
        self.assertIn("ReadWritePaths=/opt/dream-ai /var/lib/dream-ai", service)

    def test_documentation_does_not_claim_gpu_or_model_migration_is_already_live(self):
        source = DOC.read_text(encoding="utf-8").casefold()
        self.assertIn("r2", source)
        self.assertIn("owner-controlled", source)
        self.assertIn("do not change the live model settings yet", source)
        self.assertIn("no free, always-on gpu capacity", source)


if __name__ == "__main__":
    unittest.main()
