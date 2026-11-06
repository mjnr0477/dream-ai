"""Tests for DREAM's assistant-response-only supervised fine-tuning."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from dream_bpe import DreamBPETokenizer
from dream_transformer import DreamConfig, DreamTransformer, DreamByteTokenizer
from sft import load_examples, record_signatures, verify_sft_audit_gate

SCRIPT = Path(__file__).with_name("sft.py")


def source_group_fingerprint(value):
    return hashlib.sha256(f"dream-source-group-id-v1\0{value}".encode("utf-8")).hexdigest()
EVALUATOR = Path(__file__).with_name("evaluate.py")

SUBPROCESS_RUN = subprocess.run


def run_with_sft_gate(command, *args, **kwargs):
    command = list(command)
    if len(command) > 1 and command[1] == str(SCRIPT):
        train_path = Path(command[command.index("--train-data") + 1])
        validation_path = Path(command[command.index("--validation-data") + 1])
        train_bytes = train_path.read_bytes()
        validation_bytes = validation_path.read_bytes()

        def count_records(raw):
            return sum(1 for line in raw.splitlines() if line.strip())

        gate_path = train_path.with_name("test-sft-audit-gate.json")
        gate = {
            "schema": "dream-sft-audit-gate-v1",
            "passed": True,
            "dataset_sha256": hashlib.sha256(train_bytes + validation_bytes).hexdigest(),
            "splits": {
                "train": {
                    "records": count_records(train_bytes),
                    "sha256": hashlib.sha256(train_bytes).hexdigest(),
                    "possible_pii_findings": 0,
                    "unreviewed_records": 0,
                    "error_count": 0,
                    "parse_errors": 0,
                    "finding_counts": {},
                    "errors": [],
                },
                "validation": {
                    "records": count_records(validation_bytes),
                    "sha256": hashlib.sha256(validation_bytes).hexdigest(),
                    "possible_pii_findings": 0,
                    "unreviewed_records": 0,
                    "error_count": 0,
                    "parse_errors": 0,
                    "finding_counts": {},
                    "errors": [],
                },
            },
        }
        gate_path.write_text(json.dumps(gate), encoding="utf-8")
        if "--sft-audit-gate" not in command:
            command.extend(["--sft-audit-gate", str(gate_path)])
    return SUBPROCESS_RUN(command, *args, **kwargs)



class DreamSFTTests(unittest.TestCase):
    def test_sft_requires_the_privacy_audit_gate_argument(self):
        result = SUBPROCESS_RUN([
            sys.executable, str(SCRIPT),
            "--checkpoint", str(self.base),
            "--train-data", str(self.root / "missing-train.jsonl"),
            "--validation-data", str(self.root / "missing-validation.jsonl"),
            "--pretraining-manifest", str(self.manifest),
            "--pretraining-test-data", str(self.test_data),
        ], capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--sft-audit-gate", result.stderr)

    def test_sft_rejects_audit_gate_bound_to_different_train_bytes(self):
        train_path = self.root / "audit-train.jsonl"
        validation_path = self.root / "audit-validation.jsonl"
        self.write_records(train_path, "audit-train")
        self.write_records(validation_path, "audit-validation")
        gate_path = self.root / "stale-audit-gate.json"
        gate_path.write_text(json.dumps({
            "schema": "dream-sft-audit-gate-v1",
            "passed": True,
            "dataset_sha256": "a" * 64,
            "splits": {
                "train": {
                    "records": 3, "sha256": "b" * 64,
                    "possible_pii_findings": 0, "unreviewed_records": 0,
                    "finding_counts": {}
                },
                "validation": {
                    "records": 3, "sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
                    "possible_pii_findings": 0, "unreviewed_records": 0,
                    "finding_counts": {}
                }
            }
        }), encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "train file does not match"):
            verify_sft_audit_gate(gate_path, train_path, validation_path)

    def test_sft_rejects_audit_gate_with_unresolved_findings(self):
        train_path = self.root / "clean-train.jsonl"
        validation_path = self.root / "clean-validation.jsonl"
        self.write_records(train_path, "clean-train")
        self.write_records(validation_path, "clean-validation")
        gate_path = self.root / "finding-audit-gate.json"
        gate = {
            "schema": "dream-sft-audit-gate-v1",
            "passed": True,
            "dataset_sha256": "a" * 64,
            "splits": {}
        }
        for name, path in (("train", train_path), ("validation", validation_path)):
            raw = path.read_bytes()
            gate["splits"][name] = {
                "records": sum(1 for line in raw.splitlines() if line.strip()),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "possible_pii_findings": 0,
                "unreviewed_records": 0,
                "finding_counts": {"possible_email": 1} if name == "validation" else {}
            }
        gate_path.write_text(json.dumps(gate), encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "validation audit contains unresolved findings"):
            verify_sft_audit_gate(gate_path, train_path, validation_path)


    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = DreamConfig(
            vocab_size=256, block_size=64, d_model=16,
            n_heads=4, n_layers=1, dropout=0.0
        )
        self.base = self.root / "pretrained.pt"
        model = DreamTransformer(self.config)
        torch.save({
            "model_state": model.state_dict(),
            "config": self.config.__dict__,
            "tokenizer": "utf8-byte-v1",
            "training": {"completed_steps": 2, "initialization": "random"}
        }, self.base)
        self.test_data = self.root / "pretraining-test.txt"
        self.test_data.write_text(
            "This is a separate held out corpus used only to test evaluation isolation. " * 4,
            encoding="utf-8"
        )
        test_hash = hashlib.sha256(self.test_data.read_bytes()).hexdigest()
        self.manifest = self.root / "pretraining-manifest.json"
        self.write_manifest(["pretraining-test-group"], test_hash)

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self, test_groups, test_hash=None):
        if test_hash is None:
            test_hash = hashlib.sha256(self.test_data.read_bytes()).hexdigest()
        self.manifest.write_text(json.dumps({
            "schema": "dream-corpus-v1",
            "splits": {
                "test": {
                    "source_group_ids": test_groups,
                    "sha256": test_hash
                }
            }
        }), encoding="utf-8")

    def write_records(self, path, source_prefix):
        rows = []
        for index in range(3):
            rows.append({
                "id": f"{source_prefix}-{index}",
                "source": f"Reviewed DREAM source {source_prefix}-{index}",
                "source_group": f"{source_prefix}-group-{index}",
                "license": "CC0-1.0",
                "source_url": f"https://example.org/sft/{source_prefix}/{index}",
                "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "rights_basis": "original-authorship",
                "source_date": "2026-01-15",
                "rights_confirmed": True,
                "privacy_reviewed": True,
                "quality_reviewed": True,
                "approval_granted": True,
                "language": "en",
                "messages": [
                    {"role": "user", "content": f"Question from {source_prefix} number {index}?"},
                    {"role": "assistant", "content": f"Answer from {source_prefix} number {index}."}
                ]
            })
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8"
        )

    def test_sft_rejects_missing_or_unapproved_training_approval(self):
        for value in (False, None):
            train_path = self.root / "unapproved-sft.jsonl"
            self.write_records(train_path, "unapproved")
            rows = [json.loads(line) for line in train_path.read_text().splitlines()]
            if value is None:
                del rows[0]["approval_granted"]
            else:
                rows[0]["approval_granted"] = value
            train_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.subTest(value=value):
                with self.assertRaises(SystemExit) as error:
                    load_examples(train_path, DreamByteTokenizer(), self.config.block_size)
                self.assertIn("Explicit SFT training approval not confirmed", str(error.exception))

    def test_sft_rejects_missing_or_unverified_provenance(self):
        for field, value, expected in (
            ("source_url", None, "Missing source_url"),
            ("source_url", "http://example.org/source", "absolute HTTPS URL"),
            ("license_url", "http://example.org/license", "absolute HTTPS URL"),
            ("rights_basis", "unverified", "Invalid rights_basis"),
        ):
            train_path = self.root / "bad-provenance.jsonl"
            self.write_records(train_path, "provenance")
            rows = [json.loads(line) for line in train_path.read_text().splitlines()]
            if value is None:
                del rows[0][field]
            else:
                rows[0][field] = value
            train_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8"
            )
            with self.subTest(field=field, value=value):
                with self.assertRaises(SystemExit) as error:
                    load_examples(train_path, DreamByteTokenizer(), self.config.block_size)
                self.assertIn(expected, str(error.exception))

    def test_masks_user_prompt_and_trains_assistant_answers(self):
        train_path = self.root / "train.jsonl"
        val_path = self.root / "validation.jsonl"
        self.write_records(train_path, "train")
        self.write_records(val_path, "validation")
        examples = load_examples(
            train_path, DreamByteTokenizer(), self.config.block_size
        )
        self.assertEqual(len(examples), 3)
        for input_ids, targets in examples:
            self.assertEqual(len(input_ids), len(targets))
            self.assertIn(-100, targets)
            self.assertTrue(any(target != -100 for target in targets))
            supervised = bytes(target for target in targets if target != -100)
            self.assertIn(b"Answer from train number", supervised)


    def test_bpe_sft_checkpoint_preserves_tokenizer_and_evaluates(self):
        train_path = self.root / "bpe-train.jsonl"
        val_path = self.root / "bpe-validation.jsonl"
        self.write_records(train_path, "train")
        self.write_records(val_path, "validation")
        corpus = train_path.read_text(encoding="utf-8") + val_path.read_text(encoding="utf-8")
        tokenizer = DreamBPETokenizer.train(corpus, vocab_size=300)
        config = DreamConfig(vocab_size=tokenizer.vocab_size, block_size=64,
                             d_model=16, n_heads=4, n_layers=1, dropout=0.0)
        model = DreamTransformer(config)
        bpe_base = self.root / "pretrained-bpe.pt"
        torch.save({"model_state": model.state_dict(), "config": config.__dict__,
                    "tokenizer": tokenizer.to_dict(),
                    "training": {"completed_steps": 2, "initialization": "random"}}, bpe_base)
        output = self.root / "dream-bpe-instruct.pt"
        result = run_with_sft_gate([
            sys.executable, str(SCRIPT), "--checkpoint", str(bpe_base),
            "--train-data", str(train_path), "--validation-data", str(val_path),
            "--pretraining-manifest", str(self.manifest),
            "--pretraining-test-data", str(self.test_data), "--out", str(output),
            "--steps", "1", "--batch-size", "2", "--grad-accum-steps", "1",
            "--max-grad-norm", "0.5",
            "--eval-every", "1", "--eval-batches", "1", "--checkpoint-every", "1"
        ], capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        saved = torch.load(output, map_location="cpu", weights_only=True)
        self.assertEqual(saved["tokenizer"], tokenizer.to_dict())
        self.assertEqual(saved["sft_training"]["max_grad_norm"], 0.5)
        self.assertEqual(saved["config"]["vocab_size"], tokenizer.vocab_size)
        audited = run_with_sft_gate([
            sys.executable, str(Path(__file__).with_name("audit_checkpoint.py")),
            "--checkpoint", str(output)
        ], capture_output=True, text=True, timeout=30)
        self.assertEqual(audited.returncode, 0, audited.stdout + audited.stderr)
        self.assertEqual(json.loads(audited.stdout)["tokenizer"], "dream-bpe-v1")
        evaluated = run_with_sft_gate([
            sys.executable, str(EVALUATOR), "--checkpoint", str(output),
            "--data", str(self.test_data), "--sample-tokens", "2",
            "--batches", "1", "--batch-size", "1",
            "--out", str(self.root / "bpe-sft-eval.json")
        ], capture_output=True, text=True, timeout=60)
        self.assertEqual(evaluated.returncode, 0, evaluated.stdout + evaluated.stderr)

        resumed = run_with_sft_gate([
            sys.executable, str(SCRIPT), "--checkpoint", str(bpe_base),
            "--train-data", str(train_path), "--validation-data", str(val_path),
            "--pretraining-manifest", str(self.manifest),
            "--pretraining-test-data", str(self.test_data), "--out", str(output),
            "--steps", "1", "--batch-size", "2", "--grad-accum-steps", "1",
            "--max-grad-norm", "0.5",
            "--eval-every", "1", "--eval-batches", "1", "--checkpoint-every", "1",
            "--resume", str(output)
        ], capture_output=True, text=True, timeout=120)
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        resumed_checkpoint = torch.load(output, map_location="cpu", weights_only=True)
        self.assertEqual(resumed_checkpoint["sft_training"]["completed_steps"], 2)
        self.assertEqual(resumed_checkpoint["tokenizer"], tokenizer.to_dict())

        corrupted = dict(resumed_checkpoint)
        corrupted["tokenizer"] = "utf8-byte-v1"
        corrupted_path = self.root / "wrong-tokenizer-resume.pt"
        torch.save(corrupted, corrupted_path)
        rejected = run_with_sft_gate([
            sys.executable, str(SCRIPT), "--checkpoint", str(bpe_base),
            "--train-data", str(train_path), "--validation-data", str(val_path),
            "--pretraining-manifest", str(self.manifest),
            "--pretraining-test-data", str(self.test_data), "--out", str(output),
            "--steps", "1", "--batch-size", "2", "--grad-accum-steps", "1",
            "--eval-every", "1", "--eval-batches", "1", "--checkpoint-every", "1",
            "--resume", str(corrupted_path)
        ], capture_output=True, text=True, timeout=30)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("resume tokenizer does not match", rejected.stderr)

    def test_rejects_duplicate_dialogue_inside_one_sft_split(self):
        train_path = self.root / "train.jsonl"
        self.write_records(train_path, "train")
        rows = [json.loads(line) for line in train_path.read_text().splitlines()]
        rows[1]["messages"] = rows[0]["messages"]
        train_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8"
        )
        with self.assertRaisesRegex(SystemExit, "Duplicate SFT dialogue content within"):
            record_signatures(train_path)

    def test_rejects_duplicate_id_inside_one_sft_split(self):
        train_path = self.root / "train.jsonl"
        self.write_records(train_path, "train")
        rows = [json.loads(line) for line in train_path.read_text().splitlines()]
        rows[1]["id"] = rows[0]["id"]
        train_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8"
        )
        with self.assertRaisesRegex(SystemExit, "Duplicate SFT record ID within"):
            record_signatures(train_path)

    def test_sft_rejects_missing_or_invalid_source_date(self):
        train_path = self.root / "train.jsonl"
        self.write_records(train_path, "train")
        rows = [json.loads(line) for line in train_path.read_text().splitlines()]
        for value in (None, "15-01-2026", "2026-02-30"):
            candidate = [dict(row) for row in rows]
            if value is None:
                candidate[0].pop("source_date")
            else:
                candidate[0]["source_date"] = value
            path = self.root / "invalid-sft.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in candidate),
                encoding="utf-8"
            )
            with self.subTest(value=value):
                with self.assertRaises(SystemExit) as raised:
                    load_examples(path, DreamByteTokenizer(), self.config.block_size)
                self.assertIn("source_date", str(raised.exception))

    def test_sft_training_creates_resumable_checkpoint(self):
        train_path = self.root / "train.jsonl"
        val_path = self.root / "validation.jsonl"
        self.write_records(train_path, "train")
        self.write_records(val_path, "validation")
        output = self.root / "dream-instruct.pt"
        result = run_with_sft_gate(
            [
                sys.executable, str(SCRIPT),
                "--checkpoint", str(self.base),
                "--train-data", str(train_path),
                "--validation-data", str(val_path),
                "--pretraining-manifest", str(self.manifest),
                "--pretraining-test-data", str(self.test_data),
                "--out", str(output),
                "--steps", "2",
                "--batch-size", "2",
                "--grad-accum-steps", "2",
                "--eval-every", "1",
                "--eval-batches", "2",
                "--checkpoint-every", "1"
            ],
            capture_output=True, text=True, timeout=120
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(output.exists())
        saved = torch.load(output, map_location="cpu", weights_only=True)
        self.assertEqual(saved["sft_training"]["completed_steps"], 2)
        self.assertEqual(saved["sft_training"]["method"],
                         "assistant-response-only-supervised-finetuning")
        self.assertEqual(saved["sft_training"]["effective_batch_size"], 4)
        self.assertTrue(saved["base_checkpoint_sha256"])
        self.assertTrue(saved["sft_training"]["train_sha256"])
        self.assertTrue(saved["sft_training"]["validation_sha256"])
        self.assertEqual(
            saved["sft_training"]["pretraining_test_sha256"],
            hashlib.sha256(self.test_data.read_bytes()).hexdigest()
        )
        self.assertIn("python_rng_state", saved)
        self.assertIn("torch_rng_state", saved)

        resumed = run_with_sft_gate(
            [
                sys.executable, str(SCRIPT),
                "--checkpoint", str(self.base),
                "--train-data", str(train_path),
                "--validation-data", str(val_path),
                "--pretraining-manifest", str(self.manifest),
                "--pretraining-test-data", str(self.test_data),
                "--out", str(output),
                "--steps", "1",
                "--batch-size", "2",
                "--grad-accum-steps", "2",
                "--eval-every", "1",
                "--eval-batches", "2",
                "--checkpoint-every", "1",
                "--resume", str(output)
            ],
            capture_output=True, text=True, timeout=120
        )
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        saved = torch.load(output, map_location="cpu", weights_only=True)
        self.assertEqual(saved["sft_training"]["completed_steps"], 3)

        leaked = run_with_sft_gate(
            [
                sys.executable, str(EVALUATOR),
                "--checkpoint", str(output),
                "--data", str(train_path),
                "--sample-tokens", "0",
                "--out", str(self.root / "leaked-evaluation.json")
            ],
            capture_output=True, text=True, timeout=30
        )
        self.assertNotEqual(leaked.returncode, 0)
        self.assertIn("matches training or validation", leaked.stderr)

        valid_eval = run_with_sft_gate(
            [
                sys.executable, str(EVALUATOR),
                "--checkpoint", str(output),
                "--data", str(self.test_data),
                "--sample-tokens", "0",
                "--batches", "1", "--batch-size", "1",
                "--out", str(self.root / "valid-evaluation.json")
            ],
            capture_output=True, text=True, timeout=30
        )
        self.assertEqual(valid_eval.returncode, 0, valid_eval.stdout + valid_eval.stderr)

    def test_rejects_overlapping_source_groups(self):
        train_path = self.root / "train.jsonl"
        val_path = self.root / "validation.jsonl"
        self.write_records(train_path, "train")
        self.write_records(val_path, "validation")
        train_rows = [json.loads(line) for line in train_path.read_text().splitlines()]
        val_rows = [json.loads(line) for line in val_path.read_text().splitlines()]
        for train_row, val_row in zip(train_rows, val_rows):
            val_row["source_group"] = train_row["source_group"]
        val_path.write_text(
            "".join(json.dumps(row) + "\n" for row in val_rows),
            encoding="utf-8"
        )
        result = run_with_sft_gate(
            [
                sys.executable, str(SCRIPT),
                "--checkpoint", str(self.base),
                "--train-data", str(train_path),
                "--validation-data", str(val_path),
                "--pretraining-manifest", str(self.manifest),
                "--pretraining-test-data", str(self.test_data),
                "--steps", "1"
            ],
            capture_output=True, text=True, timeout=30
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source groups must be disjoint", result.stderr)

    def test_rejects_duplicate_record_ids_across_sft_splits(self):
        train_path = self.root / "train.jsonl"
        val_path = self.root / "validation.jsonl"
        self.write_records(train_path, "train")
        self.write_records(val_path, "validation")
        train_rows = [json.loads(line) for line in train_path.read_text().splitlines()]
        val_rows = [json.loads(line) for line in val_path.read_text().splitlines()]
        val_rows[0]["id"] = train_rows[0]["id"]
        val_path.write_text(
            "".join(json.dumps(row) + "\n" for row in val_rows),
            encoding="utf-8"
        )
        result = run_with_sft_gate(
            [
                sys.executable, str(SCRIPT),
                "--checkpoint", str(self.base),
                "--train-data", str(train_path),
                "--validation-data", str(val_path),
                "--pretraining-manifest", str(self.manifest),
                "--pretraining-test-data", str(self.test_data),
                "--steps", "1"
            ],
            capture_output=True, text=True, timeout=30
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("record IDs must be disjoint", result.stderr)

    def test_rejects_duplicate_dialogue_content_across_sft_splits(self):
        train_path = self.root / "train.jsonl"
        val_path = self.root / "validation.jsonl"
        self.write_records(train_path, "train")
        self.write_records(val_path, "validation")
        train_rows = [json.loads(line) for line in train_path.read_text().splitlines()]
        val_rows = [json.loads(line) for line in val_path.read_text().splitlines()]
        val_rows[0]["messages"] = train_rows[0]["messages"]
        val_path.write_text(
            "".join(json.dumps(row) + "\n" for row in val_rows),
            encoding="utf-8"
        )
        result = run_with_sft_gate(
            [
                sys.executable, str(SCRIPT),
                "--checkpoint", str(self.base),
                "--train-data", str(train_path),
                "--validation-data", str(val_path),
                "--pretraining-manifest", str(self.manifest),
                "--pretraining-test-data", str(self.test_data),
                "--steps", "1"
            ],
            capture_output=True, text=True, timeout=30
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Duplicate SFT dialogue content", result.stderr)

    def test_rejects_test_corpus_that_does_not_match_manifest(self):
        train_path = self.root / "train.jsonl"
        val_path = self.root / "validation.jsonl"
        self.write_records(train_path, "train")
        self.write_records(val_path, "validation")
        self.test_data.write_text(
            "Changed after the pretraining manifest was generated. " * 5,
            encoding="utf-8"
        )
        result = run_with_sft_gate(
            [
                sys.executable, str(SCRIPT),
                "--checkpoint", str(self.base),
                "--train-data", str(train_path),
                "--validation-data", str(val_path),
                "--pretraining-manifest", str(self.manifest),
                "--pretraining-test-data", str(self.test_data),
                "--steps", "1"
            ],
            capture_output=True, text=True, timeout=30
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("hash does not match", result.stderr)

    def test_rejects_sft_sources_used_by_pretraining_test(self):
        train_path = self.root / "train.jsonl"
        val_path = self.root / "validation.jsonl"
        self.write_records(train_path, "train")
        self.write_records(val_path, "validation")
        self.write_manifest(["train-group-0"])
        result = run_with_sft_gate(
            [
                sys.executable, str(SCRIPT),
                "--checkpoint", str(self.base),
                "--train-data", str(train_path),
                "--validation-data", str(val_path),
                "--pretraining-manifest", str(self.manifest),
                "--pretraining-test-data", str(self.test_data),
                "--steps", "1"
            ],
            capture_output=True, text=True, timeout=30
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pretraining held-out test groups", result.stderr)

    def test_rejects_sft_source_overlap_with_hashed_pretraining_manifest(self):
        train_path = self.root / "train.jsonl"
        val_path = self.root / "validation.jsonl"
        self.write_records(train_path, "train")
        self.write_records(val_path, "validation")
        self.write_manifest(["placeholder"])
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        manifest["identifier_format"] = "sha256-domain-separated-v1"
        manifest["splits"]["test"]["source_group_ids"] = [
            source_group_fingerprint("train-group-0")
        ]
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        result = run_with_sft_gate(
            [
                sys.executable, str(SCRIPT),
                "--checkpoint", str(self.base),
                "--train-data", str(train_path),
                "--validation-data", str(val_path),
                "--pretraining-manifest", str(self.manifest),
                "--pretraining-test-data", str(self.test_data),
                "--steps", "1"
            ],
            capture_output=True, text=True, timeout=30
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pretraining held-out test groups", result.stderr)

    def test_refuses_random_untrained_base(self):
        untrained = self.root / "untrained.pt"
        model = DreamTransformer(self.config)
        torch.save({
            "model_state": model.state_dict(),
            "config": self.config.__dict__,
            "tokenizer": "utf8-byte-v1",
            "training": {"completed_steps": 0}
        }, untrained)
        train_path = self.root / "train.jsonl"
        val_path = self.root / "validation.jsonl"
        self.write_records(train_path, "train")
        self.write_records(val_path, "validation")
        result = run_with_sft_gate(
            [
                sys.executable, str(SCRIPT),
                "--checkpoint", str(untrained),
                "--train-data", str(train_path),
                "--validation-data", str(val_path),
                "--pretraining-manifest", str(self.manifest),
                "--pretraining-test-data", str(self.test_data)
            ],
            capture_output=True, text=True, timeout=30
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("completed pretraining steps", result.stderr)


    def test_sft_rejects_invalid_gradient_clipping_thresholds(self):
        train_path = self.root / "gradient-train.jsonl"
        val_path = self.root / "gradient-validation.jsonl"
        self.write_records(train_path, "gradient-train")
        self.write_records(val_path, "gradient-validation")
        for invalid in ("0", "-1", "nan", "inf"):
            with self.subTest(invalid=invalid):
                result = run_with_sft_gate(
                    [
                        sys.executable, str(SCRIPT),
                        "--checkpoint", str(self.base),
                        "--train-data", str(train_path),
                        "--validation-data", str(val_path),
                        "--pretraining-manifest", str(self.manifest),
                        "--pretraining-test-data", str(self.test_data),
                        "--max-grad-norm", invalid,
                        "--steps", "1"
                    ],
                    capture_output=True, text=True, timeout=30
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "max-grad-norm must be finite and positive",
                    result.stderr
                )


if __name__ == "__main__":
    unittest.main()
