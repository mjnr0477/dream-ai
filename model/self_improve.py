"""Run one verified DREAM self-improvement cycle from approved feedback.

This controller never invents or approves training data. It fine-tunes only on
human-reviewed SFT records, evaluates on the unchanged pretraining held-out
test corpus, and promotes a candidate only when technical and regression gates
pass. Task benchmark scores are diagnostic, not a production certification.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


MODEL_DIR = Path(__file__).resolve().parent


def read_checkpoint(path):
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Cannot safely read checkpoint: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Checkpoint root must be an object")
    return value


def run_command(command, label):
    result = subprocess.run(
        [str(part) for part in command],
        cwd=MODEL_DIR.parent,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "No command output").strip()
        raise RuntimeError(f"{label} failed (exit {result.returncode}): {detail[-5000:]}")
    return result.stdout


def read_report(path, label):
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label} report: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError(f"{label} report must be a JSON object")
    return report


def decide_promotion(
    baseline_evaluation,
    candidate_evaluation,
    baseline_tasks,
    candidate_tasks,
    max_regression_bits_per_byte=0.0,
):
    """Return a deterministic gate decision; lower bits/byte is better."""
    tolerance = max_regression_bits_per_byte
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError("Regression tolerance must be numeric")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Regression tolerance must be finite and non-negative")

    baseline_loss = baseline_evaluation.get("bits_per_byte")
    candidate_loss = candidate_evaluation.get("bits_per_byte")
    for label, value in (("baseline", baseline_loss), ("candidate", candidate_loss)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{label} evaluation has invalid bits_per_byte")

    checks = {
        "held_out_language_model_loss_not_regressed":
            candidate_loss <= baseline_loss + tolerance,
        "task_benchmark_not_regressed": True,
    }
    baseline_task_rate = None
    candidate_task_rate = None
    if candidate_tasks is not None:
        candidate_task_rate = candidate_tasks.get("exact_match_rate")
        if (isinstance(candidate_task_rate, bool)
                or not isinstance(candidate_task_rate, (int, float))
                or not math.isfinite(candidate_task_rate)
                or not 0 <= candidate_task_rate <= 1):
            raise ValueError("Candidate task report has invalid exact_match_rate")
    if baseline_tasks is not None:
        baseline_task_rate = baseline_tasks.get("exact_match_rate")
        if (isinstance(baseline_task_rate, bool)
                or not isinstance(baseline_task_rate, (int, float))
                or not math.isfinite(baseline_task_rate)
                or not 0 <= baseline_task_rate <= 1):
            raise ValueError("Baseline task report has invalid exact_match_rate")
        if candidate_task_rate is None:
            raise ValueError("Candidate task report is required when a baseline task report exists")
        checks["task_benchmark_not_regressed"] = candidate_task_rate >= baseline_task_rate

    return {
        "promote": all(checks.values()),
        "checks": checks,
        "baseline_bits_per_byte": baseline_loss,
        "candidate_bits_per_byte": candidate_loss,
        "baseline_task_exact_match_rate": baseline_task_rate,
        "candidate_task_exact_match_rate": candidate_task_rate,
        "task_baseline_available": baseline_tasks is not None,
        "limitations": [
            "Held-out next-token loss is not a measure of assistant usefulness or reasoning.",
            "The starter exact-match benchmark is small and diagnostic, not comprehensive.",
            "Passing these gates does not certify factuality, safety, privacy or production readiness.",
            "Only independently reviewed and explicitly approved SFT records may be used.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Current native DREAM checkpoint")
    parser.add_argument("--train-data", required=True, help="Human-reviewed, explicitly approved SFT JSONL")
    parser.add_argument("--validation-data", required=True, help="Independent human-reviewed SFT JSONL")
    parser.add_argument("--pretraining-manifest", required=True)
    parser.add_argument("--pretraining-test-data", required=True, help="Exact held-out test file from the manifest")
    parser.add_argument("--benchmark", default="evals/dream-core-v1.jsonl")
    parser.add_argument("--out", required=True, help="Active checkpoint path to create or replace after passing gates")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--max-regression-bits-per-byte", type=float, default=0.0)
    parser.add_argument("--evaluation-batches", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()

    source = Path(args.checkpoint).resolve()
    active = Path(args.out).resolve()
    train_data = Path(args.train_data).resolve()
    validation_data = Path(args.validation_data).resolve()
    manifest = Path(args.pretraining_manifest).resolve()
    test_data = Path(args.pretraining_test_data).resolve()
    benchmark = Path(args.benchmark).resolve()

    for label, path in (
        ("checkpoint", source), ("training data", train_data),
        ("validation data", validation_data), ("pretraining manifest", manifest),
        ("held-out test data", test_data), ("task benchmark", benchmark),
    ):
        if not path.is_file():
            parser.error(f"{label} does not exist: {path}")
    if source == train_data or source == validation_data or train_data == validation_data:
        parser.error("Checkpoint, training data and validation data must be separate files.")
    if args.steps < 1 or args.batch_size < 1 or args.grad_accum_steps < 1:
        parser.error("steps, batch-size and grad-accum-steps must be positive.")
    if args.eval_batches < 1 or args.evaluation_batches < 1:
        parser.error("Evaluation batch counts must be positive.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive.")
    if not math.isfinite(args.max_regression_bits_per_byte) or args.max_regression_bits_per_byte < 0:
        parser.error("max-regression-bits-per-byte must be finite and non-negative.")
    if not 1 <= args.max_new_tokens <= 256:
        parser.error("max-new-tokens must be between 1 and 256.")

    active.parent.mkdir(parents=True, exist_ok=True)
    report_path = active.with_suffix(active.suffix + ".self-improve.json")
    backup_path = active.with_name(active.stem + ".previous" + active.suffix)
    base = read_checkpoint(source)
    if not isinstance(base.get("training"), dict) and not isinstance(base.get("sft_training"), dict):
        parser.error("The source must be a DREAM checkpoint with verified training metadata.")
    baseline_is_instruction_tuned = isinstance(base.get("sft_training"), dict)

    with tempfile.TemporaryDirectory(prefix="dream-self-improve-", dir=active.parent) as temp:
        work = Path(temp)
        candidate = work / "candidate.pt"
        baseline_audit = work / "baseline-audit.json"
        baseline_eval = work / "baseline-evaluation.json"
        candidate_audit = work / "candidate-audit.json"
        candidate_eval = work / "candidate-evaluation.json"
        candidate_tasks_path = work / "candidate-tasks.json"
        baseline_tasks_path = work / "baseline-tasks.json"
        release_gate = work / "release-gate.json"

        run_command([
            sys.executable, MODEL_DIR / "audit_checkpoint.py",
            "--checkpoint", source, "--out", baseline_audit,
        ], "Baseline checkpoint audit")
        run_command([
            sys.executable, MODEL_DIR / "evaluate.py",
            "--checkpoint", source, "--data", test_data,
            "--batches", args.evaluation_batches, "--batch-size", 8,
            "--seed", 2026, "--sample-tokens", 0, "--out", baseline_eval,
        ], "Baseline held-out evaluation")
        baseline_eval_report = read_report(baseline_eval, "baseline evaluation")

        baseline_tasks = None
        if baseline_is_instruction_tuned:
            run_command([
                sys.executable, MODEL_DIR / "evaluate_tasks.py",
                "--checkpoint", source, "--benchmark", benchmark,
                "--max-new-tokens", args.max_new_tokens, "--out", baseline_tasks_path,
            ], "Baseline task benchmark")
            baseline_tasks = read_report(baseline_tasks_path, "baseline task benchmark")

        run_command([
            sys.executable, MODEL_DIR / "sft.py",
            "--checkpoint", source,
            "--train-data", train_data,
            "--validation-data", validation_data,
            "--pretraining-manifest", manifest,
            "--pretraining-test-data", test_data,
            "--out", candidate,
            "--steps", args.steps,
            "--batch-size", args.batch_size,
            "--grad-accum-steps", args.grad_accum_steps,
            "--learning-rate", args.learning_rate,
            "--eval-batches", args.eval_batches,
        ], "DREAM supervised self-improvement training")

        run_command([
            sys.executable, MODEL_DIR / "audit_checkpoint.py",
            "--checkpoint", candidate, "--out", candidate_audit,
        ], "Candidate checkpoint audit")
        run_command([
            sys.executable, MODEL_DIR / "evaluate.py",
            "--checkpoint", candidate, "--data", test_data,
            "--batches", args.evaluation_batches, "--batch-size", 8,
            "--seed", 2026, "--sample-tokens", 0, "--out", candidate_eval,
        ], "Candidate held-out evaluation")
        run_command([
            sys.executable, MODEL_DIR / "evaluate_tasks.py",
            "--checkpoint", candidate, "--benchmark", benchmark,
            "--max-new-tokens", args.max_new_tokens, "--out", candidate_tasks_path,
        ], "Candidate task benchmark")
        run_command([
            sys.executable, MODEL_DIR / "validate_release.py",
            "--checkpoint", candidate, "--audit", candidate_audit,
            "--evaluation", candidate_eval, "--out", release_gate,
        ], "Candidate technical release gate")

        candidate_eval_report = read_report(candidate_eval, "candidate evaluation")
        candidate_tasks = read_report(candidate_tasks_path, "candidate task benchmark")
        gate = decide_promotion(
            baseline_eval_report, candidate_eval_report,
            baseline_tasks, candidate_tasks,
            args.max_regression_bits_per_byte,
        )
        gate["technical_release_preflight_passed"] = read_report(
            release_gate, "technical release gate"
        ).get("technical_preflight_passed") is True

        if not gate["technical_release_preflight_passed"]:
            gate["promote"] = False
            gate["checks"]["technical_release_preflight_passed"] = False
        else:
            gate["checks"]["technical_release_preflight_passed"] = True
            gate["promote"] = gate["promote"] and all(gate["checks"].values())

        report = {
            "schema": "dream-self-improvement-cycle-v1",
            "source_checkpoint": str(source),
            "source_checkpoint_sha256": baseline_eval_report["checkpoint_sha256"],
            "candidate_checkpoint_sha256": candidate_eval_report["checkpoint_sha256"],
            "training_data_sha256": __import__("hashlib").sha256(train_data.read_bytes()).hexdigest(),
            "validation_data_sha256": __import__("hashlib").sha256(validation_data.read_bytes()).hexdigest(),
            "pretraining_test_sha256": baseline_eval_report["evaluation_corpus_sha256"],
            "training_steps": args.steps,
            "decision": gate,
            "promoted": False,
        }

        if gate["promote"]:
            shutil.copy2(source, backup_path)
            staged = active.with_suffix(active.suffix + ".tmp")
            shutil.copy2(candidate, staged)
            os.replace(staged, active)
            report["promoted"] = True
            report["active_checkpoint"] = str(active)
            report["previous_checkpoint"] = str(backup_path)
        else:
            report["candidate_checkpoint_retained"] = False
            report["active_checkpoint_unchanged"] = str(source)

        temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_report, report_path)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["promoted"]:
        raise SystemExit("DREAM candidate was not promoted; active checkpoint was left unchanged.")


if __name__ == "__main__":
    main()
