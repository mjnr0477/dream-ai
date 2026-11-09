#!/usr/bin/env python3
"""Create and finalize DREAM AI automatic-training run records."""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3-1.7B"
HF_REPO = "msjnr0477/dream-ai-adapter"


def api(method: str, table: str, key: str, payload=None, filters=None):
    base = os.environ["DREAM_AI_SUPABASE_URL"].rstrip("/")
    query = urllib.parse.urlencode(filters or {})
    url = f"{base}/rest/v1/{table}" + (f"?{query}" if query else "")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Prefer": "return=representation",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read()
            return json.loads(raw.decode("utf-8")) if raw else []
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Supabase returned HTTP {exc.code} for {method} {table}.") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Could not reach Supabase while recording the training run.") from exc


def credentials():
    base = os.environ.get("DREAM_AI_SUPABASE_URL", "")
    key = os.environ.get("DREAM_AI_SUPABASE_SERVICE_ROLE_KEY", "")
    if not base.startswith("https://") or not base.endswith(".supabase.co") or not key:
        raise RuntimeError("Missing or invalid DREAM AI Supabase credentials.")
    return key


def start(summary_path: Path) -> str:
    key = credentials()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("eligible") is not True or summary.get("audit_gate_passed") is not True:
        raise RuntimeError("Cannot start a run without an eligible dataset and passing audit gate.")
    metrics = {
        "dataset_sha256": summary["dataset_sha256"],
        "train_records": summary["train_records"],
        "validation_records": summary["validation_records"],
        "independent_source_groups": summary["independent_source_groups"],
        "audit_gate_passed": True,
        "parent_adapter_path": summary.get("parent_adapter_path", "production-root"),
        "status": "submitted_to_kaggle",
    }
    rows = api(
        "POST",
        "dream_ai_training_runs",
        key,
        payload={
            "status": "running",
            "dataset_count": summary["train_records"] + summary["validation_records"],
            "base_model": BASE_MODEL,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        },
    )
    if not isinstance(rows, list) or len(rows) != 1 or not rows[0].get("id"):
        raise RuntimeError("Supabase did not return a training run ID.")
    return str(rows[0]["id"])


def finish(run_id: str, candidate_path: Path):
    key = credentials()
    manifest = json.loads(candidate_path.read_text(encoding="utf-8"))
    required = (
        "candidate_id", "dataset_sha256", "baseline_validation_loss",
        "candidate_validation_loss", "validation_relative_change",
        "training_budget_reached",
    )
    missing = [field for field in required if field not in manifest]
    if missing:
        raise RuntimeError("Candidate manifest is missing required fields: " + ", ".join(missing))
    if manifest.get("status") != "candidate-not-active":
        raise RuntimeError("Candidate manifest does not have the expected non-active status.")
    if manifest.get("audit_gate_passed") is not True:
        raise RuntimeError("Candidate manifest does not confirm the audit gate.")
    run_rows = api(
        "GET",
        "dream_ai_training_runs",
        key,
        filters={"select": "id,status,metrics", "id": f"eq.{run_id}", "limit": "2"},
    )
    if not isinstance(run_rows, list) or len(run_rows) != 1:
        raise RuntimeError("Could not identify the expected DREAM AI training run.")
    run_metrics = run_rows[0].get("metrics") or {}
    if isinstance(run_metrics, str):
        run_metrics = json.loads(run_metrics)
    if run_rows[0].get("status") != "running":
        raise RuntimeError("Training run is not in the running state.")
    if run_metrics.get("dataset_sha256") != manifest.get("dataset_sha256"):
        raise RuntimeError("Candidate dataset fingerprint does not match its registered run.")
    artifact = f"https://huggingface.co/{HF_REPO}/tree/main/candidates/{manifest['candidate_id']}"
    metrics = dict(manifest)
    metrics["artifact_uri"] = artifact
    metrics["registered_automatically"] = True
    updated_runs = api(
        "PATCH",
        "dream_ai_training_runs",
        key,
        payload={
            "status": "succeeded",
            "candidate_model": manifest["candidate_id"],
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        },
        filters={"id": f"eq.{run_id}"},
    )
    if not isinstance(updated_runs, list) or len(updated_runs) != 1:
        raise RuntimeError("Supabase did not confirm the training-run update.")
    versions = api(
        "POST",
        "dream_ai_model_versions",
        key,
        payload={
            "model_name": manifest["candidate_id"],
            "artifact_uri": artifact,
            "base_model": BASE_MODEL,
            "metrics": metrics,
            "status": "candidate",
        },
    )
    if not isinstance(versions, list) or len(versions) != 1:
        raise RuntimeError("Candidate was trained but model-version registration was not confirmed.")
    print(json.dumps({
        "run_id": run_id,
        "candidate_id": manifest["candidate_id"],
        "candidate_status": "candidate-not-active",
        "artifact_uri": artifact,
        "training_budget_reached": manifest["training_budget_reached"],
        "production_adapter_changed": False,
    }, indent=2))


def fail(run_id: str, reason: str):
    key = credentials()
    api(
        "PATCH",
        "dream_ai_training_runs",
        key,
        payload={
            "status": "failed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "metrics": {"failure_reason": reason[:500]},
        },
        filters={"id": f"eq.{run_id}"},
    )
    print(json.dumps({"run_id": run_id, "status": "failed"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "finish", "fail"))
    parser.add_argument("--summary")
    parser.add_argument("--run-id")
    parser.add_argument("--candidate-manifest")
    parser.add_argument("--reason", default="Kaggle training did not complete successfully.")
    args = parser.parse_args()

    if args.action == "start":
        if not args.summary:
            parser.error("--summary is required for start")
        print(start(Path(args.summary)))
    elif args.action == "finish":
        if not args.run_id or not args.candidate_manifest:
            parser.error("--run-id and --candidate-manifest are required for finish")
        finish(args.run_id, Path(args.candidate_manifest))
    else:
        if not args.run_id:
            parser.error("--run-id is required for fail")
        fail(args.run_id, args.reason)


if __name__ == "__main__":
    main()
