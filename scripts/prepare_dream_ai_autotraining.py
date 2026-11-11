#!/usr/bin/env python3
"""Prepare an automatically-triggered DREAM AI adapter run from approved v3 rows.

Only explicit opt-ins that have four recorded human approvals are accepted.
The script never approves data and never reads ordinary chat/learner tables.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "model"))

from dream_ai_optin_pipeline import export_optin_rows, prepare_sft_splits, write_jsonl
from audit_corpus import audit
from verify_dream_ai_sft_audits import verify_audit_reports

MIN_RECORDS = 32
MIN_SOURCE_GROUPS = 8
MIN_VALIDATION_RECORDS = 4
REVIEW_FLAGS = (
    "rights_confirmed",
    "privacy_reviewed",
    "quality_reviewed",
    "approval_granted",
)
BASE_MODEL = "Qwen/Qwen3-1.7B"
CONSENT_VERSION = "dream-learning-opt-in-v3"


def get_json(url: str, key: str, range_header: str | None = None):
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }
    if range_header:
        headers["Range"] = range_header
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            return json.loads(raw.decode("utf-8")) if raw else []
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Supabase read failed with HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Could not reach the DREAM AI Supabase project.") from exc


def fetch_all(base_url: str, key: str, table: str, query: dict, page_size: int = 1000):
    rows = []
    offset = 0
    while True:
        params = dict(query)
        params.update({"limit": str(page_size), "offset": str(offset)})
        url = f"{base_url.rstrip('/')}/rest/v1/{table}?{urllib.parse.urlencode(params)}"
        page = get_json(url, key, f"{offset}-{offset + page_size - 1}")
        if not isinstance(page, list):
            raise RuntimeError(f"Unexpected response while reading {table}.")
        rows.extend(page)
        if len(page) < page_size:
            return rows
        offset += page_size


def previous_success(base_url: str, key: str, dataset_sha256: str) -> dict | None:
    rows = fetch_all(
        base_url, key, "dream_ai_training_runs",
        {"select": "status,metrics,created_at", "status": "eq.succeeded"},
        page_size=1000,
    )
    for row in reversed(rows):
        metrics = row.get("metrics")
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except json.JSONDecodeError:
                continue
        if isinstance(metrics, dict) and metrics.get("dataset_sha256") == dataset_sha256:
            return metrics
    return None


def latest_candidate_parent(base_url: str, key: str) -> str:
    """Return the most recent technically validated adapter candidate, if any."""
    rows = fetch_all(
        base_url,
        key,
        "dream_ai_model_versions",
        {
            "select": "model_name,artifact_uri,base_model,metrics,created_at",
            "status": "eq.candidate",
            "base_model": f"eq.{BASE_MODEL}",
            "order": "created_at.desc",
        },
        page_size=1000,
    )
    expected_prefix = f"https://huggingface.co/msjnr0477/dream-ai-adapter/tree/main/candidates/"
    for row in rows:
        model_name = row.get("model_name")
        artifact = row.get("artifact_uri")
        metrics = row.get("metrics")
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except json.JSONDecodeError:
                continue
        if not isinstance(model_name, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", model_name):
            continue
        if artifact != expected_prefix + model_name:
            continue
        if not isinstance(metrics, dict) or metrics.get("audit_gate_passed") is not True:
            continue
        try:
            baseline = float(metrics["baseline_validation_loss"])
            candidate = float(metrics["candidate_validation_loss"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (baseline > 0 and candidate > 0 and candidate <= baseline * 1.02):
            continue
        return "candidates/" + model_name
    return ""


def parse_aware_timestamp(value: object) -> datetime | None:
    """Parse an ISO-8601 timestamp only when it carries an explicit UTC offset."""
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def eligible_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    valid = []
    skipped = {
        "not_explicit_v3_approved": 0,
        "missing_review_metadata": 0,
        "missing_source_group": 0,
        "missing_human_review_timestamp": 0,
        "invalid_consent_timestamp": 0,
        "invalid_human_review_timestamp": 0,
        "review_precedes_consent": 0,
        "missing_content_fingerprint": 0,
        "invalid_review_flags": 0,
    }
    for row in rows:
        if (
            row.get("consented") is not True
            or row.get("consent_version") != CONSENT_VERSION
            or row.get("status") != "approved"
            or row.get("review_status") != "approved"
            or not row.get("reviewed_at")
        ):
            skipped["not_explicit_v3_approved"] += 1
            continue
        consented_at = parse_aware_timestamp(row.get("consented_at"))
        if consented_at is None:
            skipped["invalid_consent_timestamp"] += 1
            continue
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = None
        if not isinstance(metadata, dict):
            skipped["missing_review_metadata"] += 1
            continue
        if not re.fullmatch(r"[a-f0-9]{64}", str(metadata.get("source_group_sha256", ""))):
            skipped["missing_source_group"] += 1
            continue
        reviewed_at = metadata.get("human_reviewed_at") or row.get("reviewed_at")
        human_reviewed_at = parse_aware_timestamp(reviewed_at)
        if human_reviewed_at is None:
            skipped["invalid_human_review_timestamp"] += 1
            continue
        if human_reviewed_at < consented_at:
            skipped["review_precedes_consent"] += 1
            continue
        if not re.fullmatch(r"[a-f0-9]{64}", str(row.get("content_sha256", ""))):
            skipped["missing_content_fingerprint"] += 1
            continue
        review = metadata.get("review")
        if not isinstance(review, dict) or any(review.get(flag) is not True for flag in REVIEW_FLAGS):
            skipped["invalid_review_flags"] += 1
            continue
        copy = dict(row)
        copy["metadata"] = metadata
        # The existing candidate exporter intentionally accepts only pending
        # rows. This local copy remains approved in Supabase; no DB status is
        # changed by this preparation step.
        copy["status"] = "pending"
        copy["review_status"] = "pending"
        valid.append(copy)
    return valid, skipped


def prepare(base_url: str, key: str, output_dir: Path) -> dict:
    rows = fetch_all(
        base_url,
        key,
        "dream_ai_training_examples",
        {
            "select": "id,mode,prompt,response,input_text,target_text,consented,consent_version,consented_at,created_at,reviewed_at,content_sha256,metadata",
            "consented": "eq.true",
            "consent_version": f"eq.{CONSENT_VERSION}",
            "status": "eq.approved",
            "review_status": "eq.approved",
            "reviewed_at": "not.is.null",
            "order": "created_at.asc",
        },
    )
    valid_rows, skipped = eligible_rows(rows)
    candidates, export_report = export_optin_rows(valid_rows)

    by_hash = {
        str(row.get("content_sha256")): row
        for row in valid_rows
    }
    reviewed_records = []
    for candidate in candidates:
        source = by_hash.get(str(candidate.get("content_sha256")))
        if source is None:
            continue
        metadata = source["metadata"]
        review = metadata.get("review") or {}
        for flag in REVIEW_FLAGS:
            candidate[flag] = review.get(flag) is True
        candidate["human_reviewed_at"] = (
            metadata.get("human_reviewed_at") or source.get("reviewed_at")
        )
        reviewed_records.append(candidate)

    source_groups = {
        row.get("source_group")
        for row in reviewed_records
        if isinstance(row.get("source_group"), str)
    }
    result = {
        "eligible": False,
        "status": "waiting_for_approved_data",
        "approved_v3_rows": len(rows),
        "valid_reviewed_rows": len(reviewed_records),
        "independent_source_groups": len(source_groups),
        "minimum_records": MIN_RECORDS,
        "minimum_source_groups": MIN_SOURCE_GROUPS,
        "skipped_review_reasons": skipped,
        "base_model": BASE_MODEL,
    }
    if len(reviewed_records) < MIN_RECORDS or len(source_groups) < MIN_SOURCE_GROUPS:
        return result

    try:
        train, validation, manifest = prepare_sft_splits(
            reviewed_records,
            min_source_groups=MIN_SOURCE_GROUPS,
            min_validation_records=MIN_VALIDATION_RECORDS,
        )
    except (ValueError, RuntimeError) as exc:
        result["status"] = "waiting_for_valid_splits"
        result["split_blocker"] = str(exc)
        return result

    if len(validation) < MIN_VALIDATION_RECORDS:
        result["status"] = "waiting_for_validation_records"
        return result
    if len(train) < 20:
        result["status"] = "waiting_for_training_records"
        result["train_records"] = len(train)
        result["validation_records"] = len(validation)
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "validation.jsonl", validation)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    train_audit = audit(output_dir / "train.jsonl")
    validation_audit = audit(output_dir / "validation.jsonl")
    (output_dir / "train-audit.json").write_text(
        json.dumps(train_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "validation-audit.json").write_text(
        json.dumps(validation_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    gate = verify_audit_reports(manifest, train_audit, validation_audit)
    (output_dir / "audit-gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    prior = previous_success(base_url, key, manifest["dataset_sha256"])
    if prior and prior.get("training_budget_reached") is not True:
        result.update({
            "eligible": False,
            "status": "already_trained_for_this_dataset",
            "dataset_sha256": manifest["dataset_sha256"],
        })
        return result

    parent_adapter_path = latest_candidate_parent(base_url, key) if not prior else str(
        (prior.get("parent_adapter_path") or "")
    )
    (output_dir / "parent_adapter_path.txt").write_text(
        parent_adapter_path + "\n", encoding="utf-8"
    )
    result.update({
        "eligible": True,
        "status": "ready",
        "dataset_sha256": manifest["dataset_sha256"],
        "train_records": len(train),
        "validation_records": len(validation),
        "independent_source_groups": manifest["source_groups"],
        "audit_gate_passed": gate["passed"],
        "output_dir": str(output_dir),
        "resume_previous_budget_limited_run": bool(prior),
        "parent_adapter_path": parent_adapter_path or "production-root",
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary", default="")
    args = parser.parse_args()

    base_url = os.environ.get("DREAM_AI_SUPABASE_URL", "").strip()
    key = os.environ.get("DREAM_AI_SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not base_url or not key:
        parser.error("Set DREAM_AI_SUPABASE_URL and DREAM_AI_SUPABASE_SERVICE_ROLE_KEY.")
    if not base_url.startswith("https://") or not base_url.endswith(".supabase.co"):
        parser.error("DREAM_AI_SUPABASE_URL must be the HTTPS DREAM AI Supabase project URL.")

    result = prepare(base_url, key, Path(args.output_dir))
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    print(encoded)
    if args.summary:
        Path(args.summary).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
