#!/usr/bin/env python3
"""Sync explicitly human-reviewed DREAM AI v3 contributions to the private store.

This script never approves candidates itself. It accepts only the output of
data/review_candidates.py after all four review prompts were answered yes.
The raw Supabase UUID is never written to the local review file or printed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REVIEW_FLAGS = (
    "rights_confirmed",
    "privacy_reviewed",
    "quality_reviewed",
    "approval_granted",
)
CONSENT_VERSION = "dream-learning-opt-in-v3"
SOURCE = "DREAM AI opt-in exchange"
LICENSE = "DREAM AI Contributor Permission v3"
SOURCE_URL = "https://dream-ai-live.onrender.com"
LICENSE_URL = "https://dream-ai-live.onrender.com/learning-data-terms.html"


def request_json(url: str, key: str, method: str = "GET", payload=None):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
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
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            return json.loads(raw.decode("utf-8")) if raw else []
    except urllib.error.HTTPError as exc:
        # Do not print response bodies: they may include database details.
        raise RuntimeError(f"Supabase returned HTTP {exc.code} during {method}.") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Could not reach the Supabase learning store.") from exc


def parse_timestamp(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing_human_review_timestamp")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid_human_review_timestamp") from exc
    if stamp.tzinfo is None:
        raise ValueError("human_review_timestamp_must_include_timezone")
    return stamp.astimezone(timezone.utc).isoformat()


def validate_reviewed_record(record: object, line: int) -> dict:
    if not isinstance(record, dict):
        raise ValueError(f"record_{line}_not_object")
    if record.get("consented") is not True or record.get("consent_version") != CONSENT_VERSION:
        raise ValueError(f"record_{line}_missing_v3_consent")
    if any(record.get(flag) is not True for flag in REVIEW_FLAGS):
        raise ValueError(f"record_{line}_missing_human_review")
    if record.get("source") != SOURCE or record.get("license") != LICENSE:
        raise ValueError(f"record_{line}_unexpected_provenance")
    if record.get("source_url") != SOURCE_URL or record.get("license_url") != LICENSE_URL:
        raise ValueError(f"record_{line}_unexpected_terms_url")
    if record.get("rights_basis") != "explicit-permission":
        raise ValueError(f"record_{line}_missing_explicit_permission")
    if not re.fullmatch(r"[a-f0-9]{64}", str(record.get("content_sha256", ""))):
        raise ValueError(f"record_{line}_missing_content_fingerprint")
    group = str(record.get("source_group", ""))
    match = re.fullmatch(r"dream-ai-session-([a-f0-9]{64})", group)
    if not match:
        raise ValueError(f"record_{line}_invalid_session_fingerprint")
    reviewed_at = parse_timestamp(record.get("human_reviewed_at"))
    return {
        "content_sha256": record["content_sha256"],
        "source_group_sha256": match.group(1),
        "human_reviewed_at": reviewed_at,
        "language": record.get("language", "und"),
        "mode": record.get("dream_mode", "general"),
    }


def sync_file(path: Path, base_url: str, key: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    base_url = base_url.rstrip("/")
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme != "https" or not parsed.hostname or not parsed.hostname.endswith(".supabase.co"):
        raise ValueError("SUPABASE_URL must be an HTTPS Supabase project URL.")

    approved = 0
    already_approved = 0
    rejected = 0
    seen = set()

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
                item = validate_reviewed_record(record, line_no)
            except (json.JSONDecodeError, ValueError) as exc:
                rejected += 1
                raise ValueError(f"Refusing review sync at line {line_no}: {exc}") from exc

            fingerprint = item["content_sha256"]
            if fingerprint in seen:
                raise ValueError(f"Duplicate reviewed contribution fingerprint at line {line_no}.")
            seen.add(fingerprint)

            query = urllib.parse.urlencode({
                "select": "id,metadata,status,review_status,consent_version,consented,content_sha256",
                "content_sha256": f"eq.{fingerprint}",
                "consent_version": f"eq.{CONSENT_VERSION}",
                "consented": "eq.true",
                "limit": "2",
            })
            rows = request_json(
                f"{base_url}/rest/v1/dream_ai_training_examples?{query}", key
            )
            if not isinstance(rows, list) or len(rows) != 1:
                raise ValueError(
                    f"Review at line {line_no} matched {len(rows) if isinstance(rows, list) else 'an invalid number of'} rows; expected exactly one."
                )

            row = rows[0]
            if row.get("content_sha256") != fingerprint:
                raise ValueError(f"Content fingerprint mismatch at line {line_no}.")
            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}

            if row.get("status") == "approved" and row.get("review_status") == "approved":
                existing = metadata.get("review")
                if isinstance(existing, dict) and all(existing.get(flag) is True for flag in REVIEW_FLAGS):
                    already_approved += 1
                    continue

            metadata.update({
                "source_group_sha256": item["source_group_sha256"],
                "language": item["language"],
                "human_reviewed_at": item["human_reviewed_at"],
                "review": {flag: True for flag in REVIEW_FLAGS},
                "review_source": "DREAM AI human review tool",
                "rights_basis": "explicit-permission",
                "source": SOURCE,
                "license": LICENSE,
                "license_url": LICENSE_URL,
            })
            patch_query = urllib.parse.urlencode({
                "content_sha256": f"eq.{fingerprint}",
                "consent_version": f"eq.{CONSENT_VERSION}",
                "consented": "eq.true",
            })
            updated = request_json(
                f"{base_url}/rest/v1/dream_ai_training_examples?{patch_query}",
                key,
                method="PATCH",
                payload={
                    "status": "approved",
                    "review_status": "approved",
                    "reviewed_at": item["human_reviewed_at"],
                    "metadata": metadata,
                },
            )
            if not isinstance(updated, list) or len(updated) != 1:
                raise RuntimeError(
                    f"Supabase did not confirm exactly one review update at line {line_no}."
                )
            approved += 1

    return {
        "review_file_records": approved + already_approved,
        "newly_synced": approved,
        "already_synced": already_approved,
        "rejected": rejected,
        "raw_database_ids_exported": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Approved JSONL from data/review_candidates.py")
    args = parser.parse_args()
    base_url = os.environ.get("DREAM_AI_SUPABASE_URL", "").strip()
    key = os.environ.get("DREAM_AI_SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not base_url or not key:
        parser.error("Set DREAM_AI_SUPABASE_URL and DREAM_AI_SUPABASE_SERVICE_ROLE_KEY in the private environment.")
    result = sync_file(Path(args.input), base_url, key)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
