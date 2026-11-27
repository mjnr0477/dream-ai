#!/usr/bin/env python3
"""Fetch new, explicitly shared DREAM AI examples and run the human review gate.

After an operator reviews each exchange, approved decisions are synced back to
the private Supabase learning store. Training remains scheduled separately.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))

from dream_ai_optin_pipeline import export_optin_rows, write_jsonl
from review_candidates import review_file
from sync_dream_ai_reviews_to_supabase import sync_file
from dream_ai_supabase_guard import validate_dream_ai_supabase_url

CONSENT_VERSION = "dream-learning-opt-in-v3"


def fetch_pending(base_url: str, key: str) -> list[dict]:
    base_url = validate_dream_ai_supabase_url(base_url)
    rows = []
    offset = 0
    page_size = 500
    while True:
        params = urllib.parse.urlencode({
            "select": "id,mode,prompt,response,consented,consent_version,consented_at,created_at,reviewed_at,content_sha256,metadata,status,review_status",
            "consented": "eq.true",
            "consent_version": f"eq.{CONSENT_VERSION}",
            "status": "eq.pending",
            "review_status": "eq.pending",
            "order": "created_at.asc",
            "limit": str(page_size),
            "offset": str(offset),
        })
        request = urllib.request.Request(
            f"{base_url}/rest/v1/dream_ai_training_examples?{params}",
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
                "Range": f"{offset}-{offset + page_size - 1}",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            page = json.loads(response.read().decode("utf-8"))
        if not isinstance(page, list):
            raise RuntimeError("Supabase returned an unexpected response.")
        rows.extend(page)
        if len(page) < page_size:
            return rows
        offset += page_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--approved-output",
        default=str(Path.home() / ".dream-ai-learning" / "approved-records.jsonl"),
    )
    parser.add_argument(
        "--decisions",
        default=str(Path.home() / ".dream-ai-learning" / "review-decisions.jsonl"),
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    base_url = os.environ.get("DREAM_AI_SUPABASE_URL", "").strip()
    key = os.environ.get("DREAM_AI_SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not base_url or not key:
        parser.error("Set DREAM_AI_SUPABASE_URL and DREAM_AI_SUPABASE_SERVICE_ROLE_KEY privately.")
    if not base_url.startswith("https://") or not base_url.endswith(".supabase.co"):
        parser.error("DREAM_AI_SUPABASE_URL must be an HTTPS Supabase project URL.")

    rows = fetch_pending(base_url, key)
    candidates, report = export_optin_rows(rows)
    print(json.dumps({
        "pending_v3_consented_rows": report["input_rows"],
        "reviewable_candidates": report["exported"],
        "skipped": report["skipped"],
        "approval_flags_set_by_exporter": False,
    }, indent=2))

    if not candidates:
        print("No new v3 contributions are waiting for review.")
        return

    approved_path = Path(args.approved_output).expanduser()
    decisions_path = Path(args.decisions).expanduser()
    approved_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dream-ai-review-") as temp:
        candidate_path = Path(temp) / "candidates.jsonl"
        write_jsonl(candidate_path, candidates)
        review_summary = review_file(
            candidate_path,
            approved_path,
            decisions_path,
            limit=args.limit,
        )

    synced = sync_file(approved_path, base_url, key)
    print(json.dumps({
        "human_review": review_summary,
        "supabase_sync": synced,
        "next": "The daily training workflow will run once the approved corpus reaches its minimum size and source diversity.",
    }, indent=2))


if __name__ == "__main__":
    main()
