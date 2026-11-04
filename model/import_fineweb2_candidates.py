"""Convert a local FineWeb2 JSONL export into unapproved DREAM review candidates.

This tool never grants rights approval or trains a model. Every output record
retains provenance and has all human-review flags set to false.
"""
import argparse
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

DATASET_URL = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2"
LICENSE_CLAIM = "ODC-By-1.0 (dataset collection; source-page rights still require review)"
RIGHTS_BASIS = (
    "unverified: dataset collection license does not independently establish "
    "rights to each underlying webpage; review source rights and Common Crawl terms"
)


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def candidate_from_row(row: dict, line_number: int) -> dict:
    required = ("id", "text", "url", "date", "language")
    if any(not isinstance(row.get(key), str) or not row[key].strip() for key in required):
        raise ValueError(f"line {line_number}: required id, text, url, date and language fields are missing")
    if len(row["text"].strip()) < 40:
        raise ValueError(f"line {line_number}: text is shorter than 40 characters")

    parsed_url = urlsplit(row["url"])
    if parsed_url.scheme not in ("http", "https") or not parsed_url.hostname:
        raise ValueError(f"line {line_number}: url must be an absolute HTTP(S) URL")
    try:
        source_date = date.fromisoformat(row["date"][:10]).isoformat()
    except ValueError as exc:
        raise ValueError(f"line {line_number}: date must begin with YYYY-MM-DD") from exc

    language = row["language"].strip().lower()
    if not language.replace("-", "").isalpha():
        raise ValueError(f"line {line_number}: language must be a language code")
    dataset_id = row["id"].strip()
    return {
        "id": "fineweb2-" + stable_id(dataset_id),
        "source": "HuggingFaceFW/fineweb-2",
        "source_group": parsed_url.hostname.lower(),
        "license": "ODC-By-1.0-claimed",
        "license_claim": LICENSE_CLAIM,
        "source_url": row["url"],
        "source_dataset_url": DATASET_URL,
        "source_date": source_date,
        "language": language,
        "language_script": row.get("language_script", ""),
        "text": row["text"].strip(),
        "rights_basis": RIGHTS_BASIS,
        "rights_confirmed": False,
        "privacy_reviewed": False,
        "quality_reviewed": False,
        "approval_granted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Local FineWeb2 JSONL export")
    parser.add_argument("--output", default="data/fineweb2-candidates.jsonl")
    parser.add_argument("--summary", default="data/fineweb2-candidates-summary.json")
    parser.add_argument("--languages", default="",
                        help="Optional comma-separated ISO language codes, e.g. eng,swa,amh")
    parser.add_argument("--max-per-language", type=int, default=0,
                        help="Optional deterministic cap per language; 0 keeps all candidates")
    args = parser.parse_args()

    if args.max_per_language < 0:
        parser.error("--max-per-language must be zero or greater")

    selected_languages = {
        item.strip().lower() for item in args.languages.split(",") if item.strip()
    }
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    rows = []
    seen_ids = set()
    seen_text = set()
    skipped_languages = 0
    duplicate_records = 0

    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"line {line_number}: each JSONL row must be an object")
            candidate = candidate_from_row(raw, line_number)
        except (json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(f"Invalid FineWeb2 record: {exc}") from exc

        if selected_languages and candidate["language"] not in selected_languages:
            skipped_languages += 1
            continue
        if candidate["id"] in seen_ids:
            duplicate_records += 1
            continue
        fingerprint = hashlib.sha256(
            " ".join(candidate["text"].split()).casefold().encode("utf-8")
        ).hexdigest()
        if fingerprint in seen_text:
            duplicate_records += 1
            continue
        seen_ids.add(candidate["id"])
        seen_text.add(fingerprint)
        rows.append(candidate)

    if args.max_per_language:
        grouped = {}
        for row in rows:
            grouped.setdefault(row["language"], []).append(row)
        rows = []
        for language, language_rows in sorted(grouped.items()):
            language_rows.sort(key=lambda item: stable_id(item["id"]))
            rows.extend(language_rows[:args.max_per_language])

    if not rows:
        raise SystemExit(
            "No eligible FineWeb2 records remain after filtering; no candidate files were written."
        )

    rows.sort(key=lambda item: (item["language"], item["id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8"
    )
    summary = {
        "schema": "dream-fineweb2-candidates-v1",
        "input": str(input_path),
        "output": str(output_path),
        "dataset": "HuggingFaceFW/fineweb-2",
        "dataset_url": DATASET_URL,
        "dataset_license_claim": LICENSE_CLAIM,
        "records": len(rows),
        "skipped_unselected_languages": skipped_languages,
        "duplicate_records_removed": duplicate_records,
        "language_counts": dict(sorted(Counter(row["language"] for row in rows).items())),
        "review_status": "UNAPPROVED_CANDIDATES_ONLY",
        "all_review_flags_false": all(
            row["rights_confirmed"] is False
            and row["privacy_reviewed"] is False
            and row["quality_reviewed"] is False
            and row["approval_granted"] is False
            for row in rows
        ),
        "required_next_step": (
            "Human reviewers must verify source-page rights, dataset terms, attribution, "
            "privacy, personal data, opt-outs and content quality before approval. "
            "This importer does not approve records or train DREAM."
        )
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
