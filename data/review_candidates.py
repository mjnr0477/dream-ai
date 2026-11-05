"""Manually review DREAM corpus candidates before training approval.

This tool never auto-approves records. Every accepted record requires explicit
human confirmation of rights, privacy, quality and training approval.
"""
import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ALLOWED_RIGHTS = {
    "original-authorship", "public-domain", "open-license", "explicit-permission"
}
REVIEW_FIELDS = (
    ("rights_confirmed", "I verified the source terms grant rights to use this exact record for DREAM training."),
    ("privacy_reviewed", "I inspected this record and found no personal, confidential or sensitive information."),
    ("quality_reviewed", "I checked this record for relevance, quality, harmful content and obvious errors."),
    ("approval_granted", "I explicitly approve this exact record for DREAM training."),
)


def record_content(record):
    text = record.get("text")
    if isinstance(text, str):
        return text.strip()
    messages = record.get("messages")
    if isinstance(messages, list):
        parts = [
            f"{item.get('role', 'unknown').upper()}: {item.get('content', '').strip()}"
            for item in messages
            if isinstance(item, dict) and isinstance(item.get("content"), str)
        ]
        return "\n\n".join(parts).strip()
    return ""


def candidate_problem(record):
    """Return a non-sensitive rejection code, or None when review can begin."""
    if not isinstance(record, dict):
        return "record_not_object"
    for field in ("id", "source", "source_group", "license", "language"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            return "missing_" + field
    for field in ("source_url", "license_url"):
        value = record.get(field)
        try:
            parsed = urlparse(value) if isinstance(value, str) else None
        except ValueError:
            parsed = None
        if not parsed or parsed.scheme != "https" or not parsed.netloc:
            return "invalid_" + field
    if record.get("rights_basis") not in ALLOWED_RIGHTS:
        return "invalid_rights_basis"
    source_date = record.get("source_date")
    try:
        parsed_date = date.fromisoformat(source_date) if isinstance(source_date, str) else None
    except ValueError:
        parsed_date = None
    if not parsed_date or parsed_date.isoformat() != source_date:
        return "invalid_source_date"
    if len(record_content(record)) < 40:
        return "missing_or_short_content"
    return None


def ask_yes_no(question, input_fn=input, output_fn=print):
    while True:
        answer = input_fn(question + " [y/N/q]: ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no", ""}:
            return False
        if answer in {"q", "quit"}:
            return None
        output_fn("Please answer y, n, or q.")


def review_candidate(record, input_fn=input, output_fn=print):
    """Return (approved_record_or_none, stop_requested, missing_review_fields)."""
    problem = candidate_problem(record)
    if problem:
        return None, False, [problem]

    output_fn("=" * 72)
    output_fn(f"Candidate: {record['id']}")
    output_fn(f"Source: {record['source']}")
    output_fn(f"License: {record['license']} — {record['license_url']}")
    output_fn(f"Source terms: {record['source_url']}")
    output_fn(f"Language: {record['language']} | Date: {record['source_date']}")
    output_fn("-" * 72)
    output_fn(record_content(record))
    output_fn("-" * 72)

    decisions = {}
    for field, question in REVIEW_FIELDS:
        answer = ask_yes_no(question, input_fn, output_fn)
        if answer is None:
            return None, True, []
        decisions[field] = answer

    missing = [field for field, _ in REVIEW_FIELDS if decisions[field] is not True]
    if missing:
        return None, False, missing

    approved = dict(record)
    approved.update(decisions)
    approved["human_reviewed_at"] = datetime.now(timezone.utc).isoformat()
    return approved, False, []


def load_ids(path):
    ids = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_number}") from exc
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                ids.add(item["id"])
    return ids


def append_jsonl(handle, payload):
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def review_file(input_path, output_path, decisions_path, limit=25,
                input_fn=input, output_fn=print):
    if limit < 1:
        raise ValueError("limit must be a positive integer")
    if not input_path.is_file():
        raise FileNotFoundError(f"Candidate file does not exist: {input_path}")
    if output_path.resolve() == decisions_path.resolve():
        raise ValueError("Approved output and decision ledger must be different files")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    approved_ids = load_ids(output_path)
    reviewed_ids = load_ids(decisions_path) | approved_ids
    summary = {"presented": 0, "approved": 0, "rejected": 0, "skipped": 0, "stopped": False}

    with input_path.open("r", encoding="utf-8") as source, \
            output_path.open("a", encoding="utf-8") as approved_file, \
            decisions_path.open("a", encoding="utf-8") as decisions_file:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid candidate JSON at line {line_number}") from exc
            record_id = record.get("id") if isinstance(record, dict) else None
            safe_id = record_id if isinstance(record_id, str) and record_id else f"line-{line_number}"
            if safe_id in reviewed_ids:
                summary["skipped"] += 1
                continue

            problem = candidate_problem(record)
            if problem:
                decision = {
                    "id": safe_id,
                    "decision": "rejected",
                    "reason": problem,
                    "reviewed_at": datetime.now(timezone.utc).isoformat()
                }
                append_jsonl(decisions_file, decision)
                reviewed_ids.add(safe_id)
                summary["rejected"] += 1
                continue

            if summary["presented"] >= limit:
                break
            summary["presented"] += 1
            approved, stop, missing = review_candidate(record, input_fn, output_fn)
            if stop:
                summary["stopped"] = True
                break
            timestamp = datetime.now(timezone.utc).isoformat()
            if approved:
                append_jsonl(approved_file, approved)
                approved_ids.add(safe_id)
                decision = {"id": safe_id, "decision": "approved", "reviewed_at": timestamp}
                summary["approved"] += 1
            else:
                decision = {
                    "id": safe_id,
                    "decision": "rejected",
                    "missing_review_fields": missing,
                    "reviewed_at": timestamp
                }
                summary["rejected"] += 1
            append_jsonl(decisions_file, decision)
            reviewed_ids.add(safe_id)

    output_fn(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Unapproved candidate JSONL file")
    parser.add_argument("--output", default="data/approved-records.jsonl",
                        help="Approved records are appended here; existing records are preserved")
    parser.add_argument("--decisions", default="data/candidate-review-decisions.jsonl",
                        help="PII-free decision ledger used to avoid reviewing the same ID twice")
    parser.add_argument("--limit", type=int, default=25,
                        help="Maximum candidates to present in one run (default: 25)")
    args = parser.parse_args()
    try:
        review_file(Path(args.input), Path(args.output), Path(args.decisions), args.limit)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
