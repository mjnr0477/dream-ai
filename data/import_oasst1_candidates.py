"""Convert the reviewed OASST1 export into UNAPPROVED DREAM corpus candidates.

This script never marks rights, privacy, quality, or DREAM training approval complete.
A human must review the candidate records before they can pass DREAM's corpus gate.
"""
import argparse
import gzip
import json
import math
from collections import Counter
from datetime import date, datetime
from pathlib import Path

DATASET_REVISION = "56e16363c12104182b8ad8173591b4d1e31c1faa"
SOURCE_URL = (
    "https://huggingface.co/datasets/OpenAssistant/oasst1/tree/"
    + DATASET_REVISION
)
LICENSE = "Apache-2.0"
LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0"
SOURCE_NAME = "OpenAssistant OASST1 (community-contributed conversations)"

BLOCKED_LABELS = (
    "spam", "pii", "not_appropriate", "hate_speech",
    "sexual_content", "violence",
)
BLOCKED_DETOXIFY = (
    "toxicity", "severe_toxicity", "obscene",
    "identity_attack", "threat", "sexual_explicit",
)


def score(mapping, name):
    """Read a label score from OASST's dict or list representation."""
    if isinstance(mapping, dict):
        value = mapping.get(name)
    elif isinstance(mapping, list):
        value = next(
            (item for item in mapping if isinstance(item, dict) and item.get("name") == name),
            None,
        )
    else:
        value = None
    if isinstance(value, dict):
        value = value.get("value")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_source_date(value):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        normalized = value.strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        return parsed.date().isoformat()
    except ValueError:
        try:
            return date.fromisoformat(value[:10]).isoformat()
        except ValueError:
            return None


def candidate_rejection(row):
    """Return a conservative rejection reason, or None for a review candidate."""
    if row.get("deleted") is not False:
        return "deleted_or_unknown"
    if row.get("synthetic") is not False:
        return "synthetic_or_unknown"
    if row.get("review_result") is not True:
        return "not_positive_human_review"
    if row.get("tree_state") != "ready_for_export":
        return "tree_not_ready_for_export"
    if not isinstance(row.get("message_id"), str) or not row["message_id"].strip():
        return "missing_message_id"
    if not isinstance(row.get("message_tree_id"), str) or not row["message_tree_id"].strip():
        return "missing_message_tree_id"
    if not isinstance(row.get("text"), str) or len(row["text"].strip()) < 40:
        return "text_too_short"
    if not isinstance(row.get("lang"), str) or not row["lang"].strip():
        return "missing_language"
    if parse_source_date(row.get("created_date")) is None:
        return "invalid_source_date"

    labels = row.get("labels")
    pii_score = score(labels, "pii")
    if pii_score is None or pii_score > 0:
        return "pii_label_missing_or_positive"
    for name in BLOCKED_LABELS:
        value = score(labels, name)
        if value is not None and value > 0:
            return "flagged_" + name

    detoxify = row.get("detoxify")
    for name in BLOCKED_DETOXIFY:
        value = score(detoxify, name)
        if value is not None and value >= 0.2:
            return "detoxify_" + name
    return None


def convert_row(row):
    """Build a candidate record; review and approval flags intentionally stay false."""
    return {
        "id": "oasst1-message-" + row["message_id"],
        "source": SOURCE_NAME,
        "source_group": "oasst1-tree-" + row["message_tree_id"],
        "source_url": SOURCE_URL,
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "rights_basis": "open-license",
        "source_date": parse_source_date(row["created_date"]),
        "rights_confirmed": False,
        "privacy_reviewed": False,
        "quality_reviewed": False,
        "approval_granted": False,
        "language": row["lang"],
        "text": row["text"].strip(),
    }


def read_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"Expected a JSON object at line {line_number}")
            yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Pinned OASST1 ready.messages.jsonl.gz")
    parser.add_argument("--output", default="data/oasst1-candidates.jsonl")
    parser.add_argument("--report", default="data/oasst1-candidate-report.json")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    seen_ids = set()
    kept = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in read_rows(args.input):
            counts["input"] += 1
            reason = candidate_rejection(row)
            if reason:
                counts["rejected_" + reason] += 1
                continue
            message_id = row["message_id"]
            if message_id in seen_ids:
                counts["rejected_duplicate_message_id"] += 1
                continue
            seen_ids.add(message_id)
            handle.write(json.dumps(convert_row(row), ensure_ascii=False) + "\n")
            kept += 1

    report = {
        "dataset": "OpenAssistant/oasst1",
        "revision": DATASET_REVISION,
        "source_url": SOURCE_URL,
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "input_records": counts["input"],
        "candidate_records": kept,
        "rejections": {
            key.removeprefix("rejected_"): value
            for key, value in sorted(counts.items())
            if key.startswith("rejected_")
        },
        "training_approved": False,
        "human_review_required": True,
        "review_flags_written_as_false": [
            "rights_confirmed", "privacy_reviewed",
            "quality_reviewed", "approval_granted",
        ],
        "warning": (
            "Candidates are not approved training data. OASST1 contains community "
            "contributions; inspect the dataset terms, review content and privacy, "
            "and explicitly approve records before DREAM's corpus gate can accept them."
        ),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
