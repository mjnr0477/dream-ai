"""Build unapproved DREAM SFT conversation candidates from flat OASST1 messages.

Conversation paths are reconstructed from parent_id/message_id. Every message in
a path must pass the conservative OASST1 review filters. User IDs are never
exported and all DREAM approval flags remain false.
"""
import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from import_oasst1_candidates import (
    DATASET_REVISION,
    LICENSE,
    LICENSE_URL,
    SOURCE_NAME,
    SOURCE_URL,
    candidate_rejection,
    parse_source_date,
)


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


def build_conversation(leaf, rows_by_id):
    """Return a safe root-to-leaf SFT path, or a stable rejection reason."""
    if leaf.get("role") != "assistant":
        return None, "not_assistant_leaf"

    chain = []
    seen = set()
    current = leaf
    while current is not None:
        message_id = current.get("message_id")
        if not isinstance(message_id, str) or not message_id:
            return None, "missing_message_id"
        if message_id in seen:
            return None, "parent_cycle"
        seen.add(message_id)
        chain.append(current)

        parent_id = current.get("parent_id")
        if parent_id in (None, ""):
            break
        if not isinstance(parent_id, str):
            return None, "invalid_parent_id"
        current = rows_by_id.get(parent_id)
        if current is None:
            return None, "missing_parent"

    chain.reverse()
    if not chain or chain[0].get("role") != "prompter":
        return None, "conversation_root_not_prompter"

    tree_id = leaf.get("message_tree_id")
    language = leaf.get("lang")
    messages = []
    for index, row in enumerate(chain):
        expected_role = "prompter" if index % 2 == 0 else "assistant"
        if row.get("role") != expected_role:
            return None, "roles_do_not_alternate"
        if row.get("message_tree_id") != tree_id:
            return None, "cross_tree_parent"
        if row.get("lang") != language:
            return None, "mixed_language_path"
        rejection = candidate_rejection(row)
        if rejection:
            return None, "unsafe_path_" + rejection
        messages.append({
            "role": "user" if expected_role == "prompter" else "assistant",
            "content": row["text"].strip(),
        })

    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        return None, "incomplete_conversation"
    if parse_source_date(leaf.get("created_date")) is None:
        return None, "invalid_source_date"

    return messages, None


def convert_conversation(leaf, messages):
    return {
        "id": "oasst1-sft-" + leaf["message_id"],
        "source": SOURCE_NAME,
        "source_group": "oasst1-tree-" + leaf["message_tree_id"],
        "source_url": SOURCE_URL,
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "rights_basis": "open-license",
        "source_date": parse_source_date(leaf["created_date"]),
        "rights_confirmed": False,
        "privacy_reviewed": False,
        "quality_reviewed": False,
        "approval_granted": False,
        "language": leaf["lang"],
        "messages": messages,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Pinned OASST1 flat ready.messages.jsonl.gz")
    parser.add_argument("--output", default="data/oasst1-sft-candidates.jsonl")
    parser.add_argument("--report", default="data/oasst1-sft-candidate-report.json")
    parser.add_argument("--max-records", type=int, default=0, help="Optional candidate cap; 0 means no cap")
    args = parser.parse_args()
    if args.max_records < 0:
        parser.error("--max-records must be zero or positive")

    rows = list(read_rows(args.input))
    rows_by_id = {}
    duplicate_ids = set()
    for row in rows:
        message_id = row.get("message_id")
        if not isinstance(message_id, str) or not message_id:
            continue
        if message_id in rows_by_id:
            duplicate_ids.add(message_id)
        else:
            rows_by_id[message_id] = row
    for message_id in duplicate_ids:
        rows_by_id.pop(message_id, None)

    counts = Counter(input_messages=len(rows), duplicate_message_ids=len(duplicate_ids))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen_conversations = set()
    kept = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            if row.get("role") != "assistant":
                continue
            messages, reason = build_conversation(row, rows_by_id)
            if reason:
                counts["rejected_" + reason] += 1
                continue
            canonical = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
            fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if fingerprint in seen_conversations:
                counts["rejected_duplicate_conversation"] += 1
                continue
            seen_conversations.add(fingerprint)
            handle.write(json.dumps(convert_conversation(row, messages), ensure_ascii=False) + "\n")
            kept += 1
            if args.max_records and kept >= args.max_records:
                counts["stopped_at_candidate_cap"] += 1
                break

    report = {
        "dataset": "OpenAssistant/oasst1",
        "revision": DATASET_REVISION,
        "source_url": SOURCE_URL,
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "input_messages": len(rows),
        "duplicate_message_ids": len(duplicate_ids),
        "candidate_conversations": kept,
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
        "source_group_policy": "All candidate conversations from one OASST1 tree share one source_group; split by source_group, never by row.",
        "warning": (
            "These are SFT review candidates, not approved training data. Verify "
            "source terms, privacy, content quality and intended-use rights before "
            "human approval. No user IDs are exported."
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
