"""Safe preparation of DREAM AI opt-in exchanges for supervised fine-tuning.

This module never approves records. It only creates unapproved candidates from
v2 opt-ins, then validates and splits records that a human has already approved.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import date, datetime
from urllib.parse import urlparse


CONSENT_VERSION = "dream-learning-opt-in-v2"
SOURCE_URL = "https://dream-ai-live.onrender.com"
PERMISSION_URL = "https://dream-ai-live.onrender.com/learning-data-terms.html"
SOURCE_LABEL = "DREAM AI opt-in exchange"
LICENSE_LABEL = "DREAM AI Contributor Permission v2"
ALLOWED_RIGHTS_BASIS = {
    "original-authorship", "public-domain", "open-license", "explicit-permission"
}
REVIEW_FLAGS = (
    "rights_confirmed", "privacy_reviewed", "quality_reviewed", "approval_granted"
)
SOURCE_GROUP_HASH = re.compile(r"^[a-f0-9]{64}$")
RECORD_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
LANGUAGE_TAG = re.compile(r"^(?:und|[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*)$")


def _truthy(value):
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def _normalized(value):
    return unicodedata.normalize("NFKC", " ".join(value.split())).casefold()


def _fingerprint(kind, value):
    payload = f"dream-{kind}-id-v1\0{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_date(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing_created_at")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError as exc:
        raise ValueError("invalid_created_at") from exc
    return parsed.isoformat()


def _https_url(value, field):
    if not isinstance(value, str):
        raise ValueError(f"missing_{field}")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(f"invalid_{field}_https")
    return value


def export_optin_rows(rows):
    """Convert eligible v2 Supabase rows to unapproved SFT candidate records.

    Rows are never approved here. The four human-review flags are always false.
    The report contains counts and reason codes only; it never contains content,
    row IDs, or raw session/group identifiers.
    """
    if not isinstance(rows, (list, tuple)):
        raise TypeError("rows must be a list or tuple")

    candidates = []
    skipped = Counter()
    seen_dialogues = set()

    for row in rows:
        if not isinstance(row, dict):
            skipped["invalid_row"] += 1
            continue
        if not _truthy(row.get("consented")):
            skipped["not_consented"] += 1
            continue
        if row.get("consent_version") != CONSENT_VERSION:
            skipped["unsupported_consent_version"] += 1
            continue
        if row.get("review_status") != "pending" or row.get("status") != "pending":
            skipped["not_pending_review"] += 1
            continue

        record_id = row.get("id")
        if not isinstance(record_id, str) or not RECORD_ID.fullmatch(record_id):
            skipped["invalid_record_id"] += 1
            continue

        metadata = row.get("metadata")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = None
        if not isinstance(metadata, dict):
            skipped["invalid_metadata"] += 1
            continue

        group_hash = metadata.get("source_group_sha256")
        if not isinstance(group_hash, str) or not SOURCE_GROUP_HASH.fullmatch(group_hash):
            skipped["invalid_source_group_hash"] += 1
            continue

        prompt = row.get("prompt") or row.get("input_text")
        response = row.get("response") or row.get("target_text")
        if (
            not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 8000
            or not isinstance(response, str) or not 1 <= len(response.strip()) <= 16000
        ):
            skipped["invalid_exchange_text"] += 1
            continue
        prompt = prompt.strip()
        response = response.strip()

        try:
            source_date = _source_date(row.get("created_at"))
        except ValueError as exc:
            skipped[str(exc)] += 1
            continue

        mode = row.get("mode") if row.get("mode") in {"general", "coding", "learning"} else "general"
        language = metadata.get("language", "und")
        if not isinstance(language, str) or not LANGUAGE_TAG.fullmatch(language):
            language = "und"

        dialogue_key = hashlib.sha256(
            (mode + "\n" + _normalized(prompt) + "\n" + _normalized(response)).encode("utf-8")
        ).hexdigest()
        if dialogue_key in seen_dialogues:
            skipped["duplicate_dialogue"] += 1
            continue
        seen_dialogues.add(dialogue_key)

        candidate = {
            "id": record_id,
            "source": SOURCE_LABEL,
            "source_group": "dream-ai-session-" + group_hash,
            "license": LICENSE_LABEL,
            "source_url": SOURCE_URL,
            "license_url": PERMISSION_URL,
            "rights_basis": "explicit-permission",
            "source_date": source_date,
            "language": language,
            "rights_confirmed": False,
            "privacy_reviewed": False,
            "quality_reviewed": False,
            "approval_granted": False,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            "dream_mode": mode,
            "consent_version": CONSENT_VERSION,
        }
        candidates.append(candidate)

    return candidates, {
        "input_rows": len(rows),
        "exported": len(candidates),
        "skipped": sum(skipped.values()),
        "skipped_reasons": dict(sorted(skipped.items())),
        "approval_flags_set": False,
    }


def _validate_sft_record(record, line_number):
    if not isinstance(record, dict):
        raise ValueError(f"record {line_number}: expected a JSON object")
    for field in (
        "id", "source", "source_group", "license", "source_url",
        "license_url", "rights_basis", "source_date", "language"
    ):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"record {line_number}: missing {field}")
    if not RECORD_ID.fullmatch(record["id"]):
        raise ValueError(f"record {line_number}: invalid id")
    _https_url(record["source_url"], "source_url")
    _https_url(record["license_url"], "license_url")
    if record["rights_basis"] not in ALLOWED_RIGHTS_BASIS:
        raise ValueError(f"record {line_number}: invalid rights_basis")
    try:
        parsed_date = date.fromisoformat(record["source_date"])
        if parsed_date.isoformat() != record["source_date"]:
            raise ValueError("invalid date format")
    except ValueError as exc:
        raise ValueError(f"record {line_number}: source_date must be YYYY-MM-DD") from exc
    if not LANGUAGE_TAG.fullmatch(record["language"]):
        raise ValueError(f"record {line_number}: invalid language tag")
    for flag in REVIEW_FLAGS:
        if record.get(flag) is not True:
            raise ValueError(f"record {line_number}: {flag} must be true")

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"record {line_number}: at least one user/assistant pair is required")
    normalized = []
    for index, message in enumerate(messages):
        if (
            not isinstance(message, dict)
            or message.get("role") not in {"system", "user", "assistant"}
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise ValueError(f"record {line_number}: invalid message")
        if message["role"] == "system" and index != 0:
            raise ValueError(f"record {line_number}: system message must be first")
        normalized.append((message["role"], message["content"].strip()))

    dialogue = normalized[1:] if normalized[0][0] == "system" else normalized
    if len(dialogue) < 2 or len(dialogue) % 2:
        raise ValueError(f"record {line_number}: conversation must end with an assistant message")
    for index, (role, _) in enumerate(dialogue):
        expected = "user" if index % 2 == 0 else "assistant"
        if role != expected:
            raise ValueError(f"record {line_number}: messages must alternate user/assistant")
    if dialogue[-1][0] != "assistant":
        raise ValueError(f"record {line_number}: final message must be assistant")
    full_text = "\n".join(f"{role}: {text}" for role, text in normalized)
    if len(full_text.strip()) < 40:
        raise ValueError(f"record {line_number}: dialogue is too short")

    canonical = "\n".join(
        f"{role}: {_normalized(text)}" for role, text in normalized
    )
    return canonical


def prepare_sft_splits(records, min_source_groups=4):
    """Validate approved SFT records and create deterministic group-isolated splits."""
    if not isinstance(records, (list, tuple)):
        raise TypeError("records must be a list or tuple")
    if not isinstance(min_source_groups, int) or isinstance(min_source_groups, bool) or min_source_groups < 2:
        raise ValueError("min_source_groups must be an integer greater than or equal to 2")

    seen_ids = set()
    seen_dialogues = set()
    groups = set()
    normalized_records = []
    language_groups = {}

    for line_number, record in enumerate(records, 1):
        canonical = _validate_sft_record(record, line_number)
        record_id = record["id"]
        if record_id in seen_ids:
            raise ValueError(f"duplicate record id at record {line_number}")
        seen_ids.add(record_id)
        dialogue_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if dialogue_hash in seen_dialogues:
            raise ValueError(f"duplicate dialogue content at record {line_number}")
        seen_dialogues.add(dialogue_hash)

        group = record["source_group"].strip()
        groups.add(group)
        language_groups.setdefault(record["language"], set()).add(group)
        normalized_records.append(dict(record))

    if len(groups) < min_source_groups:
        raise ValueError(
            f"at least {min_source_groups} independent source groups are required; "
            f"found {len(groups)}"
        )

    group_order = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            ("dream-sft-validation-split-v1\0" + group).encode("utf-8")
        ).hexdigest()
    )
    validation_count = max(1, round(len(group_order) * 0.2))
    validation_count = min(validation_count, len(group_order) - 1)
    validation_groups = set(group_order[:validation_count])

    train = sorted(
        (row for row in normalized_records if row["source_group"] not in validation_groups),
        key=lambda row: row["id"]
    )
    validation = sorted(
        (row for row in normalized_records if row["source_group"] in validation_groups),
        key=lambda row: row["id"]
    )
    if not train or not validation:
        raise RuntimeError("internal error: SFT split must contain train and validation records")

    train_groups = {row["source_group"] for row in train}
    validation_groups_actual = {row["source_group"] for row in validation}
    if train_groups & validation_groups_actual:
        raise RuntimeError("internal error: source-group leakage across SFT splits")

    def encode_jsonl(items):
        return "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in items
        ).encode("utf-8")

    train_bytes = encode_jsonl(train)
    validation_bytes = encode_jsonl(validation)
    all_bytes = encode_jsonl(sorted(normalized_records, key=lambda row: row["id"]))

    def split_meta(items, content_bytes):
        split_groups = {row["source_group"] for row in items}
        return {
            "records": len(items),
            "source_groups": len(split_groups),
            "sha256": hashlib.sha256(content_bytes).hexdigest(),
            "record_ids": sorted(_fingerprint("record", row["id"]) for row in items),
            "source_group_ids": sorted(_fingerprint("source-group", group) for group in split_groups),
            "language_counts": dict(sorted(
                Counter(row["language"] for row in items).items()
            )),
        }

    coverage = {}
    for language, language_group_set in sorted(language_groups.items()):
        coverage[language] = {
            "independent_source_groups": len(language_group_set),
            "train_source_groups": len(language_group_set & train_groups),
            "validation_source_groups": len(language_group_set & validation_groups_actual),
        }

    manifest = {
        "schema": "dream-sft-splits-v1",
        "identifier_format": "sha256-domain-separated-v1",
        "dataset_sha256": hashlib.sha256(all_bytes).hexdigest(),
        "records": len(normalized_records),
        "source_groups": len(groups),
        "minimum_source_groups": min_source_groups,
        "split_strategy": "deterministic 80/20 source-group holdout; source groups never cross splits",
        "language_split_coverage": coverage,
        "splits": {
            "train": split_meta(train, train_bytes),
            "validation": split_meta(validation, validation_bytes),
        },
    }
    return train, validation, manifest


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}") from exc
            rows.append(value)
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
