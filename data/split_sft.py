"""Create a deterministic, source-group-disjoint SFT train/validation split."""
import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

ALLOWED_RIGHTS = {
    "original-authorship", "public-domain", "open-license", "explicit-permission"
}
REVIEW_FLAGS = ("rights_confirmed", "privacy_reviewed", "quality_reviewed", "approval_granted")


def validate_record(record, line_number):
    if not isinstance(record, dict):
        raise ValueError(f"Record at line {line_number} must be a JSON object")
    for field in ("id", "source", "source_group", "license", "license_url", "source_url",
                  "rights_basis", "source_date", "language"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"Missing {field} at line {line_number}")
    if record["rights_basis"] not in ALLOWED_RIGHTS:
        raise ValueError(f"Invalid rights_basis at line {line_number}")
    for field in ("source_url", "license_url"):
        parsed = urlparse(record[field])
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError(f"Invalid {field} at line {line_number}; HTTPS required")
    try:
        source_date = date.fromisoformat(record["source_date"])
    except ValueError as exc:
        raise ValueError(f"Invalid source_date at line {line_number}") from exc
    if source_date.isoformat() != record["source_date"]:
        raise ValueError(f"source_date must use YYYY-MM-DD at line {line_number}")
    for flag in REVIEW_FLAGS:
        if record.get(flag) is not True:
            raise ValueError(f"{flag} is not confirmed at line {line_number}")

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"At least one user/assistant pair is required at line {line_number}")
    normalized = []
    for index, message in enumerate(messages):
        if (
            not isinstance(message, dict)
            or message.get("role") not in {"system", "user", "assistant"}
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise ValueError(f"Invalid message at line {line_number}")
        role = message["role"]
        if role == "system" and index != 0:
            raise ValueError(f"System message must be first at line {line_number}")
        normalized.append((role, message["content"].strip()))
    dialogue = normalized[1:] if normalized[0][0] == "system" else normalized
    if len(dialogue) < 2 or len(dialogue) % 2:
        raise ValueError(f"Conversation must end with an assistant answer at line {line_number}")
    for index, (role, _) in enumerate(dialogue):
        expected = "user" if index % 2 == 0 else "assistant"
        if role != expected:
            raise ValueError(f"Conversation roles must alternate at line {line_number}")
    return normalized


def fingerprint(kind, value):
    payload = f"dream-{kind}-id-v1\0{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_temp(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".",
        suffix=".tmp", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def write_temp_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".",
        suffix=".tmp", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def split_sft(input_path, train_path, validation_path, manifest_path,
              validation_fraction=0.2, seed=42, overwrite=False):
    input_path = Path(input_path)
    train_path = Path(train_path)
    validation_path = Path(validation_path)
    manifest_path = Path(manifest_path)
    paths = [input_path.resolve(), train_path.resolve(),
             validation_path.resolve(), manifest_path.resolve()]
    if len(set(paths)) != len(paths):
        raise ValueError("Input, train, validation and manifest paths must be different")
    if not input_path.is_file():
        raise FileNotFoundError(f"Approved SFT input does not exist: {input_path}")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be greater than 0 and less than 1")
    if not overwrite and any(path.exists() for path in (train_path, validation_path, manifest_path)):
        raise FileExistsError("SFT split outputs already exist; pass overwrite=True to regenerate")

    records = []
    seen_ids = set()
    seen_dialogues = set()
    groups = {}
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}") from exc
            normalized = validate_record(record, line_number)
            record_id = record["id"]
            if record_id in seen_ids:
                raise ValueError(f"Duplicate record ID at line {line_number}")
            seen_ids.add(record_id)
            canonical = "\n".join(
                f"{role}: {' '.join(content.split()).casefold()}"
                for role, content in normalized
            )
            dialogue_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if dialogue_hash in seen_dialogues:
                raise ValueError(f"Duplicate conversation content at line {line_number}")
            seen_dialogues.add(dialogue_hash)
            records.append(record)
            groups.setdefault(record["source_group"], []).append(record)

    group_ids = sorted(groups)
    if len(group_ids) < 2:
        raise ValueError("At least two independent source groups are required for SFT splitting")
    shuffled = list(group_ids)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, min(len(shuffled) - 1, math.ceil(len(shuffled) * validation_fraction)))
    validation_groups = set(shuffled[:validation_count])
    train_groups = set(shuffled[validation_count:])
    train_records = [record for record in records if record["source_group"] in train_groups]
    validation_records = [record for record in records if record["source_group"] in validation_groups]
    if not train_records or not validation_records:
        raise ValueError("The source-group split produced an empty partition")

    temporary_paths = []
    try:
        temporary_paths.append(write_temp(train_path, train_records))
        temporary_paths.append(write_temp(validation_path, validation_records))
        train_path.parent.mkdir(parents=True, exist_ok=True)
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_paths[0], train_path)
        os.replace(temporary_paths[1], validation_path)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)

    train_fingerprints = sorted(fingerprint("source-group", group) for group in train_groups)
    validation_fingerprints = sorted(fingerprint("source-group", group) for group in validation_groups)
    if set(train_fingerprints) & set(validation_fingerprints):
        raise AssertionError("Source-group leakage detected")

    manifest = {
        "version": 1,
        "policy": "source-group-disjoint-deterministic",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "input_sha256": file_sha256(input_path),
        "train_sha256": file_sha256(train_path),
        "validation_sha256": file_sha256(validation_path),
        "record_count": len(records),
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "source_group_count": len(group_ids),
        "train_source_group_count": len(train_groups),
        "validation_source_group_count": len(validation_groups),
        "train_source_group_fingerprints": train_fingerprints,
        "validation_source_group_fingerprints": validation_fingerprints,
        "source_groups_disjoint": True
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_temp = write_temp_json(manifest_path, manifest)
    try:
        os.replace(manifest_temp, manifest_path)
    finally:
        manifest_temp.unlink(missing_ok=True)

    return {
        "records": len(records),
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "source_groups": len(group_ids),
        "train_source_groups": len(train_groups),
        "validation_source_groups": len(validation_groups),
        "source_groups_disjoint": True
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Human-approved SFT candidate JSONL")
    parser.add_argument("--train-out", default="data/sft-train.jsonl")
    parser.add_argument("--validation-out", default="data/sft-validation.jsonl")
    parser.add_argument("--manifest", default="data/sft-split-manifest.json")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate existing split outputs from the approved input")
    args = parser.parse_args()
    try:
        summary = split_sft(
            args.input, args.train_out, args.validation_out, args.manifest,
            validation_fraction=args.validation_fraction, seed=args.seed,
            overwrite=args.overwrite
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
