"""Validate licensed DREAM JSONL records and create source-isolated splits.

Input record:
{"id":"lesson-001","source":"DREAM-authored lesson","source_group":"dream-math-v1","license":"CC0-1.0","rights_confirmed":true,"language":"en","text":"..."}
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Approved JSONL records")
    parser.add_argument("--output", default="data/prepared")
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    records = []
    seen_ids = set()
    seen_text = set()
    groups = set()

    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON at line {line_number}: {exc}") from exc

        required = ("id", "source", "source_group", "license", "language", "text")
        if any(not isinstance(item.get(key), str) or not item[key].strip() for key in required):
            raise SystemExit(f"Missing required metadata or text at line {line_number}")
        if item.get("rights_confirmed") is not True:
            raise SystemExit(f"Rights not confirmed at line {line_number}; refusing dataset")
        if item.get("privacy_reviewed") is not True:
            raise SystemExit(f"Privacy review not confirmed at line {line_number}; refusing dataset")
        if item.get("quality_reviewed") is not True:
            raise SystemExit(f"Quality review not confirmed at line {line_number}; refusing dataset")
        if len(item["text"].strip()) < 40:
            raise SystemExit(f"Text too short at line {line_number}")
        if item["id"] in seen_ids:
            raise SystemExit(f"Duplicate record id: {item['id']}")

        seen_ids.add(item["id"])
        normalized = " ".join(item["text"].split()).casefold()
        fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if fingerprint in seen_text:
            continue

        seen_text.add(fingerprint)
        groups.add(item["source_group"])
        records.append({
            "id": item["id"],
            "source": item["source"],
            "source_group": item["source_group"],
            "license": item["license"],
            "language": item["language"],
            "text": item["text"].strip()
        })

    if len(records) < 3 or len(groups) < 3:
        raise SystemExit(
            "At least three unique records from three independent source groups are required "
            "to create leakage-aware train/validation/test splits."
        )

    ordered_groups = sorted(
        groups,
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    group_count = len(ordered_groups)
    test_count = max(1, round(group_count * 0.05))
    validation_count = max(1, round(group_count * 0.05))
    if test_count + validation_count >= group_count:
        test_count = validation_count = 1
    train_end = group_count - test_count - validation_count
    split_by_group = {}
    for group in ordered_groups[:train_end]:
        split_by_group[group] = "train"
    for group in ordered_groups[train_end:train_end + validation_count]:
        split_by_group[group] = "validation"
    for group in ordered_groups[train_end + validation_count:]:
        split_by_group[group] = "test"

    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "dream-corpus-v1",
        "input_sha256": sha256(source),
        "records": len(records),
        "source_groups": group_count,
        "split_strategy": "deterministic source-group hash ranking; no group crosses splits",
        "splits": {}
    }
    for split in ("train", "validation", "test"):
        selected = [row for row in records if split_by_group[row["source_group"]] == split]
        text_path = output / f"{split}.txt"
        text_path.write_text(
            "\n\n".join(row["text"] for row in selected) + "\n",
            encoding="utf-8"
        )
        split_groups = sorted({row["source_group"] for row in selected})
        manifest["splits"][split] = {
            "records": len(selected),
            "source_groups": len(split_groups),
            "source_group_ids": split_groups,
            "sha256": sha256(text_path)
        }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
