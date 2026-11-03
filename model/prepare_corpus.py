"""Validate licensed DREAM JSONL records and make leakage-aware text splits.

Input record example:
{"id":"lesson-001","source":"DREAM-authored lesson","source_group":"dream-math-v1","license":"CC0-1.0","rights_confirmed":true,"language":"en","text":"..."}
"""
import argparse
import hashlib
import json
from pathlib import Path


def stable_split(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


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
    seen_groups = {}

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
        if len(item["text"].strip()) < 40:
            raise SystemExit(f"Text too short at line {line_number}")
        if item["id"] in seen_ids:
            raise SystemExit(f"Duplicate record id: {item['id']}")
        normalized = " ".join(item["text"].split()).casefold()
        fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if fingerprint in seen_text:
            continue

        split = stable_split(item["source_group"])
        prior = seen_groups.setdefault(item["source_group"], split)
        if prior != split:
            raise SystemExit("Source group leakage detected")
        seen_ids.add(item["id"])
        seen_text.add(fingerprint)
        records.append({
            "id": item["id"],
            "source": item["source"],
            "source_group": item["source_group"],
            "license": item["license"],
            "language": item["language"],
            "text": item["text"].strip(),
            "split": split
        })

    if len(records) < 3:
        raise SystemExit("At least three unique, rights-confirmed records are required")

    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "dream-corpus-v1",
        "input_sha256": sha256(source),
        "records": len(records),
        "split_strategy": "deterministic source_group hash 90/5/5",
        "splits": {}
    }
    for split in ("train", "validation", "test"):
        selected = [row for row in records if row["split"] == split]
        text_path = output / f"{split}.txt"
        text_path.write_text(
            "\n\n".join(row["text"] for row in selected) + "\n",
            encoding="utf-8"
        )
        manifest["splits"][split] = {
            "records": len(selected),
            "sha256": sha256(text_path)
        }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
