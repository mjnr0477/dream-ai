"""Convert a local parallel-corpus JSONL export into unapproved SFT candidates.

This importer never marks rights, privacy or quality as reviewed. It creates
candidate files only; a human must verify the dataset terms and every record
before changing the review flags and passing the data to model/sft.py.
"""
import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from urllib.parse import urlparse


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_https_url(value):
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Local JSONL export with a translation object")
    parser.add_argument("--source-lang", required=True, help="Translation object key, e.g. dav")
    parser.add_argument("--target-lang", required=True, help="Translation object key, e.g. swa")
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--license-url", required=True)
    parser.add_argument("--source-date", required=True, help="Dataset release date in YYYY-MM-DD")
    parser.add_argument("--output-dir", default="data/parallel-candidates")
    parser.add_argument("--validation-percent", type=int, default=10)
    args = parser.parse_args()

    if args.source_lang == args.target_lang:
        raise SystemExit("source-lang and target-lang must be different.")
    if not valid_https_url(args.source_url) or not valid_https_url(args.license_url):
        raise SystemExit("source-url and license-url must be absolute HTTPS URLs.")
    try:
        parsed_date = date.fromisoformat(args.source_date)
        if parsed_date.isoformat() != args.source_date:
            raise ValueError
    except ValueError as exc:
        raise SystemExit("source-date must use YYYY-MM-DD.") from exc
    if not 1 <= args.validation_percent <= 50:
        raise SystemExit("validation-percent must be between 1 and 50.")

    input_path = Path(args.input)
    records = []
    seen_pairs = set()
    seen_sources = {}
    seen_ids = set()
    duplicate_pairs_removed = 0
    group_basis_counts = {"source_group": 0, "story_id": 0, "pair_content_hash": 0}

    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON at line {line_number}: {exc}") from exc
        if not isinstance(item, dict):
            raise SystemExit(f"Record at line {line_number} must be a JSON object.")
        translations = item.get("translation")
        if not isinstance(translations, dict):
            raise SystemExit(f"Record at line {line_number} is missing a translation object.")
        source_text = translations.get(args.source_lang)
        target_text = translations.get(args.target_lang)
        if not isinstance(source_text, str) or not source_text.strip():
            raise SystemExit(f"Record at line {line_number} is missing source language {args.source_lang}.")
        if not isinstance(target_text, str) or not target_text.strip():
            raise SystemExit(f"Record at line {line_number} is missing target language {args.target_lang}.")

        source_normalized = " ".join(source_text.split()).casefold()
        target_normalized = " ".join(target_text.split()).casefold()
        pair_fingerprint = hashlib.sha256(
            (source_normalized + "\0" + target_normalized).encode("utf-8")
        ).hexdigest()
        previous_target = seen_sources.get(source_normalized)
        if previous_target is not None and previous_target != target_normalized:
            raise SystemExit(
                f"Conflicting translations for the same normalized source sentence at line {line_number}."
            )
        seen_sources[source_normalized] = target_normalized
        if pair_fingerprint in seen_pairs:
            duplicate_pairs_removed += 1
            continue
        seen_pairs.add(pair_fingerprint)

        raw_group = item.get("source_group")
        if isinstance(raw_group, str) and raw_group.strip():
            source_group = raw_group.strip()
            group_basis = "source_group"
        else:
            raw_story = item.get("story_id")
            if isinstance(raw_story, (str, int)) and str(raw_story).strip():
                source_group = f"story-{str(raw_story).strip()}"
                group_basis = "story_id"
            else:
                source_group = f"pair-{pair_fingerprint}"
                group_basis = "pair_content_hash"
        group_basis_counts[group_basis] += 1

        raw_id = item.get("id")
        record_id = str(raw_id).strip() if raw_id is not None and str(raw_id).strip() else f"parallel-{pair_fingerprint[:24]}"
        if record_id in seen_ids:
            raise SystemExit(f"Duplicate parallel record id: {record_id}")
        seen_ids.add(record_id)
        records.append({
            "id": record_id,
            "source": args.source_name,
            "source_group": source_group,
            "source_url": args.source_url,
            "license": args.license,
            "license_url": args.license_url,
            "rights_basis": "open-license",
            "source_date": args.source_date,
            "rights_confirmed": False,
            "privacy_reviewed": False,
            "quality_reviewed": False,
            "language": f"{args.source_lang}+{args.target_lang}",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Translate the following sentence from {args.source_lang} "
                        f"to {args.target_lang}. Return only the translation:\n\n{source_text.strip()}"
                    )
                },
                {"role": "assistant", "content": target_text.strip()}
            ]
        })

    if len(records) < 2:
        raise SystemExit("At least two unique parallel pairs are required.")
    groups = sorted({record["source_group"] for record in records})
    if len(groups) < 2:
        raise SystemExit("At least two independent source groups are required for train/validation separation.")

    ordered_groups = sorted(
        groups,
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    validation_count = max(1, round(len(ordered_groups) * args.validation_percent / 100))
    validation_count = min(validation_count, len(ordered_groups) - 1)
    validation_groups = set(ordered_groups[:validation_count])

    train_records = [row for row in records if row["source_group"] not in validation_groups]
    validation_records = [row for row in records if row["source_group"] in validation_groups]
    if not train_records or not validation_records:
        raise SystemExit("Could not create non-empty source-isolated train/validation candidates.")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "sft-train-candidates.jsonl"
    validation_path = output / "sft-validation-candidates.jsonl"
    summary_path = output / "import-summary.json"

    def write_jsonl(path, rows):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8"
        )
        temporary.replace(path)

    write_jsonl(train_path, train_records)
    write_jsonl(validation_path, validation_records)
    summary = {
        "schema": "dream-parallel-candidates-v1",
        "source_name": args.source_name,
        "source_url": args.source_url,
        "license": args.license,
        "license_url": args.license_url,
        "source_date": args.source_date,
        "source_language": args.source_lang,
        "target_language": args.target_lang,
        "input_pairs": len(seen_pairs) + duplicate_pairs_removed,
        "unique_pairs": len(records),
        "duplicate_pairs_removed": duplicate_pairs_removed,
        "source_groups": len(groups),
        "source_group_basis_counts": group_basis_counts,
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "train_sha256": sha256(train_path),
        "validation_sha256": sha256(validation_path),
        "review_status": "human_review_required",
        "review_flags_default": {
            "rights_confirmed": False,
            "privacy_reviewed": False,
            "quality_reviewed": False
        },
        "warnings": [
            "These are candidate records, not approved training data.",
            "Verify license terms, attribution, source provenance, consent and privacy before review flags are changed.",
            "Pair-content hashes prevent exact duplicate pairs crossing splits but do not prove story-level independence when source_group/story_id metadata is absent."
        ]
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
