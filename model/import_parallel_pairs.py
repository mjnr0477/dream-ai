"""Create unapproved, source-isolated SFT candidates from a local parallel JSONL export.

The importer does not verify source claims, license terms, consent, or data quality.
It never approves records. A human review is required before SFT can consume them.
"""
import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path
from urllib.parse import urlparse


LANGUAGE_TAG = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def valid_https_url(value):
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc) and not parsed.username and not parsed.password


def required_text(value, field, maximum=500):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise ValueError(f"{field} must be a non-empty string of at most {maximum} characters")
    return value.strip()


def write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Local JSONL export containing a translation object")
    parser.add_argument("--source-lang", required=True)
    parser.add_argument("--target-lang", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--license", required=True, help="Unverified license label copied as a claim")
    parser.add_argument("--license-url", required=True, help="Unverified HTTPS license reference")
    parser.add_argument("--source-date", required=True, help="Claimed dataset release date, YYYY-MM-DD")
    parser.add_argument("--output-dir", default="data/parallel-candidates")
    parser.add_argument("--validation-percent", type=int, default=10)
    args = parser.parse_args()

    if not LANGUAGE_TAG.fullmatch(args.source_lang) or not LANGUAGE_TAG.fullmatch(args.target_lang):
        raise SystemExit("Language tags must be valid BCP-47-like tags.")
    if args.source_lang.casefold() == args.target_lang.casefold():
        raise SystemExit("source-lang and target-lang must be different.")
    if not valid_https_url(args.source_url) or not valid_https_url(args.license_url):
        raise SystemExit("source-url and license-url must be absolute HTTPS URLs without credentials.")
    try:
        source_date = date.fromisoformat(args.source_date)
        if source_date.isoformat() != args.source_date:
            raise ValueError
        source_name = required_text(args.source_name, "source-name")
        license_label = required_text(args.license, "license")
    except ValueError as exc:
        raise SystemExit(str(exc) if str(exc) else "source-date must use YYYY-MM-DD.") from exc
    if not 1 <= args.validation_percent <= 50:
        raise SystemExit("validation-percent must be between 1 and 50.")

    input_path = Path(args.input)
    try:
        lines = input_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise SystemExit("Could not read input as UTF-8 JSONL.") from exc

    records = []
    seen_pairs = set()
    seen_sources = {}
    duplicate_pairs_removed = 0
    group_basis_counts = {"source_group": 0, "story_id": 0}

    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON at line {line_number}.") from exc
        if not isinstance(item, dict):
            raise SystemExit(f"Record at line {line_number} must be a JSON object.")
        translations = item.get("translation")
        if not isinstance(translations, dict):
            raise SystemExit(f"Record at line {line_number} needs a translation object.")
        try:
            source_text = required_text(translations.get(args.source_lang), "source translation", 20000)
            target_text = required_text(translations.get(args.target_lang), "target translation", 20000)
        except ValueError as exc:
            raise SystemExit(f"Invalid translation at line {line_number}: {exc}") from exc

        raw_group = item.get("source_group")
        basis = "source_group"
        if not isinstance(raw_group, str) or not raw_group.strip():
            raw_group = item.get("story_id")
            basis = "story_id"
        if not isinstance(raw_group, (str, int)) or isinstance(raw_group, bool) or not str(raw_group).strip():
            raise SystemExit(
                f"Missing source_group/story_id at line {line_number}; independent splits require grouping metadata."
            )
        # Hash external grouping values so arbitrary identifiers are not copied into candidate files.
        group_key = digest(str(raw_group).strip())
        group_id = f"group-{group_key[:24]}"

        source_normalized = " ".join(source_text.split()).casefold()
        target_normalized = " ".join(target_text.split()).casefold()
        previous_target = seen_sources.get(source_normalized)
        if previous_target is not None and previous_target != target_normalized:
            raise SystemExit(f"Conflicting translations for normalized source text at line {line_number}.")
        seen_sources[source_normalized] = target_normalized
        pair_key = digest(source_normalized + "\0" + target_normalized)
        if pair_key in seen_pairs:
            duplicate_pairs_removed += 1
            continue
        seen_pairs.add(pair_key)
        group_basis_counts[basis] += 1

        records.append({
            "id": f"parallel-{pair_key[:24]}",
            "source": source_name,
            "source_group": group_id,
            "source_url": args.source_url,
            "license": license_label,
            "license_url": args.license_url,
            "rights_basis": "unverified",
            "source_date": args.source_date,
            "rights_confirmed": False,
            "privacy_reviewed": False,
            "quality_reviewed": False,
            "language": f"{args.source_lang}+{args.target_lang}",
            "messages": [
                {"role": "user", "content": (
                    f"Translate the following sentence from {args.source_lang} "
                    f"to {args.target_lang}. Return only the translation:\n\n{source_text}"
                )},
                {"role": "assistant", "content": target_text},
            ],
        })

    if len(records) < 2:
        raise SystemExit("At least two unique parallel pairs are required.")
    groups = sorted({row["source_group"] for row in records})
    if len(groups) < 2:
        raise SystemExit("At least two distinct source groups are required for train/validation separation.")

    ordered_groups = sorted(groups, key=digest)
    validation_count = max(1, round(len(ordered_groups) * args.validation_percent / 100))
    validation_count = min(validation_count, len(ordered_groups) - 1)
    validation_groups = set(ordered_groups[:validation_count])
    train_records = [row for row in records if row["source_group"] not in validation_groups]
    validation_records = [row for row in records if row["source_group"] in validation_groups]
    if not train_records or not validation_records:
        raise SystemExit("Could not create non-empty source-isolated candidate splits.")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "sft-train-candidates.jsonl"
    validation_path = output / "sft-validation-candidates.jsonl"
    summary_path = output / "import-summary.json"
    write_jsonl(train_path, train_records)
    write_jsonl(validation_path, validation_records)

    def file_hash(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    summary = {
        "schema": "dream-parallel-candidates-v2",
        "source_name": source_name,
        "source_url": args.source_url,
        "license_claim": license_label,
        "license_url_claim": args.license_url,
        "source_date_claim": args.source_date,
        "source_language": args.source_lang,
        "target_language": args.target_lang,
        "input_pairs": len(seen_pairs) + duplicate_pairs_removed,
        "unique_pairs": len(records),
        "duplicate_pairs_removed": duplicate_pairs_removed,
        "source_groups": len(groups),
        "source_group_basis_counts": group_basis_counts,
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "train_sha256": file_hash(train_path),
        "validation_sha256": file_hash(validation_path),
        "review_status": "human_review_required",
        "review_flags_default": {
            "rights_confirmed": False,
            "privacy_reviewed": False,
            "quality_reviewed": False,
        },
        "approval_granted": False,
        "warnings": [
            "Source, license, URL and date are caller-supplied claims and have not been independently verified.",
            "rights_basis is unverified; this tool does not determine whether a license permits model training.",
            "Review license terms, attribution, provenance, consent, privacy and translation quality before changing review flags.",
            "Grouping identifiers are hashed; this reduces accidental identifier disclosure but does not prove source independence.",
        ],
    }
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_summary.replace(summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
