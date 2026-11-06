"""Validate licensed DREAM JSONL records and create source-isolated splits.

Input record:
{"id":"lesson-001","source":"DREAM-authored lesson","source_group":"dream-math-v1","source_url":"https://example.org/dream-math","license":"CC0-1.0","license_url":"https://creativecommons.org/publicdomain/zero/1.0/","rights_basis":"original-authorship","source_date":"2026-01-15","rights_confirmed":true,"privacy_reviewed":true,"quality_reviewed":true,"language":"en","text":"..."}
"""
import argparse
import hashlib
import json
import unicodedata
from datetime import date
from pathlib import Path
from urllib.parse import urlparse, urlunsplit

from license_policy import POLICY_VERSION, classify_record_license


ALLOWED_RIGHTS_BASIS = {
    "original-authorship", "public-domain", "open-license", "explicit-permission"
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def counts(rows, field):
    result = {}
    for row in rows:
        value = row[field]
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))



IDENTIFIER_FORMAT = "sha256-domain-separated-v1"


def identifier_fingerprint(kind: str, value: str) -> str:
    """Return a stable, domain-separated digest for caller-controlled identifiers."""
    payload = f"dream-{kind}-id-v1\0{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()



def write_text_file(path: Path, rows) -> dict:
    """Write ordered record text and return verifiable corpus metadata."""
    path.write_text(
        "\n\n".join(row["text"] for row in rows) + "\n",
        encoding="utf-8"
    )
    return {
        "path": path.name,
        "records": len(rows),
        "utf8_bytes": path.stat().st_size,
        "record_ids": sorted(identifier_fingerprint("record", row["id"]) for row in rows),
        "source_group_ids": sorted({identifier_fingerprint("source-group", row["source_group"]) for row in rows}),
        "sha256": sha256(path)
    }


def _attribution_url(value: str) -> str:
    """Strip query/fragment tokens before retaining source URLs in public manifests."""
    parsed = urlparse(value)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def write_provenance_file(path: Path, rows) -> dict:
    """Write a text-free, hash-identified source/rights ledger for one split."""
    payloads = []
    for row in sorted(rows, key=lambda item: identifier_fingerprint("record", item["id"])):
        payloads.append({
            "record_id_sha256": identifier_fingerprint("record", row["id"]),
            "source_group_sha256": identifier_fingerprint("source-group", row["source_group"]),
            "source": row["source"],
            "source_url": _attribution_url(row["source_url"]),
            "license": row["license"],
            "license_url": _attribution_url(row["license_url"]),
            "rights_basis": row["rights_basis"],
            "source_date": row["source_date"],
            "language": row["language"],
            "license_policy_code": row["license_policy_code"],
            "attribution_required": row["attribution_required"],
        })
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in payloads),
        encoding="utf-8"
    )
    return {
        "path": path.name,
        "records": len(payloads),
        "sha256": sha256(path),
        "contains_training_text": False,
        "contains_raw_record_or_group_ids": False,
    }



def assign_source_group_splits(records):
    """Create deterministic source-isolated splits with multilingual coverage.

    Languages represented by at least three independent source groups are
    guaranteed to appear in train, validation and test. Rarer languages are
    reported as not yet eligible for three-way language evaluation.
    """
    languages_by_group = {}
    groups_by_language = {}
    for row in records:
        group = row["source_group"]
        language = row["language"]
        languages_by_group.setdefault(group, set()).add(language)
        groups_by_language.setdefault(language, set()).add(group)

    ordered_groups = sorted(
        languages_by_group,
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    group_count = len(ordered_groups)
    test_target = max(1, round(group_count * 0.05))
    validation_target = max(1, round(group_count * 0.05))
    if test_target + validation_target >= group_count:
        test_target = validation_target = 1

    eligible_languages = {
        language for language, groups in groups_by_language.items()
        if len(groups) >= 3
    }
    remaining_counts = {
        language: len(groups) for language, groups in groups_by_language.items()
    }
    available = set(ordered_groups)
    selected = {"test": [], "validation": []}
    group_rank = {
        group: hashlib.sha256(group.encode("utf-8")).hexdigest()
        for group in ordered_groups
    }

    def choose_coverage(split, minimum_remaining):
        uncovered = set(eligible_languages)
        while uncovered:
            candidates = []
            for group in available:
                group_languages = languages_by_group[group] & eligible_languages
                newly_covered = group_languages & uncovered
                if not newly_covered:
                    continue
                if any(
                    remaining_counts[language] <= minimum_remaining
                    for language in group_languages
                ):
                    continue
                candidates.append((
                    -len(newly_covered), group_rank[group], group,
                    newly_covered, group_languages
                ))
            if not candidates:
                raise RuntimeError(
                    "Unable to preserve train/validation/test coverage for "
                    + ", ".join(sorted(uncovered))
                )
            _, _, group, newly_covered, group_languages = min(candidates)
            selected[split].append(group)
            available.remove(group)
            for language in group_languages:
                remaining_counts[language] -= 1
            uncovered -= newly_covered

    def fill_target(split, target, minimum_remaining):
        for group in ordered_groups:
            if len(selected[split]) >= target:
                break
            if group not in available:
                continue
            group_languages = languages_by_group[group] & eligible_languages
            if any(
                remaining_counts[language] <= minimum_remaining
                for language in group_languages
            ):
                continue
            selected[split].append(group)
            available.remove(group)
            for language in group_languages:
                remaining_counts[language] -= 1

    choose_coverage("test", minimum_remaining=2)
    fill_target("test", test_target, minimum_remaining=2)
    choose_coverage("validation", minimum_remaining=1)
    fill_target("validation", validation_target, minimum_remaining=1)

    split_by_group = {group: "train" for group in available}
    for split, split_groups in selected.items():
        for group in split_groups:
            split_by_group[group] = split

    language_coverage = {}
    for language, language_groups in sorted(groups_by_language.items()):
        language_coverage[language] = {
            "independent_source_groups": len(language_groups),
            "eligible_for_three_way_evaluation": language in eligible_languages,
            "source_groups_by_split": {
                split: sum(
                    1 for group in language_groups
                    if split_by_group[group] == split
                )
                for split in ("train", "validation", "test")
            }
        }
    return split_by_group, language_coverage


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
    duplicate_text_records = 0

    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON at line {line_number}: {exc}") from exc
        if not isinstance(item, dict):
            raise SystemExit(f"Record must be a JSON object at line {line_number}")

        required = (
            "id", "source", "source_group", "license", "license_url",
            "source_url", "rights_basis", "source_date", "language", "text"
        )
        if any(not isinstance(item.get(key), str) or not item[key].strip() for key in required):
            raise SystemExit(f"Missing required metadata or text at line {line_number}")
        for field in ("source_url", "license_url"):
            parsed_url = urlparse(item[field])
            if parsed_url.scheme != "https" or not parsed_url.hostname:
                raise SystemExit(
                    f"Invalid {field} at line {line_number}; expected an absolute HTTPS URL"
                )
        if item["rights_basis"] not in ALLOWED_RIGHTS_BASIS:
            raise SystemExit(
                f"Invalid rights_basis at line {line_number}; expected one of "
                + ", ".join(sorted(ALLOWED_RIGHTS_BASIS))
            )
        license_policy = classify_record_license(item)
        if license_policy["status"] != "allowed":
            raise SystemExit(
                f"License terms not cleared at line {line_number}: {license_policy['code']}"
            )
        try:
            parsed_source_date = date.fromisoformat(item["source_date"])
            if parsed_source_date.isoformat() != item["source_date"]:
                raise ValueError("source_date must use YYYY-MM-DD")
        except ValueError as exc:
            raise SystemExit(
                f"Invalid source_date at line {line_number}; expected YYYY-MM-DD"
            ) from exc
        if item.get("rights_confirmed") is not True:
            raise SystemExit(f"Rights not confirmed at line {line_number}; refusing dataset")
        if item.get("privacy_reviewed") is not True:
            raise SystemExit(f"Privacy review not confirmed at line {line_number}; refusing dataset")
        if item.get("quality_reviewed") is not True:
            raise SystemExit(f"Quality review not confirmed at line {line_number}; refusing dataset")
        if item.get("approval_granted") is not True:
            raise SystemExit(f"Explicit training approval not confirmed at line {line_number}; refusing dataset")
        if len(item["text"].strip()) < 40:
            raise SystemExit(f"Text too short at line {line_number}")
        if item["id"] in seen_ids:
            raise SystemExit(f"Duplicate record id at line {line_number}")

        seen_ids.add(item["id"])
        normalized = unicodedata.normalize("NFKC", " ".join(item["text"].split())).casefold()
        fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if fingerprint in seen_text:
            duplicate_text_records += 1
            continue

        seen_text.add(fingerprint)
        groups.add(item["source_group"])
        records.append({
            "id": item["id"],
            "source": item["source"],
            "source_url": item["source_url"],
            "source_group": item["source_group"],
            "license": item["license"],
            "license_url": item["license_url"],
            "license_policy_code": license_policy["code"],
            "attribution_required": license_policy["attribution_required"],
            "rights_basis": item["rights_basis"],
            "source_date": item["source_date"],
            "language": item["language"],
            "text": item["text"].strip()
        })

    if len(records) < 3 or len(groups) < 3:
        raise SystemExit(
            "At least three unique records from three independent source groups are required "
            "to create leakage-aware train/validation/test splits."
        )

    split_by_group, language_split_coverage = assign_source_group_splits(records)
    group_count = len(groups)

    output.mkdir(parents=True, exist_ok=True)
    language_counts = counts(records, "language")
    license_counts = counts(records, "license")
    source_date_range = {
        "earliest": min(row["source_date"] for row in records),
        "latest": max(row["source_date"] for row in records)
    }
    manifest = {
        "schema": "dream-corpus-v1",
        "identifier_format": IDENTIFIER_FORMAT,
        "input_sha256": sha256(source),
        "records": len(records),
        "duplicate_text_records_removed": duplicate_text_records,
        "source_groups": group_count,
        "language_counts": language_counts,
        "license_counts": license_counts,
        "license_policy_version": POLICY_VERSION,
        "license_policy_counts": counts(records, "license_policy_code"),
        "rights_basis_counts": counts(records, "rights_basis"),
        "source_date_range": source_date_range,
        "split_strategy": "deterministic source-group stratification; languages with at least three independent groups are represented in train, validation and test; no group crosses splits",
        "language_split_coverage": language_split_coverage,
        "splits": {}
    }
    for split in ("train", "validation", "test"):
        selected = [row for row in records if split_by_group[row["source_group"]] == split]
        text_path = output / f"{split}.txt"
        split_file = write_text_file(text_path, selected)
        provenance_path = output / f"{split}.provenance.jsonl"
        provenance_file = write_provenance_file(provenance_path, selected)
        split_groups = sorted({row["source_group"] for row in selected})
        language_files = {}
        for language in sorted({row["language"] for row in selected}):
            language_rows = [row for row in selected if row["language"] == language]
            language_key = hashlib.sha256(language.encode("utf-8")).hexdigest()[:16]
            language_path = output / f"{split}.language-{language_key}.txt"
            language_files[language] = write_text_file(language_path, language_rows)

        manifest["splits"][split] = {
            **split_file,
            "source_groups": len(split_groups),
            "language_counts": counts(selected, "language"),
            "license_counts": counts(selected, "license"),
            "source_date_range": {
                "earliest": min(row["source_date"] for row in selected),
                "latest": max(row["source_date"] for row in selected)
            },
            "language_files": language_files,
            "provenance": provenance_file
        }
    all_split_groups = [
        set(manifest["splits"][split]["source_group_ids"])
        for split in ("train", "validation", "test")
    ]
    if any(
        all_split_groups[left] & all_split_groups[right]
        for left in range(3) for right in range(left + 1, 3)
    ):
        raise RuntimeError("Internal error: source group leaked across corpus splits")

    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
