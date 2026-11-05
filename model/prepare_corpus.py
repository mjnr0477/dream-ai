"""Validate licensed DREAM JSONL records and create source-isolated splits.

Input record:
{"id":"lesson-001","source":"DREAM-authored lesson","source_group":"dream-math-v1","source_url":"https://example.org/dream-math","license":"CC0-1.0","license_url":"https://creativecommons.org/publicdomain/zero/1.0/","rights_basis":"original-authorship","source_date":"2026-01-15","rights_confirmed":true,"privacy_reviewed":true,"quality_reviewed":true,"language":"en","text":"..."}
"""
import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from urllib.parse import urlparse


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
        "record_ids": sorted(row["id"] for row in rows),
        "source_group_ids": sorted({row["source_group"] for row in rows}),
        "sha256": sha256(path)
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
        "input_sha256": sha256(source),
        "records": len(records),
        "duplicate_text_records_removed": duplicate_text_records,
        "source_groups": group_count,
        "language_counts": language_counts,
        "license_counts": license_counts,
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
            "language_files": language_files
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
