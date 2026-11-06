#!/usr/bin/env python3
"""Export private Supabase opt-in rows as unapproved DREAM SFT candidates.

Input may be a JSON array, JSONL, or CSV export of
public.dream_ai_training_examples. This script performs no database writes.
"""
import argparse
import csv
import json
from pathlib import Path

from dream_ai_optin_pipeline import export_optin_rows, write_jsonl


def load_export(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Export file does not exist: {path}")
    if path.suffix.lower() == ".csv":
        rows = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                metadata = row.get("metadata")
                if isinstance(metadata, str) and metadata.strip():
                    try:
                        row["metadata"] = json.loads(metadata)
                    except json.JSONDecodeError:
                        row["metadata"] = None
                rows.append(row)
        return rows

    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return []
    try:
        value = json.loads(text)
        if isinstance(value, list):
            return value
        if isinstance(value, dict) and isinstance(value.get("rows"), list):
            return value["rows"]
        raise ValueError("JSON export must be an array or an object with a rows array")
    except json.JSONDecodeError:
        rows = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}") from exc
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Private Supabase JSON, JSONL or CSV export")
    parser.add_argument("--output", default="data/dream-ai-optin-candidates.jsonl")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    report_path = Path(args.report).resolve() if args.report else output.with_suffix(".report.json")
    if source in {output, report_path} or output == report_path:
        raise SystemExit("Input, candidate output and report must be separate files.")

    rows = load_export(source)
    candidates, report = export_optin_rows(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, candidates)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "input_rows": report["input_rows"],
        "exported_candidates": report["exported"],
        "skipped": report["skipped"],
        "candidate_file": str(output),
        "report_file": str(report_path),
        "approval_flags_set": False
    }, indent=2))


if __name__ == "__main__":
    main()
