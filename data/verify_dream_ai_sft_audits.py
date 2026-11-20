"""Verify privacy audit reports against the exact DREAM SFT split manifest."""
from __future__ import annotations

from pathlib import Path
import json
import re

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _valid_sha256(value):
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _positive_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _require_zero_counter(report, field, split_name):
    value = report.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"invalid_{split_name}_{field}")
    if value != 0:
        raise ValueError(f"{split_name}_{field}_present")


def verify_audit_reports(manifest, train_report, validation_report):
    if not isinstance(manifest, dict) or manifest.get("schema") != "dream-sft-splits-v1":
        raise ValueError("invalid_sft_manifest")

    if not _valid_sha256(manifest.get("dataset_sha256")):
        raise ValueError("invalid_dataset_sha256")

    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("invalid_manifest_splits")

    verified = {}
    for split_name, report in (
        ("train", train_report),
        ("validation", validation_report),
    ):
        expected = splits.get(split_name)
        if not isinstance(expected, dict):
            raise ValueError(f"missing_manifest_split_{split_name}")

        expected_records = expected.get("records")
        if not _positive_integer(expected_records):
            raise ValueError(f"invalid_{split_name}_record_count")

        expected_hash = expected.get("sha256")
        if not _valid_sha256(expected_hash):
            raise ValueError(f"invalid_{split_name}_split_sha256")

        if not isinstance(report, dict) or report.get("schema") != "dream-corpus-audit-v1":
            raise ValueError(f"invalid_{split_name}_audit_report")

        # An audit is evidence for review, never an approval authority.
        if report.get("approval_granted") is not False:
            raise ValueError(f"{split_name}_audit_report_must_not_approve_data")

        report_hash = report.get("input_sha256")
        if not _valid_sha256(report_hash):
            raise ValueError(f"invalid_{split_name}_audit_hash")
        if report_hash != expected_hash:
            raise ValueError(f"stale_{split_name}_audit_hash")

        report_records = report.get("records")
        if not _positive_integer(report_records):
            raise ValueError(f"invalid_{split_name}_audit_record_count")
        if report_records != expected_records:
            raise ValueError(f"{split_name}_audit_record_count_mismatch")

        _require_zero_counter(report, "error_count", split_name)
        _require_zero_counter(report, "unreviewed_records", split_name)
        _require_zero_counter(report, "possible_pii_findings", split_name)

        findings = report.get("finding_counts")
        errors = report.get("errors")
        if not isinstance(findings, dict) or findings:
            raise ValueError(f"{split_name}_audit_findings_present")
        if not isinstance(errors, list) or errors:
            raise ValueError(f"{split_name}_audit_parse_errors_present")

        verified[split_name] = {
            "records": expected_records,
            "sha256": expected_hash,
            "possible_pii_findings": 0,
            "unreviewed_records": 0,
            "error_count": 0,
            "parse_errors": 0,
            "finding_counts": {},
            "errors": [],
        }

    return {
        "schema": "dream-sft-audit-gate-v1",
        "passed": True,
        "dataset_sha256": manifest["dataset_sha256"],
        "splits": verified,
    }


def read_json(path):
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Audit input does not exist: {source}")
    with source.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {source.name}")
    return value


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="manifest.json from the SFT splitter")
    parser.add_argument("--train-audit", required=True, help="audit report for train.jsonl")
    parser.add_argument("--validation-audit", required=True, help="audit report for validation.jsonl")
    parser.add_argument("--output", default="dream-sft-audit-gate.json")
    args = parser.parse_args()

    result = verify_audit_reports(
        read_json(args.manifest),
        read_json(args.train_audit),
        read_json(args.validation_audit),
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": result["passed"],
        "dataset_sha256": result["dataset_sha256"],
        "train_records": result["splits"]["train"]["records"],
        "validation_records": result["splits"]["validation"]["records"],
        "gate_report": str(target),
    }, indent=2))


if __name__ == "__main__":
    main()
