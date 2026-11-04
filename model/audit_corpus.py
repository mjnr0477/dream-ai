"""Privacy-conscious pre-ingestion audit for DREAM JSONL corpora.

This tool reports metadata, duplicate-content and likely-PII findings. It never
approves data, changes review flags, or includes raw text in its report.
"""
import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
KENYAN_PHONE_RE = re.compile(r"(?<!\w)(?:\+254[ .()-]?|254[ .()-]?|0)(?:7|1)\d(?:[ .()-]?\d){7}(?!\w)")
INTERNATIONAL_PHONE_RE = re.compile(
    r"(?<![\w])(?P<prefix>\+\d{1,3}|00\d{1,3})(?:[ .()-]*\d){7,14}(?!\w)"
)
ALLOWED_RIGHTS = {
    "original-authorship", "public-domain", "open-license", "explicit-permission"
}
REVIEW_FLAGS = ("rights_confirmed", "privacy_reviewed", "quality_reviewed")


def normalized_content(record):
    if isinstance(record.get("text"), str):
        return record["text"]
    messages = record.get("messages")
    if isinstance(messages, list):
        return "\n".join(
            message.get("content", "")
            for message in messages
            if isinstance(message, dict) and isinstance(message.get("content"), str)
        )
    return ""


def has_international_phone(text):
    """Detect explicit international numbers without exposing their values."""
    for match in INTERNATIONAL_PHONE_RE.finditer(text):
        prefix = match.group("prefix")
        country_code = prefix[1:] if prefix.startswith("+") else prefix[2:]
        digits = re.sub(r"\D", "", match.group())
        # Kenyan international numbers are already covered by KENYAN_PHONE_RE.
        if country_code == "254" and KENYAN_PHONE_RE.search(match.group()):
            continue
        if 8 <= len(digits) <= 15:
            return True
    return False


def valid_https(value):
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def audit(input_path, max_record_bytes=1_000_000):
    findings = []
    errors = []
    seen_ids = set()
    seen_content = {}
    source_groups = set()
    languages = Counter()
    licenses = Counter()
    record_count = 0
    pii_records = 0
    unreviewed_records = 0
    input_digest = hashlib.sha256()

    with input_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            input_digest.update(raw_line)
            if not raw_line.strip():
                continue
            record_count += 1
            if len(raw_line) > max_record_bytes:
                errors.append({"line": line_number, "code": "record_too_large"})
                continue
            try:
                item = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                errors.append({"line": line_number, "code": "invalid_json_or_utf8"})
                continue
            if not isinstance(item, dict):
                errors.append({"line": line_number, "code": "record_not_object"})
                continue

            record_id = item.get("id")
            # Never echo caller-controlled IDs: they may themselves contain PII.
            safe_id = f"line-{line_number}"
            if not isinstance(record_id, str) or not record_id.strip():
                findings.append({"line": line_number, "record_id": safe_id, "code": "missing_record_id", "severity": "error"})
            elif record_id in seen_ids:
                findings.append({"line": line_number, "record_id": safe_id, "code": "duplicate_record_id", "severity": "error"})
            seen_ids.add(record_id) if isinstance(record_id, str) else None

            group = item.get("source_group")
            if isinstance(group, str) and group.strip():
                source_groups.add(group.strip())
            else:
                findings.append({"line": line_number, "record_id": safe_id, "code": "missing_source_group", "severity": "warning"})

            for field in ("source_url", "license_url"):
                if not valid_https(item.get(field)):
                    findings.append({"line": line_number, "record_id": safe_id, "code": f"invalid_{field}", "severity": "error"})
            if item.get("rights_basis") not in ALLOWED_RIGHTS:
                findings.append({"line": line_number, "record_id": safe_id, "code": "invalid_rights_basis", "severity": "error"})
            missing_flags = [flag for flag in REVIEW_FLAGS if item.get(flag) is not True]
            if missing_flags:
                unreviewed_records += 1
                findings.append({
                    "line": line_number, "record_id": safe_id,
                    "code": "human_review_incomplete", "severity": "warning",
                    "missing_flags": missing_flags
                })

            language = item.get("language")
            license_name = item.get("license")
            if isinstance(language, str) and language.strip():
                languages[language] += 1
            if isinstance(license_name, str) and license_name.strip():
                licenses[license_name] += 1

            content = normalized_content(item)
            if not content.strip():
                findings.append({"line": line_number, "record_id": safe_id, "code": "missing_training_content", "severity": "error"})
                continue
            normalized = unicodedata.normalize("NFKC", " ".join(content.split())).casefold()
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if digest in seen_content:
                findings.append({
                    "line": line_number, "record_id": safe_id,
                    "code": "duplicate_normalized_content", "severity": "warning",
                    "duplicate_of_line": seen_content[digest]
                })
            else:
                seen_content[digest] = line_number

            if EMAIL_RE.search(content):
                findings.append({"line": line_number, "record_id": safe_id, "code": "possible_email", "severity": "review"})
                pii_records += 1
            if KENYAN_PHONE_RE.search(content):
                findings.append({"line": line_number, "record_id": safe_id, "code": "possible_kenyan_phone", "severity": "review"})
                pii_records += 1
            if has_international_phone(content):
                findings.append({"line": line_number, "record_id": safe_id, "code": "possible_international_phone", "severity": "review"})
                pii_records += 1
            if any(ord(char) < 32 and char not in "\n\r\t" for char in content):
                findings.append({"line": line_number, "record_id": safe_id, "code": "control_characters", "severity": "warning"})
            if len(content.strip()) < 40:
                findings.append({"line": line_number, "record_id": safe_id, "code": "short_content", "severity": "warning"})

    error_count = len(errors) + sum(f["severity"] == "error" for f in findings)
    return {
        "schema": "dream-corpus-audit-v1",
        "status": "review_required",
        "approval_granted": False,
        "input_sha256": input_digest.hexdigest(),
        "records": record_count,
        "unique_source_groups": len(source_groups),
        "unreviewed_records": unreviewed_records,
        "possible_pii_findings": pii_records,
        "language_counts": dict(sorted(languages.items())),
        "license_counts": dict(sorted(licenses.items())),
        "error_count": error_count,
        "finding_counts": dict(sorted(Counter(f["code"] for f in findings).items())),
        "findings": findings,
        "errors": errors,
        "limitations": [
            "This is a heuristic triage report, not a legal, privacy, consent, factuality or quality determination.",
            "Possible personal data patterns can be missed or falsely flagged; a qualified human must inspect the source and records.",
            "The report intentionally excludes raw training text and detected personal-data values.",
            "No data is approved for training by this command; the preparation pipeline's human review gates remain mandatory."
        ]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Candidate or approved JSONL file to audit")
    parser.add_argument("--out", default="eval-results/corpus-audit.json", help="JSON report path")
    parser.add_argument("--max-record-bytes", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.max_record_bytes < 1:
        parser.error("--max-record-bytes must be positive")
    source = Path(args.input)
    if not source.is_file():
        parser.error(f"input file does not exist: {source}")
    report = audit(source, args.max_record_bytes)
    atomic_json(Path(args.out), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(1 if report["error_count"] else 0)


if __name__ == "__main__":
    main()
