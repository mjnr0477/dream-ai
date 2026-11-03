"""Report language and writing-system balance without exposing corpus text.

This is a descriptive audit only. It never approves data or claims language
proficiency; it helps reviewers spot missing languages and severe imbalance.
"""
import argparse
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

LANGUAGE_TAG = re.compile(r"^[A-Za-z]{2,3}(?:[-+][A-Za-z0-9]{2,8})*$")


def script_of(character):
    name = unicodedata.name(character, "")
    if not name:
        return "Unassigned"
    prefixes = (
        ("LATIN", "Latin"), ("ARABIC", "Arabic"), ("CYRILLIC", "Cyrillic"),
        ("GREEK", "Greek"), ("HEBREW", "Hebrew"), ("DEVANAGARI", "Devanagari"),
        ("BENGALI", "Bengali"), ("ETHIOPIC", "Ethiopic"), ("THAI", "Thai"),
        ("HANGUL", "Hangul"), ("HIRAGANA", "Hiragana"), ("KATAKANA", "Katakana"),
        ("CJK", "Han"), ("IDEOGRAPH", "Han"), ("ARMENIAN", "Armenian"),
        ("GEORGIAN", "Georgian"), ("KHMER", "Khmer"), ("LAO", "Lao"),
        ("MYANMAR", "Myanmar"), ("TAMIL", "Tamil"), ("TELUGU", "Telugu"),
        ("KANNADA", "Kannada"), ("MALAYALAM", "Malayalam"),
        ("GURMUKHI", "Gurmukhi"), ("GUJARATI", "Gujarati"),
        ("ORIYA", "Odia"), ("SINHALA", "Sinhala")
    )
    for prefix, label in prefixes:
        if name.startswith(prefix):
            return label
    if character.isspace() or unicodedata.category(character)[0] in ("P", "N", "S"):
        return "Common"
    return "Other"


def audit(path, min_records=1000, max_share=0.5):
    languages = defaultdict(lambda: {
        "records": 0, "source_groups": set(), "characters": 0,
        "scripts": Counter()
    })
    total = 0
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Record at line {line_number} must be an object.")
        language = item.get("language")
        text = item.get("text")
        group = item.get("source_group")
        if not isinstance(language, str) or not LANGUAGE_TAG.fullmatch(language.strip()):
            raise ValueError(f"Record at line {line_number} has an invalid language tag.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Record at line {line_number} has no text.")
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f"Record at line {line_number} has no source_group.")
        language = language.strip().lower()
        row = languages[language]
        row["records"] += 1
        row["source_groups"].add(group.strip())
        row["characters"] += len(text)
        row["scripts"].update(script_of(char) for char in text if not char.isspace())
        total += 1
    if total == 0:
        raise ValueError("Corpus contains no records.")

    results = {}
    warnings = []
    for language in sorted(languages):
        row = languages[language]
        share = row["records"] / total
        results[language] = {
            "records": row["records"],
            "record_share": round(share, 6),
            "source_groups": len(row["source_groups"]),
            "characters": row["characters"],
            "scripts": dict(sorted(row["scripts"].items()))
        }
        if row["records"] < min_records:
            warnings.append({
                "code": "LOW_RECORD_COUNT",
                "language": language,
                "records": row["records"],
                "minimum": min_records
            })
        if share > max_share:
            warnings.append({
                "code": "LANGUAGE_DOMINANCE",
                "language": language,
                "record_share": round(share, 6),
                "maximum": max_share
            })
        if len(row["source_groups"]) < 3:
            warnings.append({
                "code": "FEW_SOURCE_GROUPS",
                "language": language,
                "source_groups": len(row["source_groups"])
            })
    return {
        "schema": "dream-language-balance-v1",
        "total_records": total,
        "language_count": len(results),
        "languages": results,
        "warnings": warnings,
        "status": "review_required",
        "approval_granted": False,
        "limitations": [
            "Language labels are supplied metadata and are not independently verified.",
            "Unicode script detection is approximate and does not identify language or dialect.",
            "Record counts do not measure quality, representativeness or model proficiency.",
            "A human reviewer must inspect rights, privacy, quality, dialect balance and source independence."
        ]
    }


def write_json(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent,
            prefix=output.name + ".", suffix=".tmp", delete=False
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description="Audit DREAM corpus language balance.")
    parser.add_argument("--input", required=True, help="Prepared/approved JSONL corpus")
    parser.add_argument("--out", default="eval-results/language-balance.json")
    parser.add_argument("--min-records", type=int, default=1000)
    parser.add_argument("--max-share", type=float, default=0.5)
    args = parser.parse_args()
    if args.min_records < 1 or not 0 < args.max_share <= 1:
        raise SystemExit("min-records must be positive and max-share must be in (0, 1].")
    try:
        report = audit(args.input, args.min_records, args.max_share)
        write_json(args.out, report)
    except (OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(f"Could not audit language balance: {exc}") from exc
    print(json.dumps({
        "report": args.out,
        "total_records": report["total_records"],
        "language_count": report["language_count"],
        "warning_count": len(report["warnings"]),
        "approval_granted": False
    }, indent=2))


if __name__ == "__main__":
    main()
