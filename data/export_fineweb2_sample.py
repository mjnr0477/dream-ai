"""Stream a small deterministic FineWeb2 export for DREAM's review-candidate importer.

This exports provenance-rich rows only. It does not approve rights, privacy or quality.
Install the optional dependency with: python -m pip install datasets
"""
import argparse
import json
import math
import os
from pathlib import Path

DATASET = "HuggingFaceFW/fineweb-2"


def export_row(row, min_language_score=0.95):
    if not isinstance(row, dict):
        return None
    required = ("id", "text", "url", "date", "language", "language_script")
    if any(not isinstance(row.get(key), str) or not row[key].strip() for key in required):
        return None
    try:
        score = float(row.get("language_score"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or score < min_language_score:
        return None
    return {key: row[key] for key in (
        "id", "text", "url", "date", "language", "language_script"
    )}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", default="swh_Latn",
                        help="Comma-separated FineWeb2 language-script subsets, e.g. swa_Latn,ara_Arab")
    parser.add_argument("--max-per-subset", type=int, default=1000)
    parser.add_argument("--scan-per-subset", type=int, default=20000)
    parser.add_argument("--min-language-score", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="data/fineweb2-small-export.jsonl")
    args = parser.parse_args()
    if args.max_per_subset < 1 or args.scan_per_subset < args.max_per_subset:
        parser.error("scan-per-subset must be >= max-per-subset >= 1")
    if not 0 < args.min_language_score <= 1:
        parser.error("min-language-score must be greater than 0 and at most 1")
    subsets = [item.strip() for item in args.subsets.split(",") if item.strip()]
    if not subsets or len(set(subsets)) != len(subsets):
        parser.error("provide unique comma-separated subset names")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the optional dependency first: python -m pip install datasets") from exc

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    counts = {}
    total = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for subset_index, subset in enumerate(subsets):
                stream = load_dataset(
                    DATASET, name=subset, split="train", streaming=True
                )
                if hasattr(stream, "shuffle"):
                    stream = stream.shuffle(
                        seed=args.seed + subset_index, buffer_size=10000
                    )
                accepted = 0
                scanned = 0
                for raw in stream:
                    scanned += 1
                    row = export_row(raw, args.min_language_score)
                    if row is not None:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        accepted += 1
                        total += 1
                    if accepted >= args.max_per_subset or scanned >= args.scan_per_subset:
                        break
                counts[subset] = {
                    "scanned": scanned,
                    "exported": accepted,
                    "minimum_language_score": args.min_language_score,
                }
        if total == 0:
            raise SystemExit("No eligible rows exported; no output file was published.")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    print(json.dumps({
        "dataset": DATASET,
        "subsets": counts,
        "records": total,
        "output": str(output),
        "status": "RAW_EXPORT_REQUIRES_REVIEW",
        "next_step": (
            "Run model/import_fineweb2_candidates.py. Its output is still unapproved; "
            "review each source's rights, Common Crawl terms, opt-outs, privacy and quality."
        ),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
