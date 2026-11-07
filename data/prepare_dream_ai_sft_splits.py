#!/usr/bin/env python3
"""Create deterministic source-group-isolated SFT train/validation files."""
import argparse
import json
from pathlib import Path

from dream_ai_optin_pipeline import prepare_sft_splits, read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Human-approved SFT JSONL from review_candidates.py")
    parser.add_argument("--output-dir", default="data/dream-ai-sft-prepared")
    parser.add_argument("--min-source-groups", type=int, default=8)
    parser.add_argument("--min-validation-records", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing files in output-dir")
    args = parser.parse_args()

    source = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / "train.jsonl", output_dir / "validation.jsonl", output_dir / "manifest.json"]
    if not args.overwrite and any(path.exists() for path in targets):
        raise SystemExit("Output files already exist; choose another directory or pass --overwrite.")

    records = read_jsonl(source)
    train, validation, manifest = prepare_sft_splits(
        records,
        min_source_groups=args.min_source_groups,
        min_validation_records=args.min_validation_records,
    )
    write_jsonl(targets[0], train)
    write_jsonl(targets[1], validation)
    targets[2].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )
    print(json.dumps({
        "records": manifest["records"],
        "source_groups": manifest["source_groups"],
        "train_records": len(train),
        "validation_records": len(validation),
        "minimum_validation_records": manifest["minimum_validation_records"],
        "dataset_sha256": manifest["dataset_sha256"],
        "output_dir": str(output_dir)
    }, indent=2))


if __name__ == "__main__":
    main()
