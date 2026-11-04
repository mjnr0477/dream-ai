"""Run independent, auditable held-out evaluation for each language in a corpus manifest."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from evaluate import validate_language_subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="eval-results/multilingual-test.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-tokens", type=int, default=0)
    args = parser.parse_args()

    if args.batch_size < 1 or args.batches < 1:
        raise SystemExit("batch-size and batches must be positive.")
    if not 0 <= args.sample_tokens <= 256:
        raise SystemExit("sample-tokens must be between 0 and 256.")

    manifest_path = Path(args.manifest)
    root = manifest_path.parent
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "dream-corpus-v1":
            raise ValueError("Unsupported corpus manifest schema")
        test_split = manifest["splits"]["test"]
        parent_name = test_split["path"]
        language_files = test_split["language_files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid multilingual corpus manifest: {exc}") from exc

    if not isinstance(parent_name, str) or Path(parent_name).name != parent_name:
        raise SystemExit("Manifest contains an unsafe parent test path")
    if not isinstance(language_files, dict) or not language_files:
        raise SystemExit("Manifest contains no per-language held-out test files")
    parent_test_path = root / parent_name
    checkpoint_path = Path(args.checkpoint)
    reports = {}

    with tempfile.TemporaryDirectory(prefix="dream-multilingual-eval-") as temporary:
        for index, language in enumerate(sorted(language_files)):
            metadata = language_files[language]
            if not isinstance(language, str) or not language.strip() or not isinstance(metadata, dict):
                raise SystemExit("Manifest contains an invalid language entry")
            language_name = metadata.get("path")
            if not isinstance(language_name, str) or Path(language_name).name != language_name:
                raise SystemExit(f"Manifest contains an unsafe test path for {language!r}")
            language_path = root / language_name
            try:
                parent_hash = validate_language_subset(
                    manifest_path, language, language_path, parent_test_path
                )
            except ValueError as exc:
                raise SystemExit(f"Invalid held-out corpus for {language}: {exc}") from exc

            report_path = Path(temporary) / f"report-{index}.json"
            command = [
                sys.executable, str(Path(__file__).with_name("evaluate.py")),
                "--checkpoint", str(checkpoint_path),
                "--data", str(language_path),
                "--corpus-manifest", str(manifest_path),
                "--language", language,
                "--parent-test-data", str(parent_test_path),
                "--batch-size", str(args.batch_size),
                "--batches", str(args.batches),
                "--seed", str(args.seed + index),
                "--sample-tokens", str(args.sample_tokens),
                "--out", str(report_path)
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                raise SystemExit(
                    f"Evaluation failed for language {language!r}:\n"
                    + completed.stdout[-4000:] + completed.stderr[-4000:]
                )
            reports[language] = json.loads(report_path.read_text(encoding="utf-8"))
            if reports[language].get("evaluation_parent_test_sha256") != parent_hash:
                raise SystemExit(f"Evaluation parent hash mismatch for language {language!r}")

    result = {
        "schema": "dream-multilingual-evaluation-v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": next(iter(reports.values()))["checkpoint_sha256"],
        "parent_test_corpus": str(parent_test_path),
        "parent_test_corpus_sha256": test_split["sha256"],
        "languages_evaluated": sorted(reports),
        "per_language": reports,
        "aggregation": None,
        "limitations": [
            "Per-language metrics are reported separately; no aggregate is computed that could hide weak language performance.",
            "Randomly sampled context windows can vary by seed and batch count.",
            "Cross-entropy and bits per byte are not direct measures of language quality, reasoning, instruction following or factual accuracy.",
            "A small or unrepresentative test corpus cannot establish broad multilingual capability."
        ]
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    temporary_output.replace(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
