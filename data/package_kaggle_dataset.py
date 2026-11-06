"""Package a reviewed DREAM corpus for upload as one private Kaggle Dataset.

This command does not approve records. It delegates validation and split-safety
checks to model/prepare_corpus.py, then copies the approved JSONL into a clean
directory that Kaggle can ingest as a Dataset.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_dataset(repo_root: Path, source: Path, output: Path) -> dict:
    repo_root = repo_root.resolve()
    source = source.resolve()
    output = output.resolve()
    validator = repo_root / "model" / "prepare_corpus.py"
    if not validator.is_file():
        raise FileNotFoundError(f"Canonical corpus validator not found: {validator}")
    if not source.is_file():
        raise FileNotFoundError(f"Approved JSONL file not found: {source}")
    if source == output or output in source.parents:
        raise ValueError("Output directory must not contain or replace the source file.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}. Choose a new directory; "
            "the packager never deletes existing files."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dream-kaggle-validate-") as temp:
        prepared = Path(temp) / "prepared"
        check = subprocess.run(
            [
                sys.executable, str(validator),
                "--input", str(source),
                "--output", str(prepared),
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if check.returncode:
            details = (check.stderr or check.stdout).strip()
            raise ValueError(
                "Corpus validation failed; no Kaggle package was created. "
                + (details or f"validator exited with status {check.returncode}")
            )
        manifest_path = prepared / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("Corpus validator did not produce its manifest.")
        corpus_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    stage = Path(tempfile.mkdtemp(prefix="dream-kaggle-package-", dir=output.parent))
    try:
        shutil.copy2(source, stage / "approved-records.jsonl")
        package_manifest = {
            "schema": "dream-kaggle-dataset-package-v1",
            "status": "VALIDATED_FOR_NOTEBOOK_PREFLIGHT_NOT_LEGAL_APPROVAL",
            "file": "approved-records.jsonl",
            "sha256": sha256_file(source),
            "records": corpus_manifest["records"],
            "source_groups": corpus_manifest["source_groups"],
            "language_counts": corpus_manifest["language_counts"],
            "license_counts": corpus_manifest["license_counts"],
            "corpus_manifest_schema": corpus_manifest.get("schema"),
            "note": (
                "Validation checks structure, attestations, deduplication and source-group "
                "splits. It cannot verify that human attestations are truthful or establish "
                "legal rights. The Kaggle notebook must still run its privacy/provenance audit."
            ),
        }
        (stage / "dataset-manifest.json").write_text(
            json.dumps(package_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (stage / "README.md").write_text(
            "# DREAM approved training dataset\n\n"
            "This directory is intended for one private Kaggle Dataset. "
            "It contains approved-records.jsonl, the exact filename expected "
            "by training/dream_train_kaggle.ipynb.\n\n"
            "Keep this dataset private. Do not upload personal, confidential or "
            "unlicensed material. The package validator checks schema and declared "
            "review flags; it cannot independently verify legal rights, privacy "
            "decisions or the truth of human attestations.\n",
            encoding="utf-8",
        )
        if output.exists():
            output.rmdir()
        stage.rename(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    return package_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True,
        help="Human-reviewed JSONL whose review and approval fields are all true",
    )
    parser.add_argument(
        "--output", default="build/dream-kaggle-dataset",
        help="New or empty output directory (default: build/dream-kaggle-dataset)",
    )
    args = parser.parse_args()
    try:
        result = package_dataset(
            Path(__file__).resolve().parents[1],
            Path(args.input),
            Path(args.output),
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nUpload the directory as ONE private Kaggle Dataset: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
