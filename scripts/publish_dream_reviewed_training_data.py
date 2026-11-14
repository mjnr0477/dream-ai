#!/usr/bin/env python3
"""Validate and upload an already-reviewed DREAM SFT dataset to the private model store."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys

FILES = ("train.jsonl", "validation.jsonl", "manifest.json", "audit-gate.json")
DEFAULT_REPO = "msjnr0477/dream-ai-adapter"
DEFAULT_PATH = "training-data/active"

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def safe_repo_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in value:
        raise ValueError("Repository path must be a safe relative path without parent traversal.")
    return path.as_posix()

def validate_source(source: Path) -> dict:
    if not source.is_dir():
        raise ValueError(f"Dataset source directory does not exist: {source}")
    paths = {name: source / name for name in FILES}
    missing = [name for name, path in paths.items() if not path.is_file() or path.is_symlink()]
    if missing:
        raise ValueError("Missing regular dataset files: " + ", ".join(missing))
    manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    gate = json.loads(paths["audit-gate.json"].read_text(encoding="utf-8"))
    if manifest.get("schema") != "dream-sft-splits-v1":
        raise ValueError("Unsupported DREAM SFT manifest schema.")
    if gate.get("schema") != "dream-sft-audit-gate-v1" or gate.get("passed") is not True:
        raise ValueError("The human-reviewed DREAM audit gate has not passed.")
    dataset_hash = manifest.get("dataset_sha256")
    if not isinstance(dataset_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", dataset_hash):
        raise ValueError("Manifest dataset_sha256 is missing or invalid.")
    if gate.get("dataset_sha256") != dataset_hash:
        raise ValueError("Audit gate is not bound to this exact dataset manifest.")
    split_report = {}
    for split in ("train", "validation"):
        actual = sha256_file(paths[f"{split}.jsonl"])
        expected = manifest.get("splits", {}).get(split, {}).get("sha256")
        approved = gate.get("splits", {}).get(split, {}).get("sha256")
        if actual != expected or actual != approved:
            raise ValueError(f"{split}.jsonl does not match the manifest and audit gate.")
        split_report[split] = {"sha256": actual, "bytes": paths[f"{split}.jsonl"].stat().st_size}
    return {
        "dataset_sha256": dataset_hash,
        "files": {name: {"sha256": sha256_file(path), "bytes": path.stat().st_size} for name, path in paths.items()},
        "splits": split_report,
    }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Directory containing the four reviewed files.")
    parser.add_argument("--repo", default=os.environ.get("DREAM_ADAPTER_REPO", DEFAULT_REPO))
    parser.add_argument("--path-in-repo", default=DEFAULT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        source = Path(args.source).expanduser().resolve()
        repo_path = safe_repo_path(args.path_in_repo)
        report = validate_source(source)
        print(json.dumps({"status": "VALIDATED", "repo": args.repo, "path_in_repo": repo_path, **report}, indent=2))
        if args.dry_run:
            print("DRY RUN: no upload performed.")
            return 0
        token = os.environ.get("HF_TOKEN", "").strip()
        if not token:
            raise ValueError("Set HF_TOKEN with write access to the private DREAM adapter repository.")
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        info = api.repo_info(repo_id=args.repo, repo_type="model", token=token)
        if getattr(info, "private", None) is not True:
            raise ValueError("Refusing to upload private training data to a repository that is not private.")
        for name in FILES:
            api.upload_file(
                path_or_fileobj=str(source / name),
                path_in_repo=f"{repo_path}/{name}",
                repo_id=args.repo,
                repo_type="model",
                commit_message=f"Store reviewed DREAM training dataset file: {name}",
                token=token,
            )
        print(f"UPLOAD COMPLETE: reviewed files stored under {args.repo}/{repo_path}/")
        print("No model weights were changed and no GPU job was started.")
        return 0
    except Exception as exc:
        print(f"DATASET MIGRATION BLOCKED: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
