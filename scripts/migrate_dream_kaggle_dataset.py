#!/usr/bin/env python3
"""One-time migration of DREAM's reviewed Kaggle dataset into its private model store.

No Kaggle notebook or Kaggle GPU is started. Once the files are present in the
private model repository, future training jobs no longer need Kaggle credentials.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = ("train.jsonl", "validation.jsonl", "manifest.json", "audit-gate.json")
DEFAULT_REPO = "msjnr0477/dream-ai-adapter"
DEFAULT_PATH = "training-data/active"

def run(command: list[str], *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE if capture else None, stderr=subprocess.PIPE if capture else None)
    except subprocess.CalledProcessError as exc:
        parts = [part.strip() for part in (exc.stdout, exc.stderr) if part and part.strip()]
        details = "\n".join(parts)
        token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
        if token and details:
            details = details.replace(token, "[REDACTED]")
        raise RuntimeError("Migration command failed: " + " ".join(command[:3]) + (". " + details[-2500:] if details else ".")) from exc

def configure_kaggle_credentials() -> tuple[str, str]:
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip() or os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not username or not key:
        raise ValueError("The source data is not in the private model store yet. Add temporary GitHub Actions secrets KAGGLE_USERNAME and KAGGLE_KEY (or KAGGLE_API_TOKEN) for the one-time dataset migration.")
    os.environ["KAGGLE_API_TOKEN"] = key
    return username, key

def validate_dataset_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", value.strip()):
        raise ValueError("DREAM_AI_KAGGLE_DATASET must use owner/dataset-slug form.")
    return value.strip()

def verify_private_dataset(metadata: dict, username: str, dataset_id: str) -> None:
    owner = dataset_id.split("/", 1)[0]
    if owner.lower() != username.lower():
        raise RuntimeError("The reviewed dataset must be owned by the authenticated Kaggle account.")
    info = metadata.get("info", metadata)
    privacy = info.get("isPrivate") if isinstance(info, dict) else None
    if privacy is not True and str(privacy).strip().lower() not in ("true", "1", "yes"):
        raise RuntimeError("Kaggle did not confirm the reviewed dataset is private; migration refused.")

def fetch_metadata(dataset_id: str, username: str, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "datasets", "metadata", dataset_id, "--path", str(directory)])
    path = directory / "dataset-metadata.json"
    if not path.is_file():
        raise RuntimeError("Kaggle did not provide dataset metadata.")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    verify_private_dataset(metadata, username, dataset_id)
    return metadata

def private_dataset_refs(csv_text: str, username: str) -> list[str]:
    refs = []
    for row in csv.DictReader(csv_text.splitlines()):
        ref = next((row.get(key, "") for key in ("ref", "Ref", "datasetRef", "dataset_ref") if row.get(key)), "").strip()
        if ref and "/" in ref and ref.split("/", 1)[0].lower() == username.lower():
            refs.append(ref)
    return sorted(set(refs))

def inventory_missing(csv_text: str) -> list[str]:
    rows = list(csv.DictReader(csv_text.splitlines()))
    if not rows:
        raise ValueError("Kaggle returned an empty dataset file inventory.")
    names = []
    for row in rows:
        value = next((row.get(key) for key in ("name", "Name", "fileName", "filename") if row.get(key)), "")
        if value:
            names.append(Path(value).name)
    duplicates = [name for name in REQUIRED_FILES if names.count(name) > 1]
    if duplicates:
        raise ValueError("The selected dataset contains duplicate required filenames: " + ", ".join(duplicates))
    return [name for name in REQUIRED_FILES if names.count(name) != 1]

def choose_dataset(username: str, explicit_id: str | None, temp_root: Path) -> str:
    if explicit_id:
        dataset_id = validate_dataset_id(explicit_id)
        fetch_metadata(dataset_id, username, temp_root / "selected-metadata")
        listing = run(["kaggle", "datasets", "files", dataset_id, "--csv", "--page-size", "200"]).stdout
        missing = inventory_missing(listing)
        if missing:
            raise RuntimeError("Selected private dataset is missing: " + ", ".join(missing))
        return dataset_id

    listing = run(["kaggle", "datasets", "list", "--mine", "--csv", "--page-size", "100"]).stdout
    candidates = []
    for index, dataset_id in enumerate(private_dataset_refs(listing, username)):
        try:
            fetch_metadata(dataset_id, username, temp_root / f"candidate-{index}-metadata")
            files = run(["kaggle", "datasets", "files", dataset_id, "--csv", "--page-size", "200"]).stdout
            if not inventory_missing(files):
                candidates.append(dataset_id)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError):
            continue
    if len(candidates) != 1:
        if not candidates:
            raise RuntimeError("No private dataset with exactly one copy of all four reviewed files was found. Set DREAM_AI_KAGGLE_DATASET if the intended dataset is present under a different listing.")
        raise RuntimeError("More than one eligible private dataset was found. Set DREAM_AI_KAGGLE_DATASET to the exact owner/dataset-slug.")
    return candidates[0]

def already_migrated(repo: str, path: str, token: str) -> bool:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    info = api.repo_info(repo_id=repo, repo_type="model", token=token)
    if getattr(info, "private", None) is not True:
        raise RuntimeError("The DREAM adapter/model repository must be private before storing training data.")
    files = set(api.list_repo_files(repo_id=repo, repo_type="model", token=token))
    present = {name for name in REQUIRED_FILES if f"{path}/{name}" in files}
    if len(present) == len(REQUIRED_FILES):
        print(f"REVIEWED_DATA_MIGRATION: already complete at {repo}/{path}; Kaggle was not contacted.")
        return True
    return False

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("DREAM_ADAPTER_REPO", DEFAULT_REPO))
    parser.add_argument("--path-in-repo", default=os.environ.get("DREAM_TRAINING_DATASET_PATH", DEFAULT_PATH))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        path = args.path_in_repo.strip("/")
        if not path or any(part in ("", ".", "..") for part in path.split("/")) or "\\" in path:
            raise ValueError("DREAM_TRAINING_DATASET_PATH must be a safe relative path.")
        token = os.environ.get("HF_TOKEN", "").strip()
        if not token:
            raise ValueError("HF_TOKEN is required to verify the private model repository.")
        if already_migrated(args.repo, path, token):
            return 0
        if args.check_only:
            print("REVIEWED_DATA_MIGRATION: needed; no Kaggle command or upload was run.")
            return 0
        username, _ = configure_kaggle_credentials()
        explicit_id = os.environ.get("DREAM_AI_KAGGLE_DATASET", "").strip() or None
        with tempfile.TemporaryDirectory(prefix="dream-ai-reviewed-data-migration-") as temporary:
            work = Path(temporary)
            dataset_id = choose_dataset(username, explicit_id, work)
            download_dir = work / "download"
            download_dir.mkdir(parents=True, exist_ok=True)
            run(["kaggle", "datasets", "download", dataset_id, "--path", str(download_dir), "--unzip", "--force"] )
            matches: dict[str, list[Path]] = {name: [] for name in REQUIRED_FILES}
            root = download_dir.resolve()
            for candidate in download_dir.rglob("*"):
                if candidate.is_file() and not candidate.is_symlink() and candidate.name in matches:
                    resolved = candidate.resolve()
                    if root in resolved.parents:
                        matches[candidate.name].append(resolved)
            duplicates = [name for name, items in matches.items() if len(items) > 1]
            missing = [name for name, items in matches.items() if len(items) == 0]
            if duplicates or missing:
                raise RuntimeError("Downloaded dataset file check failed. Missing: " + (", ".join(missing) or "none") + "; duplicated: " + (", ".join(duplicates) or "none"))
            staging = work / "reviewed-export"
            staging.mkdir()
            for name, items in matches.items():
                shutil.copy2(items[0], staging / name)
            uploader = ROOT / "scripts" / "publish_dream_reviewed_training_data.py"
            run([sys.executable, str(uploader), "--source", str(staging), "--repo", args.repo, "--path-in-repo", path], capture=False)
            print("REVIEWED_DATA_MIGRATION: COMPLETE")
            print("The reviewed dataset is now in the private model store. Future Lightning training jobs do not use Kaggle.")
        return 0
    except Exception as exc:
        print(f"REVIEWED_DATA_MIGRATION_BLOCKED: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
