#!/usr/bin/env python3
"""Safely launch DREAM AI's reviewed Kaggle adapter-training notebook."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "training" / "dream_ai_adapter_self_train_kaggle.ipynb"
REQUIRED_FILES = ("train.jsonl", "validation.jsonl", "manifest.json", "audit-gate.json")
KERNEL_SLUG = "dream-ai-autonomous-adapter-training"


def validate_dataset_id(value: str) -> str:
    value = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", value):
        raise ValueError("DREAM_AI_KAGGLE_DATASET must be in owner/dataset-slug form.")
    return value


def missing_required_files(csv_text: str) -> list[str]:
    rows = list(csv.DictReader(csv_text.splitlines()))
    if not rows:
        raise ValueError("Kaggle returned no dataset file rows.")
    names = []
    for row in rows:
        name = next((row[key] for key in ("name", "Name", "fileName", "filename") if row.get(key)), None)
        if name:
            names.append(Path(name).name)
    duplicates = [name for name in REQUIRED_FILES if names.count(name) > 1]
    if duplicates:
        raise ValueError("Dataset has duplicate required basenames: " + ", ".join(duplicates))
    return [name for name in REQUIRED_FILES if names.count(name) != 1]



def verify_private_dataset(csv_text: str, username: str, dataset_id: str) -> None:
    rows = list(csv.DictReader(csv_text.splitlines()))
    matches = []
    for row in rows:
        ref = next((row[key] for key in ("ref", "Ref", "datasetRef", "dataset_ref") if row.get(key)), "")
        if ref.strip().lower() == dataset_id.strip().lower():
            matches.append(row)
    if len(matches) != 1:
        raise RuntimeError("Kaggle could not uniquely confirm this dataset under your own account.")
    owner = dataset_id.split("/", 1)[0]
    if owner.lower() != username.strip().lower():
        raise RuntimeError("The training dataset must be owned by the authenticated Kaggle account.")
    row = matches[0]
    privacy = next((row[key] for key in ("isPrivate", "is_private", "private") if key in row), None)
    if str(privacy).strip().lower() not in ("true", "1", "yes"):
        raise RuntimeError("Kaggle did not confirm that the dataset is private; refusing to attach training data.")


def private_dataset_refs(csv_text: str, username: str) -> list[str]:
    refs = []
    for row in csv.DictReader(csv_text.splitlines()):
        ref = next((row[key] for key in ("ref", "Ref", "datasetRef", "dataset_ref") if row.get(key)), "")
        privacy = next((row[key] for key in ("isPrivate", "is_private", "private") if key in row), None)
        if (
            ref and "/" in ref
            and ref.split("/", 1)[0].strip().lower() == username.strip().lower()
            and str(privacy).strip().lower() in ("true", "1", "yes")
        ):
            refs.append(ref.strip())
    return sorted(set(refs))

def build_kernel_metadata(username: str, dataset_id: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", username.strip()):
        raise ValueError("KAGGLE_USERNAME is missing or invalid.")
    return {
        "id": f"{username.strip()}/{KERNEL_SLUG}",
        "title": "DREAM AI Autonomous Adapter Training",
        "code_file": "dream_ai_adapter_self_train_kaggle.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [validate_dataset_id(dataset_id)],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def configure_kaggle_credentials(username: str, key: str) -> None:
    if not username.strip() or not key.strip():
        raise ValueError("KAGGLE_USERNAME and KAGGLE_KEY GitHub Actions secrets are required.")
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    config_dir.mkdir(parents=True, exist_ok=True)
    credentials = config_dir / "kaggle.json"
    credentials.write_text(
        json.dumps({"username": username.strip(), "key": key.strip()}), encoding="utf-8"
    )
    credentials.chmod(0o600)
    os.environ["KAGGLE_CONFIG_DIR"] = str(config_dir)


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def launch() -> int:
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    dataset_id = os.environ.get("DREAM_AI_KAGGLE_DATASET", "").strip()
    if not username or not key:
        missing = [name for name, value in (("KAGGLE_USERNAME", username), ("KAGGLE_KEY", key)) if not value]
        print("WAITING_FOR_SETUP: missing " + ", ".join(missing))
        print("No training was started and no model files were changed.")
        return 0

    if not NOTEBOOK.is_file():
        raise FileNotFoundError(f"Training notebook not found: {NOTEBOOK}")
    configure_kaggle_credentials(username, key)

    own_datasets = run(
        ["kaggle", "datasets", "list", "--mine", "--csv", "--page-size", "100"],
        capture=True,
    )
    private_refs = private_dataset_refs(own_datasets.stdout, username)
    if dataset_id:
        dataset_id = validate_dataset_id(dataset_id)
        verify_private_dataset(own_datasets.stdout, username, dataset_id)
        listing = run(
            ["kaggle", "datasets", "files", dataset_id, "--csv", "--page-size", "200"],
            capture=True,
        )
    else:
        matching = []
        for candidate_ref in private_refs:
            candidate_files = run(
                ["kaggle", "datasets", "files", candidate_ref, "--csv", "--page-size", "200"],
                capture=True,
            )
            try:
                candidate_missing = missing_required_files(candidate_files.stdout)
            except ValueError:
                continue
            if not candidate_missing:
                matching.append((candidate_ref, candidate_files))
        if not matching:
            print("WAITING_FOR_PRIVATE_REVIEWED_DATASET: no private dataset with all four required files was found.")
            print("No training was started and no model files were changed.")
            return 0
        if len(matching) > 1:
            choices = ", ".join(item[0] for item in matching)
            raise RuntimeError("More than one matching private dataset exists; set DREAM_AI_KAGGLE_DATASET to select one: " + choices)
        dataset_id, listing = matching[0]
        print("AUTO_DISCOVERED_PRIVATE_DATASET:", dataset_id)

    missing = missing_required_files(listing.stdout)
    if missing:
        raise RuntimeError("Dataset must contain exactly one copy of: " + ", ".join(missing))

    metadata = build_kernel_metadata(username, dataset_id)
    kernel_id = metadata["id"]
    with tempfile.TemporaryDirectory(prefix="dream-ai-kaggle-") as temp:
        workdir = Path(temp)
        notebook_name = "dream_ai_adapter_self_train_kaggle.ipynb"
        (workdir / notebook_name).write_bytes(NOTEBOOK.read_bytes())
        (workdir / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

        # The current Kaggle CLI runs a notebook when it is first pushed.
        # Cell 1 checks HF_TOKEN before loading a model or starting training.
        # The owner can add that Kaggle Secret after this first registration run.
        probe = subprocess.run(
            ["kaggle", "kernels", "status", kernel_id],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if probe.returncode != 0:
            print("CREATING_PRIVATE_KAGGLE_KERNEL")
            run([
                "kaggle", "kernels", "push", "--path", str(workdir),
                "--accelerator", "NvidiaTeslaT4", "--timeout", "3600",
            ])
            print("ACTION_REQUIRED: open this Kaggle notebook and add Kaggle Secret HF_TOKEN.")
            print("The initial run stops at the notebook token check if the secret is absent.")
            print("Then rerun this GitHub workflow to start training.")
            return 0

        status = probe.stdout.strip()
        print("KAGGLE_KERNEL_STATUS:", status)
        if re.search(r"running|queued|initializing", status, flags=re.IGNORECASE):
            print("SKIPPED: this Kaggle kernel is already active.")
            return 0

        print("DATASET_PREFLIGHT: PASSED")
        print("KERNEL:", kernel_id)
        print("DATASET:", dataset_id)
        print("PRODUCTION_ADAPTER: protected; notebook uploads candidates only")
        run([
            "kaggle", "kernels", "push", "--path", str(workdir),
            "--accelerator", "NvidiaTeslaT4", "--timeout", "3600",
        ])
        print("KAGGLE_TRAINING_LAUNCH: SUBMITTED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="test helper logic without Kaggle credentials")
    args = parser.parse_args()
    if args.check:
        metadata = build_kernel_metadata("test-user", "test-user/test-dataset")
        assert metadata["is_private"] == "true"
        assert metadata["dataset_sources"] == ["test-user/test-dataset"]
        sample = (
            "name,size\ntrain.jsonl,10\nvalidation.jsonl,10\n"
            "manifest.json,10\naudit-gate.json,10\n"
        )
        assert missing_required_files(sample) == []
        private_listing = "ref,isPrivate\\ntest-user/test-dataset,true\\n"
        verify_private_dataset(private_listing, "test-user", "test-user/test-dataset")
        assert private_dataset_refs(
            "ref,isPrivate\\ntest-user/test-dataset,true\\ntest-user/public-data,false\\n",
            "test-user",
        ) == ["test-user/test-dataset"]
        try:
            verify_private_dataset("ref,isPrivate\\ntest-user/test-dataset,false\\n", "test-user", "test-user/test-dataset")
        except RuntimeError:
            pass
        else:
            raise AssertionError("public dataset was accepted")
        try:
            validate_dataset_id("not-a-dataset-id")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid dataset id was accepted")
        print("LAUNCHER SELF-CHECK: PASSED")
        return 0
    try:
        return launch()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"AUTONOMOUS_LAUNCH_BLOCKED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
