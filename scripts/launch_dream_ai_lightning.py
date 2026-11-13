#!/usr/bin/env python3
"""Submit and monitor a bounded DREAM AI adapter-training job on Lightning AI."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import os
import re
import shlex
import sys

GPU_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
DEFAULT_ADAPTER_REPO = "msjnr0477/dream-ai-adapter"

def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value

def validate_ref(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", value) or ".." in value.split("/") or value.startswith("/") or value.endswith("/") or "//" in value:
        raise ValueError("DREAM_AI_GIT_REF must be a branch name or commit SHA.")
    return value

def safe_dataset_path(value: str) -> str:
    parts = value.strip("/").split("/")
    if not value.strip("/") or any(part in ("", ".", "..") for part in parts) or "\\" in value:
        raise ValueError("DREAM_TRAINING_DATASET_PATH must be a safe repository-relative path.")
    return "/".join(parts)

def build_remote_command() -> str:
    bootstrap = """
import io, os, pathlib, re, urllib.request, zipfile
ref = os.environ["DREAM_AI_GIT_REF"]
archive_ref = ref if re.fullmatch(r"[0-9a-fA-F]{40}", ref) else "refs/heads/" + ref
url = "https://codeload.github.com/mjnr0477/dream-ai/zip/" + archive_ref
with urllib.request.urlopen(url, timeout=180) as response:
    payload = response.read()
destination = pathlib.Path("/tmp/dream-ai-source")
destination.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root not in target.parents and target != root:
            raise RuntimeError("Unsafe path in source archive.")
    archive.extractall(destination)
projects = [path for path in destination.iterdir() if path.is_dir() and path.name.startswith("dream-ai-")]
if len(projects) != 1:
    raise RuntimeError("Could not identify the downloaded DREAM AI source tree.")
print("DREAM_SOURCE_READY", projects[0])
"""
    packages = '"transformers>=4.51" "peft>=0.14" "accelerate>=1.0" "safetensors>=0.4" "huggingface_hub>=0.30"'
    return (
        "python -c " + shlex.quote(bootstrap)
        + " && cd /tmp/dream-ai-source/dream-ai-*"
        + " && python -m pip install --disable-pip-version-check --upgrade " + packages
        + " && python -u training/dream_ai_adapter_self_train_remote.py"
    )

def validate_configuration() -> dict:
    user_id = required_env("LIGHTNING_USER_ID")
    required_env("LIGHTNING_API_KEY")
    required_env("LIGHTNING_TEAMSPACE")
    required_env("HF_TOKEN")
    git_ref = validate_ref(os.environ.get("DREAM_AI_GIT_REF", "main").strip() or "main")
    dataset_path = safe_dataset_path(os.environ.get("DREAM_TRAINING_DATASET_PATH", "training-data/active"))
    seconds = int(os.environ.get("DREAM_TRAINING_SECONDS", "240"))
    if seconds < 30 or seconds > 900:
        raise ValueError("DREAM_TRAINING_SECONDS must be between 30 and 900.")
    adapter_repo = os.environ.get("DREAM_ADAPTER_REPO", DEFAULT_ADAPTER_REPO).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", adapter_repo):
        raise ValueError("DREAM_ADAPTER_REPO must use owner/repository form.")
    return {"lightning_user_id": user_id, "teamspace": required_env("LIGHTNING_TEAMSPACE"), "git_ref": git_ref, "dataset_path": dataset_path, "training_seconds": seconds, "adapter_repo": adapter_repo}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate environment without submitting a GPU job")
    args = parser.parse_args()
    try:
        config = validate_configuration()
        print("LIGHTNING_TRAINING_PREFLIGHT: PASSED")
        print(f"Teamspace: {config['teamspace']}")
        print(f"Source ref: {config['git_ref']}")
        print(f"Dataset path: {config['dataset_path']}")
        print(f"Training update budget: {config['training_seconds']} seconds")
        print(f"Adapter repository: {config['adapter_repo']}")
        if args.check:
            print("CHECK ONLY: no GPU job submitted.")
            return 0
        from lightning_sdk import Job, Machine, Teamspace
        teamspace = Teamspace(name=config["teamspace"], user=config["lightning_user_id"])
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        job_name = f"dream-adapter-{run_id}"
        job_env = {
            "HF_TOKEN": os.environ["HF_TOKEN"],
            "DREAM_AI_GIT_REF": config["git_ref"],
            "DREAM_TRAINING_DATASET_PATH": config["dataset_path"],
            "DREAM_TRAINING_SECONDS": str(config["training_seconds"]),
            "DREAM_ADAPTER_REPO": config["adapter_repo"],
            "DREAM_TRAINING_DATA_DIR": "/tmp/dream-ai-training-data",
            "DREAM_TRAINING_WORK_DIR": "/tmp/dream-ai-training-work",
        }
        job = Job.run(
            name=job_name,
            teamspace=teamspace,
            image=GPU_IMAGE,
            machine=Machine.T4,
            command=build_remote_command(),
            env=job_env,
        )
        print(f"LIGHTNING_JOB_SUBMITTED: {job.name}")
        print(f"LIGHTNING_JOB_LINK: {job.link}")
        job.wait(interval=20, timeout=55 * 60, stop_on_timeout=True)
        raw_status = job.status
        status = getattr(raw_status, "name", str(raw_status))
        print(f"LIGHTNING_JOB_STATUS: {status}")
        if "complete" not in status.lower() and "success" not in status.lower():
            try:
                logs = str(job.logs)
                if logs:
                    print("LIGHTNING_JOB_LOG_TAIL:")
                    print(logs[-12000:])
            except Exception as log_error:
                print(f"Could not retrieve job logs: {type(log_error).__name__}", file=sys.stderr)
            raise RuntimeError(f"Lightning training job did not complete successfully: {status}")
        print("Training job finished. Check the private model store for the candidate and resumable progress.")
        print("The active adapter was not modified; independent evaluation and human approval remain required.")
        return 0
    except Exception as exc:
        print(f"LIGHTNING_TRAINING_BLOCKED: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
