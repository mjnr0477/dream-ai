#!/usr/bin/env bash
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

SCRIPT_DIR="$(cd -- "$(dirname -- "\${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "\${SCRIPT_DIR}/.." && pwd)"
cd "\${REPO_DIR}"

[[ -f dream-host/notebook_worker.py ]] || die "Run this script from the checked-out DREAM AI repository."
[[ -f training/dream_native_persistent_notebook.ipynb ]] || die "The persistent native DREAM notebook is missing."

: "\${DREAM_HOST_NOTEBOOK_WORKER_TOKEN:?Set the existing DREAM_HOST_NOTEBOOK_WORKER_TOKEN in Lightning Studio private environment settings.}"
: "\${DREAM_NOTEBOOK_ARTIFACT_TOKEN:?Set the existing DREAM_HOST_ARTIFACT_TOKEN as DREAM_NOTEBOOK_ARTIFACT_TOKEN in Lightning Studio private environment settings.}"

DREAM_HOST_URL="\${DREAM_HOST_URL:-https://dream-host-live.onrender.com}"
[[ "\${DREAM_HOST_URL}" == https://* ]] || die "DREAM_HOST_URL must use HTTPS."
[[ "\${DREAM_HOST_NOTEBOOK_WORKER_TOKEN}" =~ ^[A-Za-z0-9._+=/-]{32,}$ ]] || die "The notebook worker token is missing or invalid."
[[ "\${DREAM_NOTEBOOK_ARTIFACT_TOKEN}" =~ ^[A-Za-z0-9._+=/-]{32,}$ ]] || die "The artifact token is missing or invalid."
[[ "\${DREAM_HOST_NOTEBOOK_WORKER_TOKEN}" != "\${DREAM_NOTEBOOK_ARTIFACT_TOKEN}" ]] || die "Notebook and artifact tokens must be different."

export DREAM_HOST_URL
export DREAM_HOST_NOTEBOOK_TRUSTED_MODE=1
export DREAM_HOST_NOTEBOOK_WORKER_NAME="DREAM AI Lightning Studio GPU"
export DREAM_REPO_DIR="\${REPO_DIR}"
export DREAM_REPO_BRANCH="$(git branch --show-current 2>/dev/null || printf 'dream-host-runtime')"

WORKSPACE="\${DREAM_NOTEBOOK_PERSISTENT_ROOT:-\${HOME}/dream-ai-workspace}"
export DREAM_NOTEBOOK_PERSISTENT_ROOT="\${WORKSPACE}"
export DREAM_CHECKPOINT_DIR="\${DREAM_CHECKPOINT_DIR:-\${WORKSPACE}/checkpoints}"
export DREAM_PREPARED_DIR="\${DREAM_PREPARED_DIR:-\${WORKSPACE}/data/prepared}"
export DREAM_WORK_DIR="\${DREAM_WORK_DIR:-\${WORKSPACE}/logs}"
export DREAM_APPROVED_RECORDS_PATH="\${DREAM_APPROVED_RECORDS_PATH:-\${WORKSPACE}/data/approved-records.jsonl}"
export DREAM_SFT_TRAIN_PATH="\${DREAM_SFT_TRAIN_PATH:-\${WORKSPACE}/data/sft-train.jsonl}"
export DREAM_SFT_VALIDATION_PATH="\${DREAM_SFT_VALIDATION_PATH:-\${WORKSPACE}/data/sft-validation.jsonl}"
export DREAM_NATIVE_TRAIN_STEPS="\${DREAM_NATIVE_TRAIN_STEPS:-2000}"
export DREAM_NATIVE_TRAIN_SECONDS="\${DREAM_NATIVE_TRAIN_SECONDS:-1500}"
export DREAM_NATIVE_SFT_STEPS="\${DREAM_NATIVE_SFT_STEPS:-1000}"
export DREAM_NATIVE_SFT_SECONDS="\${DREAM_NATIVE_SFT_SECONDS:-1500}"
export DREAM_HOST_NOTEBOOK_MAX_EXECUTION_SECONDS=1800
export DREAM_HOST_NOTEBOOK_MAX_KERNELS=1
export DREAM_HOST_NOTEBOOK_KERNEL_IDLE_SECONDS=86400

mkdir -p "\${DREAM_CHECKPOINT_DIR}" "\${DREAM_PREPARED_DIR}" "\${DREAM_WORK_DIR}" \
  "$(dirname -- "\${DREAM_APPROVED_RECORDS_PATH}")" \
  "$(dirname -- "\${DREAM_SFT_TRAIN_PATH}")" \
  "$(dirname -- "\${DREAM_SFT_VALIDATION_PATH}")"

if ! python -c 'import requests' >/dev/null 2>&1; then
  python -m pip install --disable-pip-version-check 'requests>=2.31,<3'
fi

python - <<'PY'
import os
import requests
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not available in this Studio. Select a GPU-backed Studio (T4 if it is offered free), "
        "then rerun this script. Do not install or pay for a different GPU here."
    )

device = torch.cuda.get_device_name(0)
memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print("DREAM GPU:", device)
print("GPU memory GiB:", round(memory_gib, 2))
print("PyTorch:", torch.__version__, "| CUDA:", torch.version.cuda)

host = os.environ["DREAM_HOST_URL"].rstrip("/")
artifact_token = os.environ["DREAM_NOTEBOOK_ARTIFACT_TOKEN"]

health = requests.get(host + "/health", timeout=(20, 120))
health.raise_for_status()
print("DREAM HOST reachable: OK")

vault = requests.get(
    host + "/api/v1/notebook/artifacts",
    headers={"Authorization": "Bearer " + artifact_token},
    timeout=(20, 120),
)
vault.raise_for_status()
artifacts = vault.json().get("artifacts")
if not isinstance(artifacts, list):
    raise SystemExit("DREAM R2 vault returned an invalid artifact inventory.")
print("DREAM R2 authentication: OK")
print("Existing R2 artifacts:", len(artifacts))
for item in artifacts:
    if isinstance(item, dict):
        print("-", item.get("name", "unnamed"), "|", item.get("size_bytes", 0), "bytes")

print("Preflight passed. Starting the DREAM NOTEBOOK worker.")
PY

exec python -u dream-host/notebook_worker.py
