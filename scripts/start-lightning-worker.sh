#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${DREAM_HOST_URL:=https://dream-host-live.onrender.com}"
: "${DREAM_HOST_NOTEBOOK_WORKER_TOKEN:?Set the existing DREAM_HOST_NOTEBOOK_WORKER_TOKEN in Lightning Studio environment settings}"
: "${DREAM_NOTEBOOK_ARTIFACT_TOKEN:?Set the existing DREAM_HOST_ARTIFACT_TOKEN value as DREAM_NOTEBOOK_ARTIFACT_TOKEN in Lightning Studio environment settings}"

if [[ "${DREAM_HOST_URL}" != https://* ]]; then
  echo "DREAM_HOST_URL must use HTTPS." >&2
  exit 2
fi

export DREAM_HOST_URL
export DREAM_HOST_NOTEBOOK_TRUSTED_MODE=1
export DREAM_NOTEBOOK_PERSISTENT_ROOT="${DREAM_NOTEBOOK_PERSISTENT_ROOT:-$HOME/dream-ai-workspace}"
export DREAM_REPO_BRANCH="${DREAM_REPO_BRANCH:-dream-permanent-compute}"
export DREAM_HOST_NOTEBOOK_WORKER_NAME="${DREAM_HOST_NOTEBOOK_WORKER_NAME:-DREAM AI Lightning Studio GPU}"

mkdir -p "${DREAM_NOTEBOOK_PERSISTENT_ROOT}"

if ! python -c 'import requests' >/dev/null 2>&1; then
  python -m pip install --disable-pip-version-check requests
fi

python - <<'PY'
import os
import sys
import requests
import torch

host = os.environ["DREAM_HOST_URL"].rstrip("/")
worker_token = os.environ["DREAM_HOST_NOTEBOOK_WORKER_TOKEN"]
artifact_token = os.environ["DREAM_NOTEBOOK_ARTIFACT_TOKEN"]

if len(worker_token) < 32 or len(artifact_token) < 32:
    raise SystemExit("A configured DREAM worker/artifact token is missing or too short.")

if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is not visible. Select a GPU machine for this Studio, then start this script again.")

device = torch.cuda.get_device_name(0)
memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print("GPU detected:", device)
print("GPU memory GiB:", round(memory_gib, 2))
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)

health = requests.get(host + "/health", timeout=(10, 30))
health.raise_for_status()
print("DREAM HOST health: OK")

vault = requests.get(
    host + "/api/v1/notebook/artifacts",
    headers={"Authorization": "Bearer " + artifact_token},
    timeout=(10, 60),
)
vault.raise_for_status()
items = vault.json().get("artifacts", [])
print("DREAM R2 vault: authenticated")
print("Existing R2 artifacts:", len(items))
for item in items:
    print("-", item.get("name", "unnamed"), "|", item.get("size_bytes", 0), "bytes")

print("GPU and R2 preflight passed. Starting the notebook worker.")
PY

exec python -u dream-host/notebook_worker.py
