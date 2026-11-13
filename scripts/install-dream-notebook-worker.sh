#!/usr/bin/env bash
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

if [[ "${EUID}" -ne 0 ]]; then
  die "Run as root, for example: sudo bash scripts/install-dream-notebook-worker.sh"
fi
if ! command -v systemctl >/dev/null 2>&1; then
  die "systemd is required. This installer targets an owner-controlled Linux GPU host."
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  die "NVIDIA driver is not installed or nvidia-smi is unavailable. This machine is not ready for CUDA training."
fi
if ! nvidia-smi -L >/dev/null 2>&1; then
  die "The NVIDIA driver cannot see a GPU. Fix the driver/device before installing DREAM."
fi

PYTHON_BIN="${DREAM_BASE_PYTHON:-python3}"
REPO_DIR="${DREAM_INSTALL_REPO_DIR:-/opt/dream-ai}"
BRANCH="${DREAM_REPO_BRANCH:-dream-host-runtime}"
REPO_URL="https://github.com/mjnr0477/dream-ai.git"

command -v git >/dev/null 2>&1 || die "Install git before running this installer."
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python interpreter not found: ${PYTHON_BIN}"

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("DREAM's GPU worker requires Python 3.10 or newer.")
PY

if [[ -e "${REPO_DIR}" ]]; then
  [[ -d "${REPO_DIR}/.git" ]] || die "${REPO_DIR} already exists and is not a Git checkout. Refusing to overwrite it."
  [[ -z "$(git -C "${REPO_DIR}" status --porcelain)" ]] || die "The existing ${REPO_DIR} checkout has uncommitted changes. Refusing to overwrite them."
  git -C "${REPO_DIR}" fetch --depth 1 origin "${BRANCH}"
  git -C "${REPO_DIR}" checkout -B "${BRANCH}" "origin/${BRANCH}"
else
  install -d -m 0755 "$(dirname "${REPO_DIR}")"
  git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${REPO_DIR}"
fi

if ! id dreamai >/dev/null 2>&1; then
  useradd --system --create-home --home-dir /var/lib/dream-ai --shell /usr/sbin/nologin dreamai
fi
install -d -o dreamai -g dreamai -m 0700 \
  /var/lib/dream-ai \
  /var/lib/dream-ai/data \
  /var/lib/dream-ai/checkpoints \
  /var/lib/dream-ai/logs
chown -R dreamai:dreamai "${REPO_DIR}"

"${PYTHON_BIN}" -m venv --system-site-packages "${REPO_DIR}/.venv"
"${REPO_DIR}/.venv/bin/python" -m pip install --upgrade pip
"${REPO_DIR}/.venv/bin/python" -m pip install -r "${REPO_DIR}/dream-host/requirements-notebook-worker.txt"

"${REPO_DIR}/.venv/bin/python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA-enabled PyTorch is not available in this Python environment. "
        "Install a PyTorch build compatible with this host's NVIDIA driver, then rerun the installer. "
        "The installer will not silently substitute CPU-only PyTorch."
    )
print("CUDA GPU ready:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
PY

[[ -t 0 ]] || die "Run this installer from an interactive terminal so tokens can be entered without being echoed."
read -r -s -p "DREAM_HOST_NOTEBOOK_WORKER_TOKEN from dream-host-live: " NOTEBOOK_TOKEN </dev/tty
printf '\n'
read -r -s -p "DREAM_HOST_ARTIFACT_TOKEN from dream-host-live (same value as DREAM_NOTEBOOK_ARTIFACT_TOKEN): " ARTIFACT_TOKEN </dev/tty
printf '\n'

[[ "${NOTEBOOK_TOKEN}" =~ ^[A-Za-z0-9._+=/-]{32,}$ ]] || die "Notebook worker token must be at least 32 safe characters."
[[ "${ARTIFACT_TOKEN}" =~ ^[A-Za-z0-9._+=/-]{32,}$ ]] || die "Artifact token must be at least 32 safe characters."
[[ "${NOTEBOOK_TOKEN}" != "${ARTIFACT_TOKEN}" ]] || die "Notebook worker and artifact tokens must be different."

install -d -m 0700 /etc/dream-ai
umask 077
cat > /etc/dream-ai/notebook-worker.env <<EOF
DREAM_HOST_URL=https://dream-host-live.onrender.com
DREAM_HOST_NOTEBOOK_WORKER_TOKEN=${NOTEBOOK_TOKEN}
DREAM_HOST_NOTEBOOK_TRUSTED_MODE=1
DREAM_NOTEBOOK_ARTIFACT_TOKEN=${ARTIFACT_TOKEN}
DREAM_REPO_DIR=${REPO_DIR}
DREAM_REPO_BRANCH=${BRANCH}
DREAM_CHECKPOINT_DIR=/var/lib/dream-ai/checkpoints
DREAM_WORK_DIR=/var/lib/dream-ai/logs
DREAM_APPROVED_RECORDS_PATH=/var/lib/dream-ai/data/approved-records.jsonl
DREAM_SFT_TRAIN_PATH=/var/lib/dream-ai/data/sft-train.jsonl
DREAM_SFT_VALIDATION_PATH=/var/lib/dream-ai/data/sft-validation.jsonl
DREAM_NATIVE_TRAIN_STEPS=2000
DREAM_NATIVE_TRAIN_SECONDS=1500
DREAM_NATIVE_SFT_STEPS=1000
DREAM_NATIVE_SFT_SECONDS=1500
DREAM_HOST_NOTEBOOK_MAX_EXECUTION_SECONDS=1800
DREAM_HOST_NOTEBOOK_MAX_KERNELS=1
DREAM_HOST_NOTEBOOK_KERNEL_IDLE_SECONDS=86400
EOF
chmod 0600 /etc/dream-ai/notebook-worker.env
unset NOTEBOOK_TOKEN ARTIFACT_TOKEN

install -m 0644 "${REPO_DIR}/deploy/systemd/dream-notebook-worker.service" \
  /etc/systemd/system/dream-notebook-worker.service
systemctl daemon-reload
systemctl enable --now dream-notebook-worker.service

printf '\nDREAM NOTEBOOK GPU worker installed and enabled.\n'
printf 'Service status:\n'
systemctl --no-pager --full status dream-notebook-worker.service || true
printf '\nLogs: journalctl -u dream-notebook-worker -f\n'
printf 'Persistent data directory: /var/lib/dream-ai/data\n'
