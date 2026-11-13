#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT/dream-host/compose.native-gpu.yml"
ENV_FILE="$ROOT/dream-host/native-gpu.env"

fail() {
  printf 'DREAM native GPU host: %s\n' "$*" >&2
  exit 1
}

[[ -f "$COMPOSE_FILE" ]] || fail "Compose file is missing: $COMPOSE_FILE"
[[ -f "$ENV_FILE" ]] || fail "Create $ENV_FILE from dream-host/native-gpu.env.example and fill the private tokens plus approved checkpoint SHA-256."

command -v docker >/dev/null 2>&1 || fail "Docker is not installed."
docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is required."
docker info >/dev/null 2>&1 || fail "Docker daemon is not running or this account cannot access it."
command -v nvidia-smi >/dev/null 2>&1 || fail "NVIDIA driver tools are missing. This machine needs an NVIDIA CUDA-capable GPU."
nvidia-smi -L >/dev/null 2>&1 || fail "The NVIDIA driver cannot see a working GPU."

# Load only the operator-owned local env file. Never print credential values.
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

[[ "${DREAM_HOST_WORKER_TOKEN:-}" =~ ^.{32,}$ ]] || fail "DREAM_HOST_WORKER_TOKEN must be at least 32 characters."
[[ "${DREAM_HOST_ARTIFACT_TOKEN:-}" =~ ^.{32,}$ ]] || fail "DREAM_HOST_ARTIFACT_TOKEN must be at least 32 characters."
[[ "$DREAM_HOST_WORKER_TOKEN" != "$DREAM_HOST_ARTIFACT_TOKEN" ]] || fail "Worker and artifact tokens must be different."
[[ "${DREAM_NATIVE_CHECKPOINT_SHA256:-}" =~ ^[a-fA-F0-9]{64}$ ]] || fail "DREAM_NATIVE_CHECKPOINT_SHA256 must be the SHA-256 of the audited native SFT checkpoint already stored in DREAM R2."
[[ "${DREAM_NATIVE_CHECKPOINT_ARTIFACT:-dream-native-sft-checkpoint.pt}" == "dream-native-sft-checkpoint.pt" ]] || fail "Use the approved artifact name dream-native-sft-checkpoint.pt."

cd "$ROOT"

printf '%s\n' "Building the pinned CUDA worker image..."
docker compose --project-directory "$ROOT/dream-host" -f "$COMPOSE_FILE" build dream-native-gpu-worker

printf '%s\n' "Checking that Docker can pass the physical NVIDIA GPU into the container..."
docker compose --project-directory "$ROOT/dream-host" -f "$COMPOSE_FILE" run --rm --no-deps --entrypoint python dream-native-gpu-worker -c 'import torch,sys; print("CUDA build:", torch.version.cuda); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT AVAILABLE"); sys.exit(0 if torch.cuda.is_available() else 1)'

printf '%s\n' "Starting DREAM's native inference worker with automatic container restart..."
docker compose --project-directory "$ROOT/dream-host" -f "$COMPOSE_FILE" up -d dream-native-gpu-worker

docker compose --project-directory "$ROOT/dream-host" -f "$COMPOSE_FILE" ps
printf '%s\n' "Worker logs (credentials are never printed by this launcher):"
docker compose --project-directory "$ROOT/dream-host" -f "$COMPOSE_FILE" logs --tail=40 dream-native-gpu-worker
