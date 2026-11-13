#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT/dream-host/compose.native-gpu.yml"
TRAIN_ENV="$ROOT/dream-host/native-training.env"
DATA_DIR="$ROOT/dream-host/native-training-data"
STATE_DIR="$ROOT/dream-host/training-state"

fail() {
  printf 'DREAM native GPU training: %s\n' "$*" >&2
  exit 1
}

[[ -f "$COMPOSE_FILE" ]] || fail "Compose file is missing."
[[ -f "$TRAIN_ENV" ]] || fail "Copy dream-host/native-training.env.example to dream-host/native-training.env first."
[[ -s "$DATA_DIR/approved-records.jsonl" ]] || fail "Add the reviewed approved-records.jsonl corpus to $DATA_DIR."
[[ -s "$DATA_DIR/sft-train.jsonl" ]] || fail "Add the separately reviewed sft-train.jsonl file to $DATA_DIR."
[[ -s "$DATA_DIR/sft-validation.jsonl" ]] || fail "Add the separately reviewed sft-validation.jsonl file to $DATA_DIR."
[[ -s "$DATA_DIR/sft-audit-gate.json" ]] || fail "Add the hash-bound sft-audit-gate.json produced by the review workflow."

command -v docker >/dev/null 2>&1 || fail "Docker is not installed."
docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is required."
docker info >/dev/null 2>&1 || fail "Docker daemon is not running."
command -v nvidia-smi >/dev/null 2>&1 || fail "NVIDIA driver tools are missing."
nvidia-smi -L >/dev/null 2>&1 || fail "No working NVIDIA GPU is visible to the host."

mkdir -p "$STATE_DIR"
export DREAM_NATIVE_UID="$(id -u)"
export DREAM_NATIVE_GID="$(id -g)"
cd "$ROOT"

docker compose --project-directory "$ROOT/dream-host" -f "$COMPOSE_FILE" build dream-native-gpu-trainer

printf '%s\n' "Starting resumable native DREAM training. Existing checkpoints and optimizer state are retained in dream-host/training-state."
docker compose --project-directory "$ROOT/dream-host" -f "$COMPOSE_FILE" --profile training run --rm --no-deps dream-native-gpu-trainer

printf '%s\n' "Training finished and the technical release gate passed."
printf '%s\n' "Candidate and reports: $STATE_DIR/checkpoints and $STATE_DIR/reports"
printf '%s\n' "No production model was activated. Publish only after reviewing the candidate report."
