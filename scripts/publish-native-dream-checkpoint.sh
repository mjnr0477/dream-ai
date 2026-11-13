#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT/dream-host/native-gpu.env"
CHECKPOINT="$ROOT/dream-host/training-state/checkpoints/dream-native-sft-checkpoint.pt"
RELEASE_GATE="$ROOT/dream-host/training-state/reports/native-sft-release-gate.json"
HASH_FILE="$ROOT/dream-host/training-state/reports/native-sft-checkpoint.sha256"
ARTIFACT_NAME="dream-native-sft-checkpoint.pt"
HOST_URL="https://dream-host-live.onrender.com"

fail() {
  printf 'DREAM checkpoint publisher: %s\n' "$*" >&2
  exit 1
}

[[ "${1:-}" == "--approve-reviewed-candidate" ]] || fail "Publishing requires the explicit --approve-reviewed-candidate flag after review."
[[ -f "$ENV_FILE" ]] || fail "Private file $ENV_FILE is missing."
[[ -s "$CHECKPOINT" ]] || fail "Native SFT checkpoint is missing. Run scripts/train-native-dream-gpu.sh first."
[[ -s "$RELEASE_GATE" ]] || fail "Technical release gate report is missing."
[[ -s "$HASH_FILE" ]] || fail "Checkpoint SHA-256 record is missing."

command -v curl >/dev/null 2>&1 || fail "curl is required."
command -v python3 >/dev/null 2>&1 || fail "python3 is required."

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

[[ "${DREAM_HOST_ARTIFACT_TOKEN:-}" =~ ^.{32,}$ ]] || fail "DREAM_HOST_ARTIFACT_TOKEN must be at least 32 characters."
[[ "$DREAM_HOST_ARTIFACT_TOKEN" != "${DREAM_HOST_WORKER_TOKEN:-}" ]] || fail "Artifact and worker tokens must be different."

EXPECTED_HASH="$(tr -d '[:space:]' < "$HASH_FILE")"
ACTUAL_HASH="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
[[ "$EXPECTED_HASH" =~ ^[a-fA-F0-9]{64}$ ]] || fail "The saved checkpoint hash is invalid."
[[ "$ACTUAL_HASH" == "$EXPECTED_HASH" ]] || fail "Checkpoint changed after its release audit; retrain and re-audit before publishing."

python3 - "$RELEASE_GATE" "$EXPECTED_HASH" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("checkpoint_sha256") != sys.argv[2]:
    raise SystemExit("Release gate does not match the current checkpoint SHA-256.")
if report.get("technical_preflight_passed") is not True:
    raise SystemExit("The technical checkpoint/evaluation release gate did not pass.")
if not isinstance(report.get("completed_training_steps"), int) or report["completed_training_steps"] < 1:
    raise SystemExit("The checkpoint has no completed training steps.")
PY

printf '%s\n' "Uploading the audited candidate to the private DREAM R2 artifact vault..."
RESPONSE="$(curl --fail --silent --show-error \
  -X PUT \
  -H "Authorization: Bearer $DREAM_HOST_ARTIFACT_TOKEN" \
  -H "Content-Type: application/octet-stream" \
  --data-binary "@$CHECKPOINT" \
  "$HOST_URL/api/v1/notebook/artifacts/$ARTIFACT_NAME")"

UPLOADED_HASH="$(printf '%s' "$RESPONSE" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("artifact",{}).get("sha256",""))')"
[[ "$UPLOADED_HASH" == "$EXPECTED_HASH" ]] || fail "R2 did not confirm the exact audited checkpoint hash."

if grep -q '^DREAM_NATIVE_CHECKPOINT_SHA256=' "$ENV_FILE"; then
  sed -i "s/^DREAM_NATIVE_CHECKPOINT_SHA256=.*/DREAM_NATIVE_CHECKPOINT_SHA256=$EXPECTED_HASH/" "$ENV_FILE"
else
  printf '\nDREAM_NATIVE_CHECKPOINT_SHA256=%s\n' "$EXPECTED_HASH" >> "$ENV_FILE"
fi
chmod 600 "$ENV_FILE"

printf '%s\n' "Published exact checkpoint to private R2 and pinned its SHA-256 in native-gpu.env."
printf 'artifact=%s\nsha256=%s\n' "$ARTIFACT_NAME" "$EXPECTED_HASH"
printf '%s\n' "The live HOST model identity must be set to dream-native-v1 / none before starting the GPU worker. The app's Cloudflare fallback remains separate."
