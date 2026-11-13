#!/usr/bin/env bash
set -Eeuo pipefail

DATA_DIR="${DREAM_NATIVE_DATA_DIR:-/data}"
STATE_DIR="${DREAM_NATIVE_TRAIN_DIR:-/var/lib/dream-ai/training}"
PREPARED="$STATE_DIR/prepared"
CHECKPOINTS="$STATE_DIR/checkpoints"
REPORTS="$STATE_DIR/reports"
PRETRAIN="$CHECKPOINTS/dream-native-pretrain.pt"
SFT="$CHECKPOINTS/dream-native-sft-checkpoint.pt"

fail() {
  printf 'DREAM native training: %s\n' "$*" >&2
  exit 1
}

require_file() {
  [[ -s "$1" ]] || fail "Required reviewed dataset/release file is missing or empty: $1"
}

python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU is visible to the training container.")
print("Native DREAM training GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
PY

require_file "$DATA_DIR/approved-records.jsonl"
require_file "$DATA_DIR/sft-train.jsonl"
require_file "$DATA_DIR/sft-validation.jsonl"
require_file "$DATA_DIR/sft-audit-gate.json"

mkdir -p "$PREPARED" "$CHECKPOINTS" "$REPORTS"

printf '%s\n' "Auditing approved pretraining records..."
python model/audit_corpus.py \
  --input "$DATA_DIR/approved-records.jsonl" \
  --out "$REPORTS/corpus-audit.json"

printf '%s\n' "Preparing source-isolated train, validation and held-out test splits..."
python model/prepare_corpus.py \
  --input "$DATA_DIR/approved-records.jsonl" \
  --output "$PREPARED"

if [[ ! -s "$PRETRAIN" || "${DREAM_NATIVE_CONTINUE_PRETRAINING:-false}" == "true" ]]; then
  PRETRAIN_ARGS=()
  if [[ -s "$PRETRAIN" ]]; then
    PRETRAIN_ARGS=(--resume "$PRETRAIN")
  fi
  printf '%s\n' "Training/resuming the native DREAM base model..."
  python model/train.py \
    --data "$PREPARED/train.txt" \
    --validation-data "$PREPARED/validation.txt" \
    --corpus-manifest "$PREPARED/manifest.json" \
    --out "$PRETRAIN" \
    --steps "${DREAM_NATIVE_PRETRAIN_STEPS:-2000}" \
    --max-seconds "${DREAM_NATIVE_PRETRAIN_MAX_SECONDS:-0}" \
    --checkpoint-every 100 \
    --eval-every 100 \
    --eval-batches 5 \
    --batch-size 4 \
    --grad-accum-steps 4 \
    --gradient-checkpointing \
    --tokenizer bpe \
    --vocab-size 1024 \
    "${PRETRAIN_ARGS[@]}"
else
  printf '%s\n' "Keeping the existing native DREAM pretraining checkpoint; SFT will continue from its saved state."
fi
python model/audit_checkpoint.py \
  --checkpoint "$PRETRAIN" \
  --out "$REPORTS/pretrain-audit.json"

python model/evaluate.py \
  --checkpoint "$PRETRAIN" \
  --data "$PREPARED/test.txt" \
  --batches 20 \
  --out "$REPORTS/pretrain-evaluation.json"

SFT_ARGS=()
if [[ -s "$SFT" ]]; then
  BASE_SHA="$(sha256sum "$PRETRAIN" | awk '{print $1}')"
  SFT_PARENT_SHA="$(python - "$SFT" <<'PY'
import sys, torch
payload = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
print(payload.get("base_checkpoint_sha256", ""))
PY
)"
  if [[ "$SFT_PARENT_SHA" == "$BASE_SHA" ]]; then
    SFT_ARGS=(--resume "$SFT")
  else
    ARCHIVE="$CHECKPOINTS/dream-native-sft-previous-$(date -u +%Y%m%dT%H%M%SZ).pt"
    mv "$SFT" "$ARCHIVE"
    printf 'Archived old SFT checkpoint because its base model changed: %s\n' "$ARCHIVE"
  fi
fi
printf '%s\n' "Instruction-tuning/resuming from the audited native pretraining checkpoint..."
python model/sft.py \
  --checkpoint "$PRETRAIN" \
  --train-data "$DATA_DIR/sft-train.jsonl" \
  --validation-data "$DATA_DIR/sft-validation.jsonl" \
  --sft-audit-gate "$DATA_DIR/sft-audit-gate.json" \
  --pretraining-manifest "$PREPARED/manifest.json" \
  --pretraining-test-data "$PREPARED/test.txt" \
  --out "$SFT" \
  --steps "${DREAM_NATIVE_SFT_STEPS:-1000}" \
  --max-seconds "${DREAM_NATIVE_SFT_MAX_SECONDS:-0}" \
  --checkpoint-every 100 \
  --eval-every 100 \
  --eval-batches 5 \
  --batch-size 2 \
  --grad-accum-steps 4 \
  "${SFT_ARGS[@]}"

python model/audit_checkpoint.py \
  --checkpoint "$SFT" \
  --out "$REPORTS/native-sft-audit.json"

python model/evaluate.py \
  --checkpoint "$SFT" \
  --data "$PREPARED/test.txt" \
  --batches 20 \
  --out "$REPORTS/native-sft-evaluation.json"

python model/validate_release.py \
  --checkpoint "$SFT" \
  --audit "$REPORTS/native-sft-audit.json" \
  --evaluation "$REPORTS/native-sft-evaluation.json" \
  --out "$REPORTS/native-sft-release-gate.json"

sha256sum "$SFT" | awk '{print $1}' > "$REPORTS/native-sft-checkpoint.sha256"
printf '%s\n' "NATIVE_DREAM_CHECKPOINT_CANDIDATE_READY"
printf 'checkpoint=%s\n' "$SFT"
printf 'sha256=%s\n' "$(cat "$REPORTS/native-sft-checkpoint.sha256")"
printf 'completed_sft_steps=%s\n' "$(python - "$SFT" <<'PY'
import sys, torch
payload = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
print(payload.get("sft_training", {}).get("completed_steps", 0))
PY
)"
printf '%s\n' "This is a candidate. Review the report bundle before explicitly publishing it to the private R2 vault."
