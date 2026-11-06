#!/usr/bin/env bash
set -Eeuo pipefail

KERNEL_ID="${DREAM_KAGGLE_KERNEL:-mathiassimiyu/notebook21501a7fa8}"
OUT_DIR="${DREAM_KAGGLE_OUTPUT_DIR:-$PWD/dream-ai-kaggle-output}"
WAIT_SECONDS="${DREAM_KAGGLE_WAIT_SECONDS:-60}"
MAX_WAIT_SECONDS="${DREAM_KAGGLE_MAX_WAIT_SECONDS:-43200}"

command -v kaggle >/dev/null 2>&1 || {
  echo "Kaggle CLI is not installed in this environment." >&2
  exit 2
}
if [[ ! -f "$HOME/.kaggle/kaggle.json" && -z "${KAGGLE_USERNAME:-}" && -z "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "Kaggle API credentials are not configured here. Configure ~/.kaggle/kaggle.json locally; do not paste credentials into chat or Git." >&2
  exit 2
fi

WORK_DIR="$(mktemp -d)"
cleanup() { rm -rf "$WORK_DIR"; }
trap cleanup EXIT

mkdir -p "$OUT_DIR"
echo "Pulling existing Kaggle kernel metadata and attached inputs: $KERNEL_ID"
kaggle kernels pull "$KERNEL_ID" -p "$WORK_DIR" -m
META_FILE="$WORK_DIR/kernel-metadata.json"
[[ -f "$META_FILE" ]] || { echo "Kaggle metadata file was not downloaded." >&2; exit 3; }

python - "$META_FILE" "$KERNEL_ID" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
expected_id = sys.argv[2]
data = json.loads(path.read_text(encoding="utf-8"))
if data.get("id") != expected_id:
    raise SystemExit(f"Kernel metadata id mismatch: expected {expected_id!r}, got {data.get('id')!r}")
datasets = data.get("dataset_sources") or []
if len(datasets) != 1:
    raise SystemExit(
        "This DREAM notebook requires exactly one attached Kaggle dataset containing "
        "approved-records.jsonl. The existing kernel metadata has " + str(len(datasets)) + " dataset source(s). "
        "Attach the reviewed private dataset to the existing Kaggle notebook first; no training was started."
    )
data["code_file"] = "dream_train_kaggle.ipynb"
data["kernel_type"] = "notebook"
data["language"] = "python"
data["enable_gpu"] = True
data["enable_internet"] = True
data["is_private"] = True
path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY

NOTEBOOK_URL="https://raw.githubusercontent.com/mjnr0477/dream-ai/main/training/dream_train_kaggle.ipynb"
echo "Fetching the current DREAM training notebook"
curl -fsSL "$NOTEBOOK_URL" -o "$WORK_DIR/dream_train_kaggle.ipynb"
python - "$WORK_DIR/dream_train_kaggle.ipynb" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
doc = json.loads(p.read_text(encoding="utf-8"))
if doc.get("nbformat") != 4 or not doc.get("cells"):
    raise SystemExit("Downloaded DREAM Kaggle notebook is not a valid notebook.")
for i, cell in enumerate(doc["cells"]):
    if cell.get("cell_type") == "code":
        compile("".join(cell.get("source", [])), f"notebook-cell-{i}", "exec")
print(f"Notebook preflight passed: {len(doc['cells'])} cells")
PY

echo "Pushing the DREAM training notebook to Kaggle (GPU + Internet enabled)"
kaggle kernels push -p "$WORK_DIR"

elapsed=0
while (( elapsed < MAX_WAIT_SECONDS )); do
  status="$(kaggle kernels status "$KERNEL_ID" 2>&1 || true)"
  echo "$status"
  status_lc="$(printf '%s' "$status" | tr '[:upper:]' '[:lower:]')"
  if [[ "$status_lc" == *"complete"* ]]; then
    break
  fi
  if [[ "$status_lc" == *"error"* || "$status_lc" == *"failed"* || "$status_lc" == *"cancel"* ]]; then
    echo "Kaggle reports an unsuccessful run; output was not treated as a trained model." >&2
    exit 4
  fi
  sleep "$WAIT_SECONDS"
  elapsed=$((elapsed + WAIT_SECONDS))
done
if (( elapsed >= MAX_WAIT_SECONDS )); then
  echo "Timed out waiting for Kaggle. Check the notebook status in Kaggle; no completion is claimed." >&2
  exit 5
fi

echo "Downloading Kaggle notebook outputs"
kaggle kernels output "$KERNEL_ID" -p "$OUT_DIR"
if ! find "$OUT_DIR" -type f -name 'dream-training-artifacts.zip' -print -quit | grep -q .; then
  echo "No dream-training-artifacts.zip was returned. Inspect the Kaggle notebook output and logs:" >&2
  find "$OUT_DIR" -maxdepth 3 -type f -printf '%P\n' >&2 || true
  exit 6
fi

echo "DREAM training artifact bundle downloaded to: $OUT_DIR"
