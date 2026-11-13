"""DREAM AI adapter trainer for a remote managed GPU worker.

This is a provider-neutral port of the reviewed Kaggle notebook. It preserves
the consent, provenance, validation, resume and candidate-only release gates.
"""

from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json, math, os, random, re, shutil, sys, time
import torch

BASE_MODEL = os.environ.get("DREAM_BASE_MODEL", "Qwen/Qwen3-1.7B").strip()
PRODUCTION_ADAPTER_REPO = os.environ.get("DREAM_ADAPTER_REPO", "msjnr0477/dream-ai-adapter").strip()
TRAINING_WORK_DIR = Path(os.environ.get("DREAM_TRAINING_WORK_DIR", "/tmp/dream-ai-training")).resolve()
TRAINING_WORK_DIR.mkdir(parents=True, exist_ok=True)
MIN_TOTAL_RECORDS = 32
MIN_SOURCE_GROUPS = 8
MAX_SEQ_LEN = 2048
BATCH_SIZE = 1
GRAD_ACCUMULATION = 4
EPOCHS = 2
LEARNING_RATE = 1e-5
MAX_GRAD_NORM = 1.0
MAX_TRAINING_SECONDS = int(os.environ.get("DREAM_TRAINING_SECONDS", "240"))

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is required in the GPU job environment.")
if MAX_TRAINING_SECONDS < 30 or MAX_TRAINING_SECONDS > 900:
    raise RuntimeError("DREAM_TRAINING_SECONDS must be between 30 and 900 seconds.")
if not torch.cuda.is_available():
    raise RuntimeError("No GPU is available. Select a Kaggle GPU accelerator before training.")

for package in ("transformers", "peft", "accelerate", "safetensors", "huggingface_hub"):
    print(package, importlib.metadata.version(package))
print("GPU:", torch.cuda.get_device_name(0))
print("HF token available:", bool(HF_TOKEN))
print("Training target:", PRODUCTION_ADAPTER_REPO, "→ candidate folder only")

from huggingface_hub import hf_hub_download

input_root = Path(os.environ.get("DREAM_TRAINING_DATA_DIR", "/tmp/dream-ai-training-data")).resolve()
input_root.mkdir(parents=True, exist_ok=True)
dataset_path = os.environ.get("DREAM_TRAINING_DATASET_PATH", "training-data/active").strip("/")
if not dataset_path or ".." in Path(dataset_path).parts:
    raise RuntimeError("DREAM_TRAINING_DATASET_PATH must be a safe repository-relative path.")
for required_name in ("train.jsonl", "validation.jsonl", "manifest.json", "audit-gate.json"):
    destination = input_root / required_name
    if not destination.is_file():
        cached_file = hf_hub_download(
            repo_id=PRODUCTION_ADAPTER_REPO,
            filename=f"{dataset_path}/{required_name}",
            repo_type="model",
            token=HF_TOKEN,
        )
        shutil.copy2(cached_file, destination)

def find_exact(name):
    found = sorted(input_root.glob("**/" + name))
    if len(found) != 1:
        raise RuntimeError(f"Attach exactly one private dataset file named {name}; found {len(found)}")
    return found[0]

train_path = find_exact("train.jsonl")
validation_path = find_exact("validation.jsonl")
manifest_path = find_exact("manifest.json")
gate_path = find_exact("audit-gate.json")
train_bytes = train_path.read_bytes()
validation_bytes = validation_path.read_bytes()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
audit_gate = json.loads(gate_path.read_text(encoding="utf-8"))

def digest_bytes(data):
    return sha256(data).hexdigest()

if manifest.get("schema") != "dream-sft-splits-v1":
    raise RuntimeError("Unsupported or missing DREAM SFT split manifest.")
if audit_gate.get("schema") != "dream-sft-audit-gate-v1" or audit_gate.get("passed") is not True:
    raise RuntimeError("The human-reviewed SFT privacy/provenance audit gate has not passed.")
if audit_gate.get("dataset_sha256") != manifest.get("dataset_sha256"):
    raise RuntimeError("Audit gate is bound to a different dataset manifest.")
for split_name, raw in (("train", train_bytes), ("validation", validation_bytes)):
    expected = manifest.get("splits", {}).get(split_name, {}).get("sha256")
    if digest_bytes(raw) != expected:
        raise RuntimeError(f"{split_name} file hash does not match the signed-off manifest.")
    if audit_gate.get("splits", {}).get(split_name, {}).get("sha256") != expected:
        raise RuntimeError(f"{split_name} audit gate does not match the exact file hash.")

def read_jsonl(path):
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL in {path.name} at line {line_number}") from exc
    return rows

train_rows = read_jsonl(train_path)
validation_rows = read_jsonl(validation_path)
all_rows = train_rows + validation_rows
for split_name, rows in (("train", train_rows), ("validation", validation_rows)):
    split_meta = manifest.get("splits", {}).get(split_name, {})
    if len(rows) != split_meta.get("records"):
        raise RuntimeError(f"{split_name} record count does not match the manifest.")
    if len({row.get("source_group") for row in rows if isinstance(row, dict)}) != split_meta.get("source_groups"):
        raise RuntimeError(f"{split_name} source-group count does not match the manifest.")
if len(all_rows) < MIN_TOTAL_RECORDS:
    raise RuntimeError(f"Need at least {MIN_TOTAL_RECORDS} human-approved examples; found {len(all_rows)}.")
if len(validation_rows) < 4:
    raise RuntimeError("At least four independent validation examples are required.")

def validate_approved_row(row, index):
    if not isinstance(row, dict):
        raise RuntimeError(f"Record {index} is not a JSON object.")
    required_flags = ("rights_confirmed", "privacy_reviewed", "quality_reviewed", "approval_granted")
    if row.get("consented") is not True or row.get("consent_version") != "dream-learning-opt-in-v3":
        raise RuntimeError(f"Record {index} lacks explicit v3 training consent.")
    if any(row.get(flag) is not True for flag in required_flags):
        raise RuntimeError(f"Record {index} has an incomplete human review.")
    if row.get("source") != "DREAM AI opt-in exchange" or row.get("license") != "DREAM AI Contributor Permission v3":
        raise RuntimeError(f"Record {index} has unexpected source or license metadata.")
    if row.get("rights_basis") != "explicit-permission":
        raise RuntimeError(f"Record {index} is missing explicit permission.")
    if row.get("source_url") != "https://dream-ai-live.onrender.com" or row.get("license_url") != "https://dream-ai-live.onrender.com/learning-data-terms.html":
        raise RuntimeError(f"Record {index} has unexpected provenance URLs.")
    if not re.fullmatch(r"dream-optin-[a-f0-9]{64}", str(row.get("id", ""))):
        raise RuntimeError(f"Record {index} has an invalid privacy-preserving record ID.")
    if not re.fullmatch(r"dream-ai-session-[a-f0-9]{64}", str(row.get("source_group", ""))):
        raise RuntimeError(f"Record {index} has an invalid privacy-preserving session group.")
    try:
        datetime.fromisoformat(str(row.get("consented_at", "")).replace("Z", "+00:00"))
        datetime.fromisoformat(str(row.get("human_reviewed_at", "")).replace("Z", "+00:00"))
        datetime.strptime(str(row.get("source_date", "")), "%Y-%m-%d")
    except ValueError as exc:
        raise RuntimeError(f"Record {index} has missing or invalid consent/review/source dates.") from exc
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise RuntimeError(f"Record {index} does not contain a complete conversation.")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant") or not isinstance(message.get("content"), str) or not message["content"].strip():
            raise RuntimeError(f"Record {index} contains an invalid message.")
    if messages[-1].get("role") != "assistant":
        raise RuntimeError(f"Record {index} does not end with an assistant response.")
    dialogue = messages[1:] if messages[0].get("role") == "system" else messages
    if len(dialogue) % 2 or any(item["role"] != ("user" if i % 2 == 0 else "assistant") for i, item in enumerate(dialogue)):
        raise RuntimeError(f"Record {index} messages do not alternate user/assistant.")

for i, row in enumerate(all_rows, 1):
    validate_approved_row(row, i)
train_groups = {row["source_group"] for row in train_rows}
validation_groups = {row["source_group"] for row in validation_rows}
if train_groups & validation_groups:
    raise RuntimeError("Source-group leakage detected between train and validation.")
source_groups = train_groups | validation_groups
if len(source_groups) < MIN_SOURCE_GROUPS:
    raise RuntimeError(f"Need at least {MIN_SOURCE_GROUPS} independent session groups; found {len(source_groups)}.")

print("Approved training records:", len(train_rows))
print("Approved validation records:", len(validation_rows))
print("Independent source groups:", len(source_groups))
print("Dataset SHA-256:", manifest["dataset_sha256"])
print("Audit gate: PASSED")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from torch.utils.data import DataLoader
from huggingface_hub import HfApi, hf_hub_download

random.seed(47)
torch.manual_seed(47)
torch.cuda.manual_seed_all(47)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    token=HF_TOKEN,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
base_model.config.use_cache = False
api = HfApi(token=HF_TOKEN)
adapter_repo_info = api.repo_info(repo_id=PRODUCTION_ADAPTER_REPO, repo_type="model", token=HF_TOKEN)
if getattr(adapter_repo_info, "private", None) is not True:
    raise RuntimeError("The DREAM adapter/model repository must be private before storing training data and progress.")
production_adapter_file = hf_hub_download(
    repo_id=PRODUCTION_ADAPTER_REPO,
    filename="adapter_model.safetensors",
    repo_type="model",
    token=HF_TOKEN,
)
parent_adapter_sha256 = sha256(Path(production_adapter_file).read_bytes()).hexdigest()
audit_gate_sha256 = digest_bytes(gate_path.read_bytes())
progress_prefix = (
    f"training-progress/{manifest['dataset_sha256']}-"
    f"{audit_gate_sha256}-{parent_adapter_sha256}"
)
progress_dir = TRAINING_WORK_DIR / "dream-ai-adapter-progress"
progress_dir.mkdir(parents=True, exist_ok=True)
progress_meta = None
repo_files = set(api.list_repo_files(
    repo_id=PRODUCTION_ADAPTER_REPO,
    repo_type="model",
    token=HF_TOKEN,
))
progress_meta_name = progress_prefix + "/progress.json"
if progress_meta_name in repo_files:
    for filename in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer-state.pt",
        "rng-state.pt",
        "progress.json",
    ):
        downloaded = hf_hub_download(
            repo_id=PRODUCTION_ADAPTER_REPO,
            filename=progress_prefix + "/" + filename,
            repo_type="model",
            token=HF_TOKEN,
        )
        shutil.copy2(downloaded, progress_dir / filename)
    progress_meta = json.loads((progress_dir / "progress.json").read_text(encoding="utf-8"))
    expected_progress = {
        "schema": "dream-ai-adapter-progress-v1",
        "dataset_sha256": manifest["dataset_sha256"],
        "audit_gate_sha256": audit_gate_sha256,
        "parent_adapter_sha256": parent_adapter_sha256,
    }
    for key, value in expected_progress.items():
        if progress_meta.get(key) != value:
            raise RuntimeError("Saved progress does not match this exact dataset, audit gate and parent adapter. Refusing to resume.")
    print("Resuming private adapter training progress from", progress_prefix)
    adapter_source = str(progress_dir)
else:
    print("No matching training progress found. Starting from the current production adapter.")
    adapter_source = PRODUCTION_ADAPTER_REPO

model = PeftModel.from_pretrained(
    base_model,
    adapter_source,
    token=HF_TOKEN if adapter_source == PRODUCTION_ADAPTER_REPO else None,
    is_trainable=True,
    autocast_adapter_dtype=True,
)
model.config.use_cache = False
if progress_meta is not None:
    rng_state = torch.load(
        progress_dir / "rng-state.pt",
        map_location="cpu",
        weights_only=True,
    )
    torch.set_rng_state(rng_state["torch_rng_state"].cpu())
    if torch.cuda.is_available() and rng_state.get("cuda_rng_states"):
        torch.cuda.set_rng_state_all(rng_state["cuda_rng_states"])
model.gradient_checkpointing_enable()
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
if not trainable_parameters or trainable_count < 1:
    raise RuntimeError("No trainable adapter parameters were found. Refusing to run a no-op training cycle.")
input_device = model.get_input_embeddings().weight.device
if input_device.type != "cuda":
    raise RuntimeError("The base model was not placed on the Kaggle GPU. Refusing to train on CPU.")
for parameter in trainable_parameters:
    if parameter.dtype != torch.float32:
        parameter.data = parameter.data.float()
print("Foundation model:", BASE_MODEL)
print("Starting adapter:", PRODUCTION_ADAPTER_REPO)
print("Trainable adapter parameters:", f"{trainable_count:,}")
print("Training device:", input_device)

def encode_record(row):
    messages = row["messages"]
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    prefix_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full_ids = list(full_ids)
    prefix_ids = list(prefix_ids)
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise RuntimeError("Tokenizer chat template did not preserve the assistant-only prefix boundary.")
    if len(full_ids) > MAX_SEQ_LEN:
        return None
    labels = ([-100] * len(prefix_ids)) + full_ids[len(prefix_ids):]
    if sum(token != -100 for token in labels) < 3:
        raise RuntimeError("An approved response is too short to train safely.")
    return {"input_ids": full_ids, "labels": labels}

encoded_train = [item for row in train_rows if (item := encode_record(row)) is not None]
encoded_validation = [item for row in validation_rows if (item := encode_record(row)) is not None]
if len(encoded_train) < 20 or len(encoded_validation) < 4:
    raise RuntimeError("Too many examples exceed the context limit; at least 20 train and 4 validation examples must fit.")

def collate_batch(items):
    width = max(len(item["input_ids"]) for item in items)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    labels = torch.full((len(items), width), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(items), width), dtype=torch.long)
    for index, item in enumerate(items):
        size = len(item["input_ids"])
        input_ids[index, :size] = torch.tensor(item["input_ids"], dtype=torch.long)
        labels[index, :size] = torch.tensor(item["labels"], dtype=torch.long)
        attention_mask[index, :size] = 1
    return {
        "input_ids": input_ids.to(input_device),
        "labels": labels.to(input_device),
        "attention_mask": attention_mask.to(input_device),
    }

train_loader = DataLoader(encoded_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
validation_loader = DataLoader(encoded_validation, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
print("Train examples used:", len(encoded_train), "· excluded for length:", len(train_rows) - len(encoded_train))
print("Validation examples used:", len(encoded_validation), "· excluded for length:", len(validation_rows) - len(encoded_validation))

def evaluate_masked_loss(data_loader):
    model.eval()
    weighted_loss = 0.0
    target_tokens = 0
    with torch.inference_mode():
        for batch in data_loader:
            target_count = int(batch["labels"].ne(-100).sum().item())
            if target_count == 0:
                continue
            output = model(**batch)
            weighted_loss += float(output.loss.item()) * target_count
            target_tokens += target_count
    if target_tokens == 0:
        raise RuntimeError("The validation split contains no assistant target tokens.")
    return weighted_loss / target_tokens

baseline_validation_loss = evaluate_masked_loss(validation_loader)
if not math.isfinite(baseline_validation_loss):
    raise RuntimeError("The existing adapter produced a non-finite validation loss.")
print("Current adapter validation loss:", round(baseline_validation_loss, 6))

import urllib.request

def generate_answer(prompt, max_new_tokens=96):
    model.eval()
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(input_device)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

benchmark_url = "https://raw.githubusercontent.com/mjnr0477/dream-ai/main/evals/dream-core-v1.jsonl"
benchmark_tasks = []
try:
    with urllib.request.urlopen(benchmark_url, timeout=30) as response:
        raw_benchmark = response.read().decode("utf-8")
    benchmark_tasks = [json.loads(line) for line in raw_benchmark.splitlines() if line.strip()][:24]
except Exception as error:
    print("Benchmark file could not be downloaded; loss validation remains available:", type(error).__name__)

baseline_completions = []
if benchmark_tasks:
    for task in benchmark_tasks:
        baseline_completions.append(generate_answer(task["prompt"]))
    print("Saved pre-training baseline responses for", len(baseline_completions), "task prompts.")

model.config.use_cache = False

optimizer = torch.optim.AdamW(trainable_parameters, lr=LEARNING_RATE, weight_decay=0.01)
completed_updates_before = int((progress_meta or {}).get("completed_updates", 0))
if progress_meta is not None:
    optimizer_state = torch.load(
        progress_dir / "optimizer-state.pt",
        map_location="cpu",
        weights_only=True,
    )
    optimizer.load_state_dict(optimizer_state)
    print("Restored optimizer state after", completed_updates_before, "updates.")

training_steps = 0
training_started_at = time.monotonic()
training_budget_reached = False
model.train()
for epoch in range(EPOCHS):
    epoch_batches = list(train_loader)
    for start in range(0, len(epoch_batches), GRAD_ACCUMULATION):
        group = epoch_batches[start:start + GRAD_ACCUMULATION]
        optimizer.zero_grad(set_to_none=True)
        group_loss = 0.0
        for batch in group:
            output = model(**batch)
            loss = output.loss
            (loss / len(group)).backward()
            group_loss += float(loss.detach().item())
        torch.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
        optimizer.step()
        training_steps += 1
        elapsed = time.monotonic() - training_started_at
        print(
            f"epoch={epoch + 1}/{EPOCHS} step={training_steps} "
            f"cumulative_updates={completed_updates_before + training_steps} "
            f"elapsed={elapsed:.1f}s mean_loss={group_loss / len(group):.5f}"
        )
        if elapsed >= MAX_TRAINING_SECONDS:
            training_budget_reached = True
            break
    if training_budget_reached:
        break
    torch.cuda.empty_cache()

training_seconds = time.monotonic() - training_started_at
if training_steps < 1:
    raise RuntimeError("No adapter optimizer updates completed; no checkpoint will be saved.")
print("Training seconds:", round(training_seconds, 2))
print("Training budget reached:", training_budget_reached)

candidate_validation_loss = evaluate_masked_loss(validation_loader)
if not math.isfinite(candidate_validation_loss):
    raise RuntimeError("Candidate adapter produced a non-finite validation loss.")
relative_change = (candidate_validation_loss - baseline_validation_loss) / max(abs(baseline_validation_loss), 1e-8)
print("Baseline validation loss:", round(baseline_validation_loss, 6))
print("Candidate validation loss:", round(candidate_validation_loss, 6))
print("Relative loss change:", round(relative_change * 100, 2), "%")
if candidate_validation_loss > baseline_validation_loss * 1.02:
    raise RuntimeError("Candidate regressed by more than 2% on held-out validation. It will not be uploaded.")
print("Technical validation passed. Human evaluation is still required before promotion.")

candidate_completions = [generate_answer(task["prompt"]) for task in benchmark_tasks] if benchmark_tasks else []
run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + manifest["dataset_sha256"][:12]
candidate_dir = TRAINING_WORK_DIR / "dream-ai-adapter-candidate"
candidate_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(candidate_dir, safe_serialization=True)
tokenizer.save_pretrained(candidate_dir)

# Preserve the exact reviewed-data continuation separately from candidate release files.
progress_export = TRAINING_WORK_DIR / "dream-ai-adapter-progress-export"
progress_export.mkdir(parents=True, exist_ok=True)
model.save_pretrained(progress_export, safe_serialization=True)
torch.save(optimizer.state_dict(), progress_export / "optimizer-state.pt")
torch.save({
    "torch_rng_state": torch.get_rng_state().cpu(),
    "cuda_rng_states": [
        state.cpu() for state in torch.cuda.get_rng_state_all()
    ] if torch.cuda.is_available() else [],
}, progress_export / "rng-state.pt")
progress_manifest = {
    "schema": "dream-ai-adapter-progress-v1",
    "dataset_sha256": manifest["dataset_sha256"],
    "audit_gate_sha256": audit_gate_sha256,
    "parent_adapter_sha256": parent_adapter_sha256,
    "completed_updates": completed_updates_before + training_steps,
    "session_updates": training_steps,
    "training_seconds": training_seconds,
    "training_budget_seconds": MAX_TRAINING_SECONDS,
    "validation_loss": candidate_validation_loss,
    "created_at": datetime.now(timezone.utc).isoformat(),
}
(progress_export / "progress.json").write_text(
    json.dumps(progress_manifest, indent=2) + "\n", encoding="utf-8"
)

comparison_path = TRAINING_WORK_DIR / "dream-ai-eval-comparison.jsonl"
with comparison_path.open("w", encoding="utf-8") as handle:
    for index, task in enumerate(benchmark_tasks):
        record = {
            "id": task.get("id"),
            "category": task.get("category"),
            "language": task.get("language"),
            "prompt": task.get("prompt"),
            "reference": task.get("reference"),
            "baseline_response": baseline_completions[index],
            "candidate_response": candidate_completions[index],
            "human_review_required": True,
        }
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

candidate_manifest = {
    "schema": "dream-ai-adapter-candidate-v1",
    "candidate_id": run_id,
    "status": "candidate-not-active",
    "base_model": BASE_MODEL,
    "parent_adapter_repo": PRODUCTION_ADAPTER_REPO,
    "dataset_sha256": manifest["dataset_sha256"],
    "audit_gate_passed": True,
    "audit_gate_sha256": digest_bytes(gate_path.read_bytes()),
    "train_records": len(encoded_train),
    "validation_records": len(encoded_validation),
    "source_groups": len(source_groups),
    "training_steps": training_steps,
    "cumulative_training_steps": completed_updates_before + training_steps,
    "training_seconds": training_seconds,
    "training_budget_seconds": MAX_TRAINING_SECONDS,
    "training_budget_reached": training_budget_reached,
    "resumed_from_progress": progress_meta is not None,
    "progress_prefix": progress_prefix,
    "baseline_validation_loss": baseline_validation_loss,
    "candidate_validation_loss": candidate_validation_loss,
    "validation_relative_change": relative_change,
    "benchmark_tasks": len(benchmark_tasks),
    "benchmark_comparison": str(comparison_path),
    "created_at": datetime.now(timezone.utc).isoformat(),
    "promotion_requires_human_review": True,
}
(candidate_dir / "dream-candidate-manifest.json").write_text(json.dumps(candidate_manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(candidate_manifest, indent=2))

api = HfApi(token=HF_TOKEN)
api.repo_info(repo_id=PRODUCTION_ADAPTER_REPO, repo_type="model", token=HF_TOKEN)
progress_uploaded = api.upload_folder(
    repo_id=PRODUCTION_ADAPTER_REPO,
    repo_type="model",
    folder_path=str(progress_export),
    path_in_repo=progress_prefix,
    commit_message="Save validated DREAM adapter training progress " + run_id,
    token=HF_TOKEN,
)
print("Private resumable progress uploaded:", progress_uploaded)
candidate_path = "candidates/" + run_id
uploaded = api.upload_folder(
    repo_id=PRODUCTION_ADAPTER_REPO,
    repo_type="model",
    folder_path=str(candidate_dir),
    path_in_repo=candidate_path,
    commit_message="Add reviewed-data DREAM adapter candidate " + run_id,
    token=HF_TOKEN,
)
print("Candidate uploaded:", uploaded)
print("Private candidate path:", f"https://huggingface.co/{PRODUCTION_ADAPTER_REPO}/tree/main/{candidate_path}")
print("Active adapter root was not modified.")
print("Review:", comparison_path)
print("Candidate manifest:", candidate_dir / "dream-candidate-manifest.json")
