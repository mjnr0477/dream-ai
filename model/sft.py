"""Supervised instruction fine-tuning for DREAM's own pretrained checkpoint.

Only assistant response tokens contribute to loss. No external model or API is used.
Input records are human-reviewed JSONL conversation examples with HTTPS source/license URLs and an explicit, verified rights basis.
"""
import argparse
import hashlib
import json
import math
import random
from datetime import date
from pathlib import Path
from urllib.parse import urlparse


ALLOWED_RIGHTS_BASIS = {
    "original-authorship", "public-domain", "open-license", "explicit-permission"
}

import torch

from checkpoint_tokenizer import load_tokenizer
from dream_transformer import DreamConfig, DreamTransformer


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_groups(path):
    groups = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            groups.add(json.loads(line)["source_group"])
    return groups


def record_signatures(path):
    """Return record IDs and normalized dialogue hashes for split-leak checks."""
    record_ids = set()
    content_hashes = set()
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON at line {line_number}: {exc}") from exc
        record_id = item.get("id")
        if isinstance(record_id, str) and record_id:
            record_ids.add(record_id)
        messages = item.get("messages")
        if isinstance(messages, list):
            normalized = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = message.get("role")
                content = message.get("content")
                if isinstance(role, str) and isinstance(content, str):
                    normalized.append(
                        f"{role.strip().lower()}: {' '.join(content.split()).casefold()}"
                    )
            if normalized:
                canonical = "\n".join(normalized).encode("utf-8")
                content_hashes.add(hashlib.sha256(canonical).hexdigest())
    return record_ids, content_hashes


def load_examples(path, tokenizer, block_size):
    examples = []
    seen_ids = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON at line {line_number}: {exc}") from exc
        for field in (
            "id", "source", "source_group", "license", "license_url",
            "source_url", "rights_basis", "source_date", "language"
        ):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise SystemExit(f"Missing {field} at line {line_number}")
        for field in ("source_url", "license_url"):
            parsed_url = urlparse(item[field])
            if parsed_url.scheme != "https" or not parsed_url.hostname:
                raise SystemExit(
                    f"Invalid {field} at line {line_number}; expected an absolute HTTPS URL"
                )
        if item["rights_basis"] not in ALLOWED_RIGHTS_BASIS:
            raise SystemExit(
                f"Invalid rights_basis at line {line_number}; expected one of "
                + ", ".join(sorted(ALLOWED_RIGHTS_BASIS))
            )
        try:
            parsed_source_date = date.fromisoformat(item["source_date"])
            if parsed_source_date.isoformat() != item["source_date"]:
                raise ValueError("source_date must use YYYY-MM-DD")
        except ValueError as exc:
            raise SystemExit(
                f"Invalid source_date at line {line_number}; expected YYYY-MM-DD"
            ) from exc
        for field, label in (
            ("rights_confirmed", "rights"),
            ("privacy_reviewed", "privacy"),
            ("quality_reviewed", "quality")
        ):
            if item.get(field) is not True:
                raise SystemExit(f"{label.title()} review not confirmed at line {line_number}")
        if item["id"] in seen_ids:
            raise SystemExit(f"Duplicate SFT record id: {item['id']}")
        seen_ids.add(item["id"])
        messages = item.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise SystemExit(f"At least one user/assistant pair required at line {line_number}")

        normalized = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise SystemExit(f"Invalid message at line {line_number}")
            role = message.get("role")
            text = message.get("content")
            if role not in ("system", "user", "assistant") or not isinstance(text, str) or not text.strip():
                raise SystemExit(f"Invalid role/content at line {line_number}")
            if role == "system" and index != 0:
                raise SystemExit(f"System message must be first at line {line_number}")
            normalized.append((role, text.strip()))

        dialogue = normalized[1:] if normalized[0][0] == "system" else normalized
        if len(dialogue) < 2 or len(dialogue) % 2:
            raise SystemExit(f"Conversation must end with an assistant answer at line {line_number}")
        if any(role != ("user" if index % 2 == 0 else "assistant")
               for index, (role, _) in enumerate(dialogue)):
            raise SystemExit(f"Conversation roles must alternate user/assistant at line {line_number}")

        for answer_index, (role, answer) in enumerate(normalized):
            if role != "assistant":
                continue
            prefix_messages = normalized[:answer_index]
            prefix = "".join(f"{message_role.upper()}: {message_text}\n"
                             for message_role, message_text in prefix_messages)
            prefix += "ASSISTANT:"
            prefix_ids = tokenizer.encode(prefix)
            answer_ids = tokenizer.encode(" " + answer + "\n")
            if len(answer_ids) >= block_size:
                raise SystemExit(
                    f"Assistant answer exceeds block_size at line {line_number}; "
                    "split or shorten the reviewed example."
                )
            available_prefix = max(0, block_size - len(answer_ids))
            prefix_ids = prefix_ids[-available_prefix:] if available_prefix else []
            sequence = prefix_ids + answer_ids
            labels = ([-100] * len(prefix_ids)) + answer_ids
            if len(sequence) < 2:
                continue
            input_ids = sequence[:-1]
            target_ids = labels[1:]
            if not any(target != -100 for target in target_ids):
                continue
            examples.append((input_ids, target_ids))

    if not examples:
        raise SystemExit(f"No usable supervised assistant examples found in {path}")
    return examples


def make_batch(examples, batch_size, device):
    selected = random.choices(examples, k=batch_size)
    length = max(len(input_ids) for input_ids, _ in selected)
    inputs = torch.zeros((batch_size, length), dtype=torch.long)
    targets = torch.full((batch_size, length), -100, dtype=torch.long)
    for row, (input_ids, target_ids) in enumerate(selected):
        inputs[row, :len(input_ids)] = torch.tensor(input_ids, dtype=torch.long)
        targets[row, :len(target_ids)] = torch.tensor(target_ids, dtype=torch.long)
    return inputs.to(device), targets.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="DREAM pretrained checkpoint")
    parser.add_argument("--train-data", required=True, help="Reviewed training JSONL")
    parser.add_argument("--validation-data", required=True, help="Independent reviewed validation JSONL")
    parser.add_argument(
        "--pretraining-manifest", required=True,
        help="Manifest from the pretraining corpus; SFT sources must not overlap its held-out test split"
    )
    parser.add_argument(
        "--pretraining-test-data", required=True,
        help="Exact held-out test text file recorded in the pretraining manifest"
    )
    parser.add_argument("--out", default="checkpoints/dream-instruct.pt")
    parser.add_argument("--resume", default="")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if any(value < 1 for value in (
        args.steps, args.batch_size, args.grad_accum_steps,
        args.eval_every, args.eval_batches, args.checkpoint_every
    )):
        raise SystemExit("All step, batch and evaluation counts must be positive.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise SystemExit("learning-rate must be finite and positive.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_path = Path(args.train_data)
    validation_path = Path(args.validation_data)
    train_hash = sha256(train_path)
    validation_hash = sha256(validation_path)
    if train_hash == validation_hash:
        raise SystemExit("SFT training and validation content hashes must differ.")

    base = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base_steps = base.get("training", {}).get("completed_steps", 0)
    if not isinstance(base_steps, int) or base_steps < 1:
        raise SystemExit("SFT requires a DREAM checkpoint with completed pretraining steps.")
    config = DreamConfig(**base["config"])
    try:
        tokenizer = load_tokenizer(base, config)
    except ValueError as exc:
        raise SystemExit(f"Unsupported base DREAM tokenizer: {exc}") from exc
    tokenizer_metadata = base["tokenizer"]
    try:
        pretraining_manifest = json.loads(
            Path(args.pretraining_manifest).read_text(encoding="utf-8")
        )
        test_split = pretraining_manifest["splits"]["test"]
        pretraining_test_hash = test_split["sha256"]
        pretraining_test_groups = set(test_split["source_group_ids"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SystemExit(
            "Invalid pretraining manifest: expected a test split with sha256 and source_group_ids."
        ) from exc
    if (
        pretraining_manifest.get("schema") != "dream-corpus-v1"
        or not isinstance(pretraining_test_hash, str)
        or len(pretraining_test_hash) != 64
        or not pretraining_test_groups
        or any(not isinstance(group, str) or not group for group in pretraining_test_groups)
    ):
        raise SystemExit("Pretraining manifest has invalid held-out test provenance.")
    pretraining_test_path = Path(args.pretraining_test_data)
    if sha256(pretraining_test_path) != pretraining_test_hash:
        raise SystemExit(
            "Pretraining held-out test file hash does not match the supplied manifest."
        )
    pretraining_manifest_hash = sha256(args.pretraining_manifest)

    train_groups = source_groups(train_path)
    validation_groups = source_groups(validation_path)
    if train_groups & validation_groups:
        raise SystemExit("SFT training and validation source groups must be disjoint.")
    train_ids, train_content = record_signatures(train_path)
    validation_ids, validation_content = record_signatures(validation_path)
    if train_ids & validation_ids:
        raise SystemExit("SFT training and validation record IDs must be disjoint.")
    if train_content & validation_content:
        raise SystemExit(
            "Duplicate SFT dialogue content detected across training and validation."
        )
    if (train_groups | validation_groups) & pretraining_test_groups:
        raise SystemExit(
            "SFT training/validation source groups must be disjoint from the pretraining held-out test groups."
        )
    train_examples = load_examples(train_path, tokenizer, config.block_size)
    validation_examples = load_examples(validation_path, tokenizer, config.block_size)

    model = DreamTransformer(config)
    model.load_state_dict(base["model_state"])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    start_step = 0
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=True)
        meta = saved.get("sft_training", {})
        if saved.get("tokenizer") != tokenizer_metadata:
            raise SystemExit("SFT resume tokenizer does not match the base checkpoint.")
        if saved.get("base_checkpoint_sha256") != sha256(args.checkpoint):
            raise SystemExit("SFT resume base checkpoint hash does not match.")
        if meta.get("train_sha256") != train_hash or meta.get("validation_sha256") != validation_hash:
            raise SystemExit("SFT resume dataset hashes do not match.")
        if meta.get("learning_rate") != args.learning_rate or meta.get("batch_size") != args.batch_size:
            raise SystemExit("SFT resume hyperparameters do not match.")
        if meta.get("grad_accum_steps") != args.grad_accum_steps or meta.get("eval_batches") != args.eval_batches:
            raise SystemExit("SFT resume accumulation/evaluation settings do not match.")
        if (
            meta.get("pretraining_test_sha256") != pretraining_test_hash
            or meta.get("pretraining_manifest_sha256") != pretraining_manifest_hash
        ):
            raise SystemExit("SFT resume held-out test manifest does not match.")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        if "torch_rng_state" in saved:
            torch.set_rng_state(saved["torch_rng_state"].cpu())
        if "python_rng_state" in saved:
            random.setstate(saved["python_rng_state"])
        if device == "cuda" and saved.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
        start_step = int(meta.get("completed_steps", 0))
        if start_step < 0:
            raise SystemExit("Invalid SFT completed step count.")

    @torch.no_grad()
    def estimate(examples):
        model.eval()
        losses = []
        for _ in range(args.eval_batches):
            x, y = make_batch(examples, args.batch_size, device)
            _, loss = model(x, y)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("Invalid SFT evaluation loss.")
            losses.append(float(loss.item()))
        model.train()
        return sum(losses) / len(losses)

    def save_checkpoint(completed_steps, train_loss, validation_loss):
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state().cpu(),
            "python_rng_state": random.getstate(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "config": config.__dict__,
            "tokenizer": tokenizer_metadata,
            "base_checkpoint_sha256": sha256(args.checkpoint),
            "sft_training": {
                "method": "assistant-response-only-supervised-finetuning",
                "completed_steps": completed_steps,
                "train_data": str(train_path),
                "train_sha256": train_hash,
                "validation_data": str(validation_path),
                "validation_sha256": validation_hash,
                "pretraining_test_sha256": pretraining_test_hash,
                "pretraining_test_data": str(pretraining_test_path),
                "pretraining_test_source_groups": sorted(pretraining_test_groups),
                "pretraining_manifest_sha256": pretraining_manifest_hash,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "effective_batch_size": args.batch_size * args.grad_accum_steps,
                "eval_batches": args.eval_batches,
                "last_train_loss": train_loss,
                "last_validation_loss": validation_loss,
                "device": device
            }
        }
        temporary = output.with_suffix(output.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(output)
        print(f"Saved DREAM instruction checkpoint at step {completed_steps}: {output}")

    final_step = start_step + args.steps
    for step in range(start_step + 1, final_step + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum_steps):
            x, y = make_batch(train_examples, args.batch_size, device)
            _, loss = model(x, y)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("Invalid SFT training loss.")
            (loss / args.grad_accum_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == start_step + 1 or step % args.eval_every == 0 or step % args.checkpoint_every == 0 or step == final_step:
            train_loss = estimate(train_examples)
            validation_loss = estimate(validation_examples)
            print(json.dumps({
                "step": step,
                "train_loss": round(train_loss, 4),
                "validation_loss": round(validation_loss, 4),
                "method": "assistant-response-only-supervised-finetuning"
            }))
            if step % args.checkpoint_every == 0 or step == final_step:
                save_checkpoint(step, train_loss, validation_loss)


if __name__ == "__main__":
    main()
