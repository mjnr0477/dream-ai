"""Train DREAM's randomly initialized transformer on a plain-text corpus."""
import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

import torch

from dream_bpe import DreamBPETokenizer
from dream_transformer import DreamByteTokenizer, DreamConfig, DreamTransformer
from license_policy import POLICY_VERSION, classify_record_license



SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
REQUIRED_SPLITS = ("train", "validation", "test")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_https_url(value):
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
        return (
            parsed.scheme == "https"
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
        )
    except ValueError:
        return False


def verify_pretraining_corpus_manifest(manifest_path, train_path, validation_path):
    """Fail closed unless exact text files have verified license provenance."""
    manifest_path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("Pretraining corpus manifest is missing or invalid.") from exc

    if not isinstance(manifest, dict) or manifest.get("schema") != "dream-corpus-v1":
        raise SystemExit("Pretraining corpus manifest has an unsupported schema.")
    if manifest.get("identifier_format") != "sha256-domain-separated-v1":
        raise SystemExit("Pretraining corpus manifest has an unsupported identifier format.")
    if manifest.get("license_policy_version") != POLICY_VERSION:
        raise SystemExit("Pretraining corpus license policy version is missing or stale.")
    if not isinstance(manifest.get("input_sha256"), str) or not SHA256_RE.fullmatch(manifest["input_sha256"]):
        raise SystemExit("Pretraining corpus manifest has an invalid source hash.")

    split_manifest = manifest.get("splits")
    if not isinstance(split_manifest, dict) or any(name not in split_manifest for name in REQUIRED_SPLITS):
        raise SystemExit("Pretraining corpus manifest must include train, validation and test splits.")

    requested_paths = {
        "train": Path(train_path).resolve(),
        "validation": Path(validation_path).resolve(),
    }
    seen_record_ids = set()
    seen_source_groups = set()
    aggregate_policy_counts = Counter()
    verified = {}

    for split_name in REQUIRED_SPLITS:
        split = split_manifest.get(split_name)
        if not isinstance(split, dict):
            raise SystemExit(f"Pretraining corpus manifest has an invalid {split_name} split.")

        text_name = split.get("path")
        if not isinstance(text_name, str) or not text_name or Path(text_name).name != text_name:
            raise SystemExit(f"Pretraining {split_name} text path is invalid.")
        text_path = requested_paths.get(split_name, (manifest_path.parent / text_name).resolve())
        if text_path.name != text_name or not text_path.is_file():
            raise SystemExit(f"Pretraining {split_name} file does not match its corpus manifest path.")
        if split.get("sha256") != file_sha256(text_path):
            raise SystemExit(f"Pretraining {split_name} text hash does not match its corpus manifest.")
        if split.get("utf8_bytes") != text_path.stat().st_size:
            raise SystemExit(f"Pretraining {split_name} byte count does not match its corpus manifest.")

        record_count = split.get("records")
        if not isinstance(record_count, int) or isinstance(record_count, bool) or record_count < 1:
            raise SystemExit(f"Pretraining {split_name} record count is invalid.")

        provenance = split.get("provenance")
        if not isinstance(provenance, dict):
            raise SystemExit(f"Pretraining {split_name} provenance sidecar is missing.")
        provenance_name = provenance.get("path")
        if (
            not isinstance(provenance_name, str)
            or not provenance_name
            or Path(provenance_name).name != provenance_name
            or provenance.get("contains_training_text") is not False
            or provenance.get("contains_raw_record_or_group_ids") is not False
            or provenance.get("offset_semantics") != "utf8-byte-start-inclusive-end-exclusive-v1"
        ):
            raise SystemExit(f"Pretraining {split_name} provenance sidecar metadata is invalid.")
        provenance_path = (manifest_path.parent / provenance_name).resolve()
        if provenance_path.parent != manifest_path.parent or not provenance_path.is_file():
            raise SystemExit(f"Pretraining {split_name} provenance sidecar is missing or unsafe.")
        provenance_hash = file_sha256(provenance_path)
        if provenance.get("sha256") != provenance_hash:
            raise SystemExit(f"Pretraining {split_name} provenance hash does not match its manifest.")
        if provenance.get("records") != record_count:
            raise SystemExit(f"Pretraining {split_name} provenance count does not match its text split.")

        split_record_ids = set()
        split_groups = set()
        policy_counts = Counter()
        text_segments = []
        with provenance_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise SystemExit(f"Blank provenance row in {split_name} sidecar at line {line_number}.")
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid JSON in {split_name} provenance at line {line_number}.") from exc
                if not isinstance(item, dict):
                    raise SystemExit(f"Invalid provenance row in {split_name} sidecar at line {line_number}.")

                for field in (
                    "record_id_sha256", "source_group_sha256", "source", "source_url",
                    "license", "license_url", "rights_basis", "source_date", "language",
                    "license_policy_code"
                ):
                    value = item.get(field)
                    if not isinstance(value, str) or not value.strip():
                        raise SystemExit(f"Missing {field} in {split_name} provenance at line {line_number}.")
                if not SHA256_RE.fullmatch(item["record_id_sha256"]) or not SHA256_RE.fullmatch(item["source_group_sha256"]):
                    raise SystemExit(f"Invalid hashed identifier in {split_name} provenance at line {line_number}.")
                text_start = item.get("text_byte_start")
                text_end = item.get("text_byte_end")
                text_digest = item.get("text_sha256")
                if (
                    not isinstance(text_start, int) or isinstance(text_start, bool) or text_start < 0
                    or not isinstance(text_end, int) or isinstance(text_end, bool) or text_end <= text_start
                    or not isinstance(text_digest, str) or not SHA256_RE.fullmatch(text_digest)
                ):
                    raise SystemExit(f"Invalid text byte attestation in {split_name} provenance at line {line_number}.")
                text_segments.append((text_start, text_end, text_digest))
                if any(field in item for field in ("id", "source_group", "text", "input_text", "target_text")):
                    raise SystemExit(f"Raw identifiers or training text found in {split_name} provenance.")
                if not _valid_https_url(item["source_url"]) or not _valid_https_url(item["license_url"]):
                    raise SystemExit(f"Non-HTTPS provenance URL in {split_name} sidecar at line {line_number}.")
                try:
                    parsed_date = date.fromisoformat(item["source_date"])
                    if parsed_date.isoformat() != item["source_date"]:
                        raise ValueError("invalid date format")
                except ValueError as exc:
                    raise SystemExit(f"Invalid source date in {split_name} provenance at line {line_number}.") from exc

                for flag in ("rights_confirmed", "privacy_reviewed", "quality_reviewed", "approval_granted"):
                    if item.get(flag) is not True:
                        raise SystemExit(f"{flag} is not confirmed in {split_name} provenance at line {line_number}.")
                if not isinstance(item.get("attribution_required"), bool):
                    raise SystemExit(f"Attribution status is missing in {split_name} provenance at line {line_number}.")

                license_result = classify_record_license(item)
                if (
                    license_result.get("status") != "allowed"
                    or license_result.get("code") != item["license_policy_code"]
                    or license_result.get("attribution_required") is not item["attribution_required"]
                ):
                    raise SystemExit(
                        f"License policy did not clear {split_name} provenance row {line_number}: "
                        f"{license_result.get('code', 'unknown')}"
                    )

                record_id = item["record_id_sha256"]
                source_group = item["source_group_sha256"]
                if record_id in split_record_ids:
                    raise SystemExit(f"Duplicate hashed record identifier in {split_name} provenance.")
                split_record_ids.add(record_id)
                split_groups.add(source_group)
                policy_counts[item["license_policy_code"]] += 1

        if len(split_record_ids) != record_count or len(split_groups) != split.get("source_groups"):
            raise SystemExit(f"Pretraining {split_name} provenance identifiers do not match the manifest.")
        if len(text_segments) != record_count:
            raise SystemExit(f"Pretraining {split_name} text attestations do not match the record count.")

        # Prove that every provenance row identifies the exact UTF-8 byte range
        # in the actual text file that the model will consume.
        ordered_segments = sorted(text_segments, key=lambda segment: segment[0])
        cursor = 0
        with text_path.open("rb") as text_handle:
            for index, (text_start, text_end, text_digest) in enumerate(ordered_segments):
                if text_start != cursor:
                    raise SystemExit(f"Pretraining {split_name} text/provenance offsets are not contiguous.")
                text_handle.seek(text_start)
                segment_bytes = text_handle.read(text_end - text_start)
                if len(segment_bytes) != text_end - text_start:
                    raise SystemExit(f"Pretraining {split_name} text/provenance range exceeds the text file.")
                if hashlib.sha256(segment_bytes).hexdigest() != text_digest:
                    raise SystemExit(f"Pretraining {split_name} record text does not match its license provenance.")
                cursor = text_end
                if index < len(ordered_segments) - 1:
                    text_handle.seek(cursor)
                    if text_handle.read(2) != b"\\n\\n":
                        raise SystemExit(f"Pretraining {split_name} text/provenance separator is invalid.")
                    cursor += 2
                else:
                    text_handle.seek(cursor)
                    if text_handle.read() != b"\\n":
                        raise SystemExit(f"Pretraining {split_name} text/provenance ending is invalid.")
        if sorted(split_record_ids) != split.get("record_ids"):
            raise SystemExit(f"Pretraining {split_name} record identifiers do not match the manifest.")
        if sorted(split_groups) != split.get("source_group_ids"):
            raise SystemExit(f"Pretraining {split_name} source groups do not match the manifest.")
        if dict(sorted(policy_counts.items())) != split.get("license_policy_counts"):
            raise SystemExit(f"Pretraining {split_name} license-policy counts do not match the manifest.")

        if seen_record_ids & split_record_ids or seen_source_groups & split_groups:
            raise SystemExit("Pretraining source records or groups overlap across corpus splits.")
        seen_record_ids.update(split_record_ids)
        seen_source_groups.update(split_groups)
        aggregate_policy_counts.update(policy_counts)
        verified[split_name] = {
            "text_sha256": split["sha256"],
            "provenance_sha256": provenance_hash,
            "records": record_count,
            "source_groups": len(split_groups),
        }

    if manifest.get("records") != sum(item["records"] for item in verified.values()):
        raise SystemExit("Pretraining corpus record total does not match the split manifest.")
    if dict(sorted(aggregate_policy_counts.items())) != manifest.get("license_policy_counts"):
        raise SystemExit("Pretraining corpus license-policy totals do not match the split manifests.")

    return {
        "manifest_sha256": file_sha256(manifest_path),
        "license_policy_version": POLICY_VERSION,
        "splits": verified,
    }


def batches(data, batch_size, block_size, device):
    # max_start is inclusive: the final valid window predicts the last token.
    max_start = len(data) - block_size - 1
    if max_start < 0:
        raise ValueError("Corpus is too small for the configured block_size")
    starts = torch.randint(max_start + 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts.tolist()])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts.tolist()])
    return x.to(device), y.to(device)


def split_language_streams(text, tokenizer, block_size):
    """Encode language-tagged documents separately for balanced sampling."""
    marker = re.compile(r"(?<!\S)<\|language:([A-Za-z]{2,3}(?:[-+][A-Za-z0-9]{2,8})*)\|>\n")
    matches = list(marker.finditer(text))
    if not matches:
        raise ValueError("Balanced sampling requires <|language:TAG|> document markers.")
    streams = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        language = match.group(1).lower()
        encoded = tokenizer.encode(text[match.end():end])
        if encoded:
            streams.setdefault(language, []).extend(encoded)
    too_small = sorted(language for language, tokens in streams.items()
                        if len(tokens) < block_size + 2)
    if too_small:
        raise ValueError(
            "Balanced sampling needs at least block_size + 2 tokens per language; "
            "insufficient: " + ", ".join(too_small)
        )
    return {language: torch.tensor(tokens, dtype=torch.long)
            for language, tokens in sorted(streams.items())}


def balanced_batches(streams, batch_size, block_size, device):
    languages = tuple(streams)
    if not languages:
        raise ValueError("No language streams are available for balanced sampling.")
    selected = [random.choice(languages) for _ in range(batch_size)]
    xs, ys = [], []
    for language in selected:
        data = streams[language]
        max_start = len(data) - block_size
        start = random.randrange(max_start)
        xs.append(data[start:start + block_size])
        ys.append(data[start + 1:start + block_size + 1])
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/prepared/train.txt")
    parser.add_argument("--validation-data", required=True,
                        help="Independent source-isolated validation text file")
    parser.add_argument("--corpus-manifest", required=True,
                        help="Verified manifest.json from model/prepare_corpus.py")
    parser.add_argument("--out", default="checkpoints/dream-small.pt")
    parser.add_argument("--resume", default="", help="Resume from a DREAM checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Micro-batches accumulated per optimizer update")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Positive finite gradient clipping threshold")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Trade extra compute for lower activation memory during training")
    parser.add_argument("--tokenizer", choices=("byte", "bpe"), default="byte",
                        help="Tokenizer family; byte preserves the legacy default")
    parser.add_argument("--vocab-size", type=int, default=1024,
                        help="BPE vocabulary size when --tokenizer=bpe")
    parser.add_argument("--language-sampling", choices=("proportional", "balanced"),
                        default="proportional",
                        help="Sample token windows proportionally or choose languages uniformly")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    validation_path = Path(args.validation_data)
    corpus_identity = verify_pretraining_corpus_manifest(
        args.corpus_manifest, args.data, validation_path
    )
    text = Path(args.data).read_text(encoding="utf-8")
    if args.tokenizer == "bpe":
        try:
            tokenizer = DreamBPETokenizer.train(text, vocab_size=args.vocab_size)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"Invalid BPE tokenizer configuration: {exc}") from exc
        tokenizer_metadata = tokenizer.to_dict()
    else:
        tokenizer = DreamByteTokenizer()
        tokenizer_metadata = "utf8-byte-v1"
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data = tokens
    validation_text = validation_path.read_text(encoding="utf-8")
    val_data = torch.tensor(tokenizer.encode(validation_text), dtype=torch.long)
    language_streams = None
    if args.language_sampling == "balanced":
        try:
            language_streams = split_language_streams(text, tokenizer, args.block_size)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    if Path(args.data).resolve() == validation_path.resolve():
        raise SystemExit("Training and validation files must be different.")
    if len(train_data) < args.block_size + 2 or len(val_data) < args.block_size + 2:
        raise SystemExit(
            "Corpus too small. Supply a properly licensed, high-quality corpus "
            "with enough UTF-8 bytes for train and validation splits."
        )

    if (
        args.steps < 1 or args.batch_size < 1 or args.grad_accum_steps < 1
        or args.eval_every < 1 or args.checkpoint_every < 1 or args.eval_batches < 1
    ):
        raise SystemExit("steps, batch-size, grad-accum-steps, eval-every, eval-batches and checkpoint-every must be positive.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise SystemExit("learning-rate must be finite and positive.")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        raise SystemExit("max-grad-norm must be finite and positive.")

    def file_sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    training_hash = file_sha256(args.data)
    validation_hash = file_sha256(validation_path)
    if training_hash == validation_hash:
        raise SystemExit("Training and validation content hashes must differ.")

    config = DreamConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size, d_model=args.d_model,
        n_heads=args.heads, n_layers=args.layers
    )
    model = DreamTransformer(config).to(device)
    model.gradient_checkpointing = args.gradient_checkpointing
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.1
    )
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if checkpoint.get("config") != config.__dict__:
            raise SystemExit("Resume checkpoint architecture does not match this run.")
        if checkpoint.get("tokenizer") != tokenizer_metadata:
            raise SystemExit("Resume checkpoint tokenizer does not match this run.")
        training_meta = checkpoint.get("training", {})
        if training_meta.get("corpus_sha256") != training_hash:
            raise SystemExit("Resume checkpoint training corpus hash does not match.")
        if training_meta.get("validation_sha256") != validation_hash:
            raise SystemExit("Resume checkpoint validation corpus hash does not match.")
        if training_meta.get("corpus_manifest_sha256") != corpus_identity["manifest_sha256"]:
            raise SystemExit("Resume checkpoint corpus manifest does not match.")
        if training_meta.get("license_policy_version") != POLICY_VERSION:
            raise SystemExit("Resume checkpoint license policy version does not match.")
        for split_name in ("train", "validation"):
            expected_key = f"{split_name}_provenance_sha256"
            if training_meta.get(expected_key) != corpus_identity["splits"][split_name]["provenance_sha256"]:
                raise SystemExit(f"Resume checkpoint {split_name} provenance does not match.")
        if training_meta.get("learning_rate") != args.learning_rate:
            raise SystemExit("Resume checkpoint learning rate does not match.")
        if training_meta.get("max_grad_norm", 1.0) != args.max_grad_norm:
            raise SystemExit("Resume checkpoint max gradient norm does not match.")
        if training_meta.get("batch_size") != args.batch_size:
            raise SystemExit("Resume checkpoint batch size does not match.")
        if training_meta.get("grad_accum_steps") != args.grad_accum_steps:
            raise SystemExit("Resume checkpoint gradient accumulation does not match.")
        if training_meta.get("eval_batches") != args.eval_batches:
            raise SystemExit("Resume checkpoint evaluation batch count does not match.")
        if training_meta.get("eval_every") != args.eval_every:
            raise SystemExit("Resume checkpoint evaluation interval does not match.")
        if training_meta.get("gradient_checkpointing", False) != args.gradient_checkpointing:
            raise SystemExit("Resume checkpoint gradient checkpointing setting does not match.")
        if training_meta.get("language_sampling", "proportional") != args.language_sampling:
            raise SystemExit("Resume checkpoint language sampling mode does not match.")
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" not in checkpoint:
            raise SystemExit("Resume checkpoint does not contain optimizer state.")
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        elif args.language_sampling == "balanced":
            raise SystemExit("Balanced-sampling resume checkpoint is missing Python RNG state.")
        if device == "cuda" and checkpoint.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        start_step = int(training_meta.get("completed_steps", 0))
        if start_step < 0:
            raise SystemExit("Resume checkpoint has invalid completed_steps.")
        print(f"Resuming DREAM training at step {start_step}.")
    print(json.dumps({
        "project": "DREAM",
        "initialization": "resumed" if args.resume else "random",
        "device": device,
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_tokens": len(train_data),
        "training_tokens_by_language": ({key: len(value) for key, value in language_streams.items()}
                                         if language_streams is not None else None),
        "language_sampling": args.language_sampling,
        "validation_tokens": len(val_data),
        "micro_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "config": config.__dict__
    }, indent=2))

    @torch.no_grad()
    def estimate(data, language_streams_for_sampling=None):
        model.eval()
        losses = []
        for _ in range(args.eval_batches):
            if language_streams_for_sampling is None:
                x, y = batches(data, args.batch_size, args.block_size, device)
            else:
                x, y = balanced_batches(
                    language_streams_for_sampling, args.batch_size,
                    args.block_size, device
                )
            _, loss = model(x, y)
            losses.append(float(loss.item()))
        model.train()
        return sum(losses) / len(losses)

    def save_checkpoint(completed_steps, train_loss=None, val_loss=None):
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        training_meta = {
            "steps": args.steps,
            "completed_steps": completed_steps,
            "seed": args.seed,
            "corpus": str(Path(args.data)),
            "corpus_sha256": training_hash,
            "validation_corpus": str(validation_path),
            "validation_sha256": validation_hash,
            "corpus_manifest": str(Path(args.corpus_manifest)),
            "corpus_manifest_sha256": corpus_identity["manifest_sha256"],
            "license_policy_version": POLICY_VERSION,
            "train_provenance_sha256": corpus_identity["splits"]["train"]["provenance_sha256"],
            "validation_provenance_sha256": corpus_identity["splits"]["validation"]["provenance_sha256"],
            "initialization": "random" if not args.resume else "resumed-from-dream-checkpoint",
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "max_grad_norm": args.max_grad_norm,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
            "eval_batches": args.eval_batches,
            "eval_every": args.eval_every,
            "gradient_checkpointing": args.gradient_checkpointing,
            "language_sampling": args.language_sampling,
            "train_loss_sampling": args.language_sampling,
            "validation_loss_sampling": "proportional",
            "training_tokens_by_language": ({key: len(value) for key, value in language_streams.items()}
                                             if language_streams is not None else None),
            "tokenizer_format": "dream-bpe-v1" if args.tokenizer == "bpe" else "utf8-byte-v1",
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "device": device,
            "last_train_loss": train_loss,
            "last_validation_loss": val_loss
        }
        checkpoint = {
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state().cpu(),
            "python_rng_state": random.getstate(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "config": config.__dict__,
            "tokenizer": tokenizer_metadata,
            "training": training_meta
        }
        output = Path(args.out)
        temporary = output.with_suffix(output.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(output)
        print(f"Saved DREAM checkpoint at step {completed_steps}: {output}")

    for step in range(start_step + 1, start_step + args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        for _ in range(args.grad_accum_steps):
            if language_streams is None:
                x, y = batches(train_data, args.batch_size, args.block_size, device)
            else:
                x, y = balanced_batches(language_streams, args.batch_size, args.block_size, device)
            _, loss = model(x, y)
            micro_losses.append(float(loss.detach().item()))
            (loss / args.grad_accum_steps).backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.max_grad_norm, error_if_nonfinite=True
        )
        optimizer.step()
        final_step = start_step + args.steps
        if (
            step == start_step + 1
            or step % args.eval_every == 0
            or step % args.checkpoint_every == 0
            or step == final_step
        ):
            train_loss = estimate(train_data, language_streams)
            val_loss = estimate(val_data)
            print(json.dumps({
                "step": step,
                "train_loss": round(train_loss, 4),
                "validation_loss": round(val_loss, 4),
                "perplexity": round(math.exp(min(val_loss, 20)), 3)
            }))
            if step % args.checkpoint_every == 0 or step == final_step:
                save_checkpoint(step, train_loss, val_loss)

    if start_step == start_step + args.steps:
        save_checkpoint(start_step)


if __name__ == "__main__":
    main()
