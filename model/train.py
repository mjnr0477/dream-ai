"""Train DREAM's randomly initialized transformer on a plain-text corpus."""
import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path

import torch

from dream_transformer import DreamByteTokenizer, DreamConfig, DreamTransformer
from dream_bpe import DreamBPETokenizer


def batches(data, batch_size, block_size, device):
    max_start = len(data) - block_size
    if max_start < 1:
        raise ValueError("Corpus is too small for the configured block_size")
    starts = torch.randint(max_start, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts.tolist()])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts.tolist()])
    return x.to(device), y.to(device)


def split_language_streams(text, tokenizer, block_size):
    """Encode each language-tagged document stream separately for balanced sampling."""
    marker = re.compile(r"(?m)^<\|language:([A-Za-z]{2,3}(?:[-+][A-Za-z0-9]{2,8})*)\|>\n")
    matches = list(marker.finditer(text))
    if not matches:
        raise ValueError("Balanced sampling requires <|language:TAG|> document markers.")
    streams = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        language = match.group(1).lower()
        content = text[match.end():end]
        encoded = tokenizer.encode(content)
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
    parser.add_argument("--manifest", default="",
                        help="Optional prepare_corpus manifest; verifies train/validation/test hashes and source isolation")
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
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tokenizer", choices=("auto", "byte", "bpe"), default="auto",
                        help="Tokenizer mode; auto preserves the tokenizer when resuming")
    parser.add_argument("--tokenizer-vocab-size", type=int, default=2048)
    parser.add_argument("--language-sampling", choices=("proportional", "balanced"),
                        default="proportional",
                        help="Sample token windows proportionally or choose languages uniformly")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = Path(args.data).read_text(encoding="utf-8")
    validation_path = Path(args.validation_data)
    validation_text = validation_path.read_text(encoding="utf-8")
    resume_checkpoint = (torch.load(args.resume, map_location="cpu", weights_only=True)
                         if args.resume else None)
    saved_tokenizer = resume_checkpoint.get("tokenizer", "utf8-byte-v1") if resume_checkpoint else None
    if args.resume and args.tokenizer not in ("auto", "byte" if saved_tokenizer == "utf8-byte-v1" else "bpe"):
        raise SystemExit("Requested tokenizer does not match the resume checkpoint.")
    tokenizer_mode = ("byte" if saved_tokenizer == "utf8-byte-v1" else "bpe") if args.resume else ("byte" if args.tokenizer in ("auto", "byte") else "bpe")
    if tokenizer_mode == "bpe":
        if args.resume:
            tokenizer_payload = resume_checkpoint.get("tokenizer_data")
            if not tokenizer_payload:
                raise SystemExit("BPE resume checkpoint is missing tokenizer data.")
            tokenizer = DreamBPETokenizer.from_dict(tokenizer_payload)
        else:
            tokenizer = DreamBPETokenizer.train(text, vocab_size=args.tokenizer_vocab_size)
        tokenizer_name = DreamBPETokenizer.format_version
    else:
        tokenizer = DreamByteTokenizer()
        tokenizer_name = "utf8-byte-v1"
    train_data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
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
            "with enough tokens for train and validation splits."
        )

    if (
        args.steps < 1 or args.batch_size < 1 or args.grad_accum_steps < 1
        or args.eval_every < 1 or args.checkpoint_every < 1 or args.eval_batches < 1
    ):
        raise SystemExit("steps, batch-size, grad-accum-steps, eval-every, eval-batches and checkpoint-every must be positive.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise SystemExit("learning-rate must be finite and positive.")

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

    manifest_hash = None
    if args.manifest:
        manifest_path = Path(args.manifest)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Could not read corpus manifest: {exc}") from exc
        if not isinstance(manifest, dict):
            raise SystemExit("Corpus manifest must be a JSON object.")
        if manifest.get("schema") != "dream-corpus-v1":
            raise SystemExit("Unsupported corpus manifest schema.")
        split_metadata = manifest.get("splits")
        if not isinstance(split_metadata, dict):
            raise SystemExit("Corpus manifest is missing split metadata.")

        supplied_paths = {
            "train": Path(args.data),
            "validation": validation_path,
            "test": manifest_path.parent / "test.txt"
        }
        split_groups = {}
        for split, split_path in supplied_paths.items():
            details = split_metadata.get(split)
            if not isinstance(details, dict):
                raise SystemExit(f"Corpus manifest is missing the {split} split.")
            if not split_path.is_file():
                raise SystemExit(f"Manifest-verified {split} file does not exist: {split_path}")
            actual_hash = file_sha256(split_path)
            if details.get("sha256") != actual_hash:
                raise SystemExit(f"{split.capitalize()} corpus hash does not match the manifest.")
            group_ids = details.get("source_group_ids")
            if (
                not isinstance(group_ids, list)
                or not group_ids
                or any(not isinstance(group, str) or not group.strip() for group in group_ids)
                or len(group_ids) != len(set(group_ids))
            ):
                raise SystemExit(f"Manifest has invalid {split} source-group metadata.")
            split_groups[split] = set(group_ids)

        split_names = ("train", "validation", "test")
        for index, left in enumerate(split_names):
            for right in split_names[index + 1:]:
                if split_groups[left] & split_groups[right]:
                    raise SystemExit(
                        f"Corpus manifest has source-group leakage between {left} and {right}."
                    )
        manifest_hash = file_sha256(manifest_path)

    config = DreamConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size, d_model=args.d_model,
        n_heads=args.heads, n_layers=args.layers
    )
    model = DreamTransformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.1
    )
    start_step = 0
    if args.resume:
        checkpoint = resume_checkpoint
        if checkpoint.get("tokenizer", "utf8-byte-v1") != tokenizer_name:
            raise SystemExit("Resume checkpoint tokenizer does not match this run.")
        if checkpoint.get("config") != config.__dict__:
            raise SystemExit("Resume checkpoint architecture does not match this run.")
        training_meta = checkpoint.get("training", {})
        if training_meta.get("corpus_manifest_sha256") != manifest_hash:
            raise SystemExit("Resume checkpoint corpus manifest does not match this run.")
        if training_meta.get("data_provenance_verified", False) != (manifest_hash is not None):
            raise SystemExit("Resume checkpoint data provenance status does not match this run.")
        if training_meta.get("corpus_sha256") != training_hash:
            raise SystemExit("Resume checkpoint training corpus hash does not match.")
        if training_meta.get("validation_sha256") != validation_hash:
            raise SystemExit("Resume checkpoint validation corpus hash does not match.")
        if training_meta.get("learning_rate") != args.learning_rate:
            raise SystemExit("Resume checkpoint learning rate does not match.")
        if training_meta.get("batch_size") != args.batch_size:
            raise SystemExit("Resume checkpoint batch size does not match.")
        if training_meta.get("grad_accum_steps") != args.grad_accum_steps:
            raise SystemExit("Resume checkpoint gradient accumulation does not match.")
        if training_meta.get("eval_batches") != args.eval_batches:
            raise SystemExit("Resume checkpoint evaluation batch count does not match.")
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
            raise SystemExit(
                "Balanced-sampling resume checkpoint is missing Python RNG state."
            )
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
        "training_bytes": len(text.encode("utf-8")),
        "validation_bytes": len(validation_text.encode("utf-8")),
        "micro_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "config": config.__dict__
    }, indent=2))

    @torch.no_grad()
    def estimate(data):
        model.eval()
        losses = []
        for _ in range(args.eval_batches):
            x, y = batches(data, args.batch_size, args.block_size, device)
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
            "corpus_manifest_sha256": manifest_hash,
            "data_provenance_verified": manifest_hash is not None,
            "initialization": "random" if not args.resume else "resumed-from-dream-checkpoint",
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
            "eval_batches": args.eval_batches,
            "language_sampling": args.language_sampling,
            "training_tokens_by_language": ({key: len(value) for key, value in language_streams.items()}
                                             if language_streams is not None else None),
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
            "tokenizer": tokenizer_name,
            "tokenizer_data": tokenizer.to_dict() if tokenizer_mode == "bpe" else None,
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_step = start_step + args.steps
        if (
            step == start_step + 1
            or step % args.eval_every == 0
            or step % args.checkpoint_every == 0
            or step == final_step
        ):
            train_loss = estimate(train_data)
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
