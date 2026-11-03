"""Train DREAM's randomly initialized transformer on a plain-text corpus."""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch

from dream_transformer import DreamByteTokenizer, DreamConfig, DreamTransformer


def batches(data, batch_size, block_size, device):
    max_start = len(data) - block_size - 1
    if max_start < 1:
        raise ValueError("Corpus is too small for the configured block_size")
    starts = torch.randint(max_start, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts.tolist()])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts.tolist()])
    return x.to(device), y.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/prepared/train.txt")
    parser.add_argument("--validation-data", required=True,
                        help="Independent source-isolated validation text file")
    parser.add_argument("--out", default="checkpoints/dream-small.pt")
    parser.add_argument("--resume", default="", help="Resume from a DREAM checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = Path(args.data).read_text(encoding="utf-8")
    tokenizer = DreamByteTokenizer()
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data = tokens
    validation_path = Path(args.validation_data)
    validation_text = validation_path.read_text(encoding="utf-8")
    val_data = torch.tensor(tokenizer.encode(validation_text), dtype=torch.long)
    if Path(args.data).resolve() == validation_path.resolve():
        raise SystemExit("Training and validation files must be different.")
    if len(train_data) < args.block_size + 2 or len(val_data) < args.block_size + 2:
        raise SystemExit(
            "Corpus too small. Supply a properly licensed, high-quality corpus "
            "with enough UTF-8 bytes for train and validation splits."
        )

    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1 or args.checkpoint_every < 1:
        raise SystemExit("steps, batch-size, eval-every and checkpoint-every must be positive.")
    if args.learning_rate <= 0:
        raise SystemExit("learning-rate must be positive.")

    def file_sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    config = DreamConfig(
        block_size=args.block_size, d_model=args.d_model,
        n_heads=args.heads, n_layers=args.layers
    )
    model = DreamTransformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.1
    )
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if checkpoint.get("config") != config.__dict__:
            raise SystemExit("Resume checkpoint architecture does not match this run.")
        training_meta = checkpoint.get("training", {})
        if training_meta.get("corpus_sha256") != file_sha256(args.data):
            raise SystemExit("Resume checkpoint training corpus hash does not match.")
        if training_meta.get("validation_sha256") != file_sha256(validation_path):
            raise SystemExit("Resume checkpoint validation corpus hash does not match.")
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" not in checkpoint:
            raise SystemExit("Resume checkpoint does not contain optimizer state.")
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(training_meta.get("completed_steps", 0))
        if start_step < 0:
            raise SystemExit("Resume checkpoint has invalid completed_steps.")
        print(f"Resuming DREAM training at step {start_step}.")
    print(json.dumps({
        "project": "DREAM",
        "initialization": "random",
        "device": device,
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_bytes": len(train_data),
        "validation_bytes": len(val_data),
        "config": config.__dict__
    }, indent=2))

    @torch.no_grad()
    def estimate(data):
        model.eval()
        x, y = batches(data, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        model.train()
        return float(loss.item())

    def save_checkpoint(completed_steps, train_loss=None, val_loss=None):
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        training_meta = {
            "steps": args.steps,
            "completed_steps": completed_steps,
            "seed": args.seed,
            "corpus": str(Path(args.data)),
            "corpus_sha256": file_sha256(args.data),
            "validation_corpus": str(validation_path),
            "validation_sha256": file_sha256(validation_path),
            "initialization": "random" if not args.resume else "resumed-from-dream-checkpoint",
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "device": device,
            "last_train_loss": train_loss,
            "last_validation_loss": val_loss
        }
        checkpoint = {
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "config": config.__dict__,
            "tokenizer": "utf8-byte-v1",
            "training": training_meta
        }
        output = Path(args.out)
        temporary = output.with_suffix(output.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(output)
        print(f"Saved DREAM checkpoint at step {completed_steps}: {output}")

    for step in range(start_step + 1, start_step + args.steps + 1):
        model.train()
        x, y = batches(train_data, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            train_loss = estimate(train_data)
            val_loss = estimate(val_data)
            print(json.dumps({
                "step": step,
                "train_loss": round(train_loss, 4),
                "validation_loss": round(val_loss, 4),
                "perplexity": round(math.exp(min(val_loss, 20)), 3)
            }))
            if step % args.checkpoint_every == 0 or step == start_step + args.steps:
                save_checkpoint(step, train_loss, val_loss)

    if start_step == start_step + args.steps:
        save_checkpoint(start_step)


if __name__ == "__main__":
    main()
