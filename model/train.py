"""Train DREAM's randomly initialized transformer on a plain-text corpus."""
import argparse
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
    parser.add_argument("--validation-data", default="data/prepared/validation.txt")
    parser.add_argument("--out", default="checkpoints/dream-small.pt")
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
    if validation_path.exists():
        validation_text = validation_path.read_text(encoding="utf-8")
        val_data = torch.tensor(tokenizer.encode(validation_text), dtype=torch.long)
    else:
        split = int(len(tokens) * 0.9)
        train_data, val_data = tokens[:split], tokens[split:]
    if len(train_data) < args.block_size + 2 or len(val_data) < args.block_size + 2:
        raise SystemExit(
            "Corpus too small. Supply a properly licensed, high-quality corpus "
            "with enough UTF-8 bytes for train and validation splits."
        )

    config = DreamConfig(
        block_size=args.block_size, d_model=args.d_model,
        n_heads=args.heads, n_layers=args.layers
    )
    model = DreamTransformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.1
    )
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

    for step in range(1, args.steps + 1):
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

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.cpu().state_dict(),
        "config": config.__dict__,
        "tokenizer": "utf8-byte-v1",
        "training": {
            "steps": args.steps,
            "seed": args.seed,
            "corpus": str(Path(args.data)),
            "validation_corpus": str(validation_path) if validation_path.exists() else "tail-10-percent",
            "initialization": "random"
        }
    }, output)
    print("Saved DREAM checkpoint:", output)


if __name__ == "__main__":
    main()
