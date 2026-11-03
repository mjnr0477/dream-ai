"""Evaluate a DREAM checkpoint on a held-out UTF-8 text corpus."""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

from dream_transformer import DreamByteTokenizer, DreamConfig, DreamTransformer


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="Held-out validation/test text")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-prompt", default="DREAM AI:")
    parser.add_argument("--sample-tokens", type=int, default=120)
    parser.add_argument("--out", default="eval-results/native-checkpoint.json")
    args = parser.parse_args()

    if args.batch_size < 1 or args.batches < 1:
        raise SystemExit("batch-size and batches must be positive.")
    if not 0 <= args.sample_tokens <= 256:
        raise SystemExit("sample-tokens must be between 0 and 256.")

    checkpoint_path = Path(args.checkpoint)
    data_path = Path(args.data)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("tokenizer") != "utf8-byte-v1":
        raise SystemExit("Unsupported checkpoint tokenizer.")
    config = DreamConfig(**checkpoint["config"])
    tokenizer = DreamByteTokenizer()
    text = data_path.read_text(encoding="utf-8")
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if len(tokens) < config.block_size + 2:
        raise SystemExit("Evaluation corpus is too small for the checkpoint context length.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DreamTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    tokens = tokens.to(device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    losses = []
    max_start = len(tokens) - config.block_size - 1

    with torch.inference_mode():
        for _ in range(args.batches):
            starts = torch.randint(
                max_start, (args.batch_size,), generator=generator, device="cpu"
            ).tolist()
            x = torch.stack([
                tokens[start:start + config.block_size] for start in starts
            ])
            y = torch.stack([
                tokens[start + 1:start + config.block_size + 1] for start in starts
            ])
            _, loss = model(x, y)
            losses.append(float(loss.item()))

        prompt_ids = tokenizer.encode(args.sample_prompt)
        if not prompt_ids:
            prompt_ids = [32]
        prompt = torch.tensor(
            [prompt_ids[-config.block_size:]], dtype=torch.long, device=device
        )
        torch.manual_seed(args.seed)
        generated = model.generate(
            prompt, max_new_tokens=args.sample_tokens, temperature=0.8, top_k=40
        )[0].tolist()
        sample = tokenizer.decode(generated[len(prompt[0]):])

    mean_loss = sum(losses) / len(losses)
    result = {
        "schema": "dream-evaluation-v1",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "evaluation_corpus": str(data_path),
        "evaluation_corpus_sha256": sha256(data_path),
        "evaluation_bytes": len(tokens),
        "architecture": config.__dict__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": device,
        "seed": args.seed,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "mean_cross_entropy_nats_per_byte": mean_loss,
        "bits_per_byte": mean_loss / math.log(2),
        "perplexity_per_byte": math.exp(min(mean_loss, 20)),
        "sample_prompt": args.sample_prompt,
        "sample_generation": sample,
        "limitations": [
            "Byte-level next-token metrics do not measure instruction following or factual accuracy.",
            "A low validation loss does not certify safety, reasoning, usefulness or generalization.",
            "Sample generation is qualitative and must be reviewed by humans."
        ]
    }

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
