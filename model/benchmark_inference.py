"""Measure native DREAM checkpoint generation latency and throughput."""
import argparse
import hashlib
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from dream_transformer import DreamByteTokenizer, DreamConfig, DreamTransformer
from dream_bpe import DreamBPETokenizer


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, fraction):
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    tokenizer_name = checkpoint.get("tokenizer", "utf8-byte-v1")
    if tokenizer_name == "utf8-byte-v1":
        tokenizer = DreamByteTokenizer()
    elif tokenizer_name == DreamBPETokenizer.format_version:
        tokenizer = DreamBPETokenizer.from_dict(checkpoint["tokenizer_data"])
    else:
        raise ValueError("Unsupported DREAM tokenizer in checkpoint.")
    config = DreamConfig(**checkpoint["config"])
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Checkpoint vocabulary does not match its tokenizer.")
    model = DreamTransformer(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    return model, tokenizer, config, tokenizer_name


def benchmark(model, tokenizer, config, prompt, device, warmup, runs,
              max_new_tokens, temperature=0.0):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a non-negative integer.")
    if type(runs) is not int or runs < 1:
        raise ValueError("runs must be a positive integer.")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 256:
        raise ValueError("max_new_tokens must be between 1 and 256.")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative.")

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("prompt must encode to at least one token.")
    prompt_ids = prompt_ids[-config.block_size:]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    def synchronize():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        with torch.inference_mode():
            model.generate(input_ids, max_new_tokens=max_new_tokens,
                           temperature=temperature)
        synchronize()

    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    durations = []
    generated_counts = []
    for _ in range(runs):
        synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                temperature=temperature,
                token_byte_pieces=getattr(tokenizer, "token_byte_pieces", None)
            )
        synchronize()
        durations.append(time.perf_counter() - started)
        generated_counts.append(int(output.shape[1] - input_ids.shape[1]))

    total_generated = sum(generated_counts)
    total_seconds = sum(durations)
    result = {
        "schema": "dream-inference-benchmark-v1",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "runs": runs,
        "warmup_runs": warmup,
        "prompt_tokens": len(prompt_ids),
        "requested_new_tokens_per_run": max_new_tokens,
        "generated_tokens": total_generated,
        "mean_latency_seconds": statistics.mean(durations),
        "median_latency_seconds": statistics.median(durations),
        "p95_latency_seconds": percentile(durations, 0.95),
        "mean_generated_tokens_per_second": (
            total_generated / total_seconds if total_seconds > 0 else 0.0
        ),
        "mean_seconds_per_generated_token": (
            total_seconds / total_generated if total_generated else None
        ),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "context_window": config.block_size,
        "tokenizer": tokenizer.format_version if hasattr(tokenizer, "format_version") else "utf8-byte-v1"
    }
    if str(device).startswith("cuda"):
        result["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark generation speed for a trained native DREAM checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="USER: Explain photosynthesis simply.\nASSISTANT:")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--out", default="eval-results/inference-benchmark.json")
    args = parser.parse_args()

    if args.warmup < 0 or args.runs < 1 or not 1 <= args.tokens <= 256:
        raise SystemExit("warmup must be >= 0, runs >= 1 and tokens between 1 and 256.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, tokenizer, config, _ = load_checkpoint(args.checkpoint, device)
        report = benchmark(
            model, tokenizer, config, args.prompt, device,
            args.warmup, args.runs, args.tokens, args.temperature
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Could not benchmark DREAM checkpoint: {exc}") from exc
    report["checkpoint_sha256"] = sha256(args.checkpoint)
    report["prompt_sha256"] = hashlib.sha256(args.prompt.encode("utf-8")).hexdigest()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved DREAM inference benchmark: {output}")


if __name__ == "__main__":
    main()
