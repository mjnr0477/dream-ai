"""Compare cached autoregressive generation with exact full-prefix recomputation.

The benchmark uses greedy decoding so both paths produce exactly the same bytes.
It measures local latency only; results depend on hardware, checkpoint and context.
"""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from dream_transformer import (
    DreamByteTokenizer,
    DreamConfig,
    DreamTransformer,
    parameter_count,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_without_cache(model, prompt_ids, max_new_tokens):
    """Reference greedy decoder that recomputes the complete prefix each step."""
    generated = prompt_ids.clone()
    batch_size = generated.shape[0]
    device = generated.device
    utf8_state = (
        torch.zeros(batch_size, dtype=torch.long, device=device),
        torch.full((batch_size,), 0x80, dtype=torch.long, device=device),
        torch.full((batch_size,), 0xBF, dtype=torch.long, device=device),
    )
    with torch.inference_mode():
        for step in range(max_new_tokens):
            context = generated[:, -model.config.block_size:]
            logits, _ = model(context)
            next_token, utf8_state = model._sample_next_byte(
                logits[:, -1, :], 0, 0, utf8_state,
                max_new_tokens - step - 1,
            )
            generated = torch.cat((generated, next_token), dim=1)
    return generated


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(function, runs, warmup, device):
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(runs):
        synchronize(device)
        started = time.perf_counter()
        function()
        synchronize(device)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), samples


def benchmark(model, prompt_ids, max_new_tokens, runs=5, warmup=1):
    if not isinstance(runs, int) or isinstance(runs, bool) or runs < 1:
        raise ValueError("runs must be a positive integer")
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if prompt_ids.ndim != 2 or prompt_ids.shape[1] < 1 or prompt_ids.dtype != torch.long:
        raise ValueError("prompt_ids must be a non-empty torch.long [batch, sequence] tensor")
    if prompt_ids.shape[1] + max_new_tokens > model.config.block_size:
        raise ValueError("prompt plus generated tokens must fit inside block_size for a cache benchmark")

    model.eval()
    with torch.inference_mode():
        cached_output = model.generate(
            prompt_ids.clone(), max_new_tokens=max_new_tokens,
            temperature=0, top_k=0,
        )
        reference_output = generate_without_cache(model, prompt_ids, max_new_tokens)
    if not torch.equal(cached_output, reference_output):
        raise RuntimeError("Cached and full-prefix generation produced different token IDs")

    device = prompt_ids.device
    cached_median, cached_samples = measure(
        lambda: model.generate(
            prompt_ids.clone(), max_new_tokens=max_new_tokens,
            temperature=0, top_k=0,
        ),
        runs, warmup, device,
    )
    reference_median, reference_samples = measure(
        lambda: generate_without_cache(model, prompt_ids, max_new_tokens),
        runs, warmup, device,
    )
    return {
        "equivalent_outputs": True,
        "runs": runs,
        "warmup_runs": warmup,
        "prompt_tokens": prompt_ids.shape[1],
        "generated_tokens": max_new_tokens,
        "cached": {
            "median_seconds": cached_median,
            "tokens_per_second": max_new_tokens / cached_median,
            "samples_seconds": cached_samples,
        },
        "full_prefix_reference": {
            "median_seconds": reference_median,
            "tokens_per_second": max_new_tokens / reference_median,
            "samples_seconds": reference_samples,
        },
        "speedup_ratio": reference_median / cached_median,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="USER: Hello DREAM.\nASSISTANT:")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("tokenizer") != "utf8-byte-v1":
        raise SystemExit("Unsupported DREAM checkpoint tokenizer.")
    training = checkpoint.get("sft_training") or checkpoint.get("training") or {}
    completed_steps = training.get("completed_steps", 0)
    if not isinstance(completed_steps, int) or isinstance(completed_steps, bool) or completed_steps < 1:
        raise SystemExit("Benchmark requires a checkpoint with completed DREAM training steps.")

    try:
        config = DreamConfig(**checkpoint["config"])
        model = DreamTransformer(config)
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Invalid DREAM checkpoint: {exc}") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    tokenizer = DreamByteTokenizer()
    prompt = tokenizer.encode(args.prompt)
    if not prompt:
        raise SystemExit("Prompt must not be empty.")
    prompt_ids = torch.tensor([prompt], dtype=torch.long, device=device)

    try:
        results = benchmark(model, prompt_ids, args.tokens, args.runs, args.warmup)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    report = {
        "schema": "dream-generation-cache-benchmark-v1",
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint": str(checkpoint_path),
        "completed_training_steps": completed_steps,
        "device": str(device),
        "parameters": parameter_count(model),
        "config": config.__dict__,
        **results,
        "interpretation": (
            "Local greedy-decoding latency comparison. A speedup ratio above 1 means "
            "cached generation was faster in this run. This is not a model-quality metric."
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
