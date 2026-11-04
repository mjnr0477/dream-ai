"""Estimate DREAM model parameter and baseline memory budgets.

These are analytical lower bounds, not total GPU-memory predictions. They exclude
activations, CUDA workspaces, allocator fragmentation, dataloader buffers and
framework overhead.
"""
import argparse
import json
from pathlib import Path

from dream_transformer import DreamConfig


def estimate_model(config: DreamConfig) -> dict:
    """Return exact parameter count and transparent parameter-memory estimates.

    The current transformer ties the output projection to token embeddings.
    Each block has QKV + attention projection (4 * d_model**2), a 4x
    feed-forward pair (8 * d_model**2), and two LayerNorms (4 * d_model).
    """
    d = config.d_model
    v = config.vocab_size
    b = config.block_size
    layers = config.n_layers

    parameters = (
        v * d
        + b * d
        + layers * (12 * d * d + 9 * d)
        + 2 * d
    )
    return {
        "schema": "dream-model-budget-v1",
        "architecture": "dream-decoder-only-transformer-v1",
        "config": {
            "vocab_size": v,
            "block_size": b,
            "d_model": d,
            "n_heads": config.n_heads,
            "n_layers": layers,
            "dropout": config.dropout,
        },
        "parameter_count": parameters,
        "parameter_memory": {
            "fp32_inference_bytes": parameters * 4,
            "fp16_inference_bytes": parameters * 2,
            "fp32_adam_training_floor_bytes": parameters * 16,
        },
        "training_floor_assumptions": (
            "4 bytes FP32 weights + 4 bytes gradients + 8 bytes for two "
            "FP32 Adam moments per parameter"
        ),
        "excluded_memory": [
            "activations and saved tensors for backpropagation",
            "CUDA kernels and temporary workspaces",
            "optimizer implementation overhead",
            "allocator fragmentation",
            "dataloader and host-side buffers",
        ],
        "warning": (
            "Analytical parameter-memory lower bounds only; do not use these "
            "values as a complete GPU VRAM requirement."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    try:
        config = DreamConfig(
            vocab_size=args.vocab_size,
            block_size=args.block_size,
            d_model=args.d_model,
            n_heads=args.heads,
            n_layers=args.layers,
            dropout=args.dropout,
        )
        report = estimate_model(config)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

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
