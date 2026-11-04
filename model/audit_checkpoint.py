"""Audit a DREAM checkpoint's structure and tensor integrity without modifying it."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from dream_transformer import DreamConfig, DreamTransformer


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_checkpoint(path):
    checkpoint_path = Path(path)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise ValueError(f"Checkpoint cannot be safely loaded: {exc}") from exc

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint root must be an object")
    if checkpoint.get("tokenizer") != "utf8-byte-v1":
        raise ValueError("Unsupported or missing DREAM tokenizer")

    config_data = checkpoint.get("config")
    if not isinstance(config_data, dict):
        raise ValueError("Checkpoint config must be an object")
    try:
        config = DreamConfig(**config_data)
        if config.__dict__ != config_data:
            raise ValueError("Checkpoint config contains unexpected or missing fields")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid DREAM model config: {exc}") from exc

    metadata_kind = (
        "sft_training"
        if isinstance(checkpoint.get("sft_training"), dict)
        and checkpoint["sft_training"]
        else "training"
    )
    metadata = checkpoint.get(metadata_kind)
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint has no training metadata")
    completed_steps = metadata.get("completed_steps")
    if (
        not isinstance(completed_steps, int)
        or isinstance(completed_steps, bool)
        or completed_steps < 1
    ):
        raise ValueError("Checkpoint must record a positive integer completed_steps")

    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint model_state must be an object")
    model = DreamTransformer(config)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"Model state keys do not match architecture (missing={missing}, "
            f"unexpected={unexpected})"
        )

    tensor_count = 0
    parameter_count = 0
    for name, expected_tensor in expected.items():
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Model state entry {name} is not a tensor")
        if tensor.shape != expected_tensor.shape:
            raise ValueError(f"Model state tensor {name} has an incorrect shape")
        if tensor.dtype != expected_tensor.dtype:
            raise ValueError(f"Model state tensor {name} has an incorrect dtype")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise ValueError(f"Model state tensor {name} contains NaN or infinity")
        tensor_count += 1
        parameter_count += tensor.numel()

    # Strict load catches any remaining state-dict incompatibility.
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"Model state cannot be loaded strictly: {exc}") from exc

    return {
        "schema": "dream-checkpoint-audit-v1",
        "checkpoint_sha256": sha256(checkpoint_path),
        "tokenizer": checkpoint["tokenizer"],
        "config": config.__dict__,
        "completed_steps": completed_steps,
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "all_floating_tensors_finite": True,
        "strict_state_dict_load": True,
        "training_metadata_kind": metadata_kind,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    try:
        report = audit_checkpoint(args.checkpoint)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Checkpoint audit failed: {exc}") from exc

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
