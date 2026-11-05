"""Fail-closed validation and loading for native DREAM inference checkpoints."""
from pathlib import Path

import torch

from checkpoint_tokenizer import load_tokenizer
from dream_transformer import DreamConfig, DreamTransformer


def load_validated_checkpoint(checkpoint_path):
    """Load a trained DREAM checkpoint only after validating its full model contract.

    Uses PyTorch's restricted weights-only loader, exact architecture metadata,
    tokenizer/config compatibility, positive training-step metadata, and a
    complete finite state-dict check before returning a model.
    """
    path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Checkpoint cannot be safely loaded: {exc}") from exc

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint root must be an object")

    config_data = checkpoint.get("config")
    if not isinstance(config_data, dict):
        raise ValueError("Checkpoint config must be an object")
    expected_config_fields = set(DreamConfig.__dataclass_fields__)
    if set(config_data) != expected_config_fields:
        raise ValueError("Checkpoint config contains unexpected or missing fields")
    try:
        config = DreamConfig(**config_data)
    except TypeError as exc:
        # Keep schema failures explicit even if the dataclass rejects a field
        # before the exact-field guard can classify it.
        raise ValueError(
            f"Checkpoint config contains unexpected or missing fields: {exc}"
        ) from exc
    except ValueError as exc:
        raise ValueError(f"Invalid DREAM model config: {exc}") from exc

    try:
        tokenizer = load_tokenizer(checkpoint, config)
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"Invalid DREAM checkpoint tokenizer: {exc}") from exc

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
        raise ValueError("Refusing to serve a checkpoint without completed DREAM training.")

    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint model_state must be an object")

    model = DreamTransformer(config)
    expected = model.state_dict()
    if any(not isinstance(name, str) for name in state):
        raise ValueError("Model state keys must all be strings")
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"Model state keys do not match architecture (missing={missing}, "
            f"unexpected={unexpected})"
        )

    for name, expected_tensor in expected.items():
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Model state entry {name} is not a tensor")
        if tensor.shape != expected_tensor.shape:
            raise ValueError(f"Model state tensor {name} has an incorrect shape")
        if tensor.dtype != expected_tensor.dtype:
            raise ValueError(f"Model state tensor {name} has an incorrect dtype")
        if tensor.layout != expected_tensor.layout:
            raise ValueError(f"Model state tensor {name} has an incorrect layout")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise ValueError(f"Model state tensor {name} contains NaN or infinity")

    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"Model state cannot be loaded strictly: {exc}") from exc

    return model, tokenizer, config, metadata
