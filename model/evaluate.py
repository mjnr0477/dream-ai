"""Evaluate a DREAM checkpoint on a held-out UTF-8 text corpus."""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F

from checkpoint_tokenizer import load_tokenizer
from dream_transformer import DreamConfig, DreamTransformer


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def validate_language_subset(manifest_path, language, data_path, parent_test_path):
    """Verify a language file is an intact, source-isolated subset of the held-out test split."""
    manifest_path = Path(manifest_path)
    data_path = Path(data_path)
    parent_test_path = Path(parent_test_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        test_split = manifest["splits"]["test"]
        language_meta = test_split["language_files"][language]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid corpus manifest or missing test language {language!r}") from exc

    parent_name = test_split.get("path")
    language_name = language_meta.get("path")
    if not isinstance(parent_name, str) or Path(parent_name).name != parent_name:
        raise ValueError("Manifest contains an unsafe parent test path")
    if not isinstance(language_name, str) or Path(language_name).name != language_name:
        raise ValueError("Manifest contains an unsafe language test path")
    if parent_test_path.name != parent_name or data_path.name != language_name:
        raise ValueError("Evaluation files do not match the paths recorded in the manifest")

    parent_hash = sha256(parent_test_path)
    language_hash = sha256(data_path)
    if parent_hash != test_split.get("sha256"):
        raise ValueError("Parent test corpus hash does not match the corpus manifest")
    if language_hash != language_meta.get("sha256"):
        raise ValueError("Language test corpus hash does not match the corpus manifest")

    parent_groups = test_split.get("source_group_ids")
    language_groups = language_meta.get("source_group_ids")
    if (not isinstance(parent_groups, list) or not parent_groups
            or not isinstance(language_groups, list) or not language_groups
            or not all(isinstance(group, str) and group for group in parent_groups + language_groups)
            or len(set(parent_groups)) != len(parent_groups)
            or len(set(language_groups)) != len(language_groups)
            or not set(language_groups).issubset(set(parent_groups))):
        raise ValueError("Language test source groups are not a valid subset of parent test groups")
    return parent_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="Held-out validation/test text")
    parser.add_argument("--corpus-manifest", help="Manifest authorizing a per-language test subset")
    parser.add_argument("--language", help="Exact language tag for a manifest-verified test subset")
    parser.add_argument("--parent-test-data", help="Combined held-out test file containing the language subset")
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
    config = DreamConfig(**checkpoint["config"])
    try:
        tokenizer = load_tokenizer(checkpoint, config)
    except ValueError as exc:
        raise SystemExit(f"Unsupported checkpoint tokenizer: {exc}") from exc
    text = data_path.read_text(encoding="utf-8")
    evaluation_hash = sha256(data_path)
    subset_args = (args.corpus_manifest, args.language, args.parent_test_data)
    if any(subset_args) and not all(subset_args):
        raise SystemExit("corpus-manifest, language and parent-test-data must be supplied together.")
    parent_test_hash = None
    if args.language:
        try:
            parent_test_hash = validate_language_subset(
                args.corpus_manifest, args.language, data_path, args.parent_test_data
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    training_meta = checkpoint.get("training") or checkpoint.get("sft_training", {})
    training_hash = training_meta.get("corpus_sha256", training_meta.get("train_sha256"))
    validation_hash = training_meta.get("validation_sha256")
    if evaluation_hash in {training_hash, validation_hash}:
        raise SystemExit(
            "Evaluation corpus hash matches training or validation data; "
            "use a genuinely held-out test corpus."
        )
    sft_metadata = checkpoint.get("sft_training")
    if sft_metadata:
        expected_test_hash = sft_metadata.get("pretraining_test_sha256")
        authorized_test_hash = parent_test_hash if args.language else evaluation_hash
        if not expected_test_hash or authorized_test_hash != expected_test_hash:
            raise SystemExit(
                "SFT checkpoint evaluation must use the exact held-out test corpus "
                "or a manifest-verified language subset of that corpus."
            )
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if len(tokens) < config.block_size + 2:
        raise SystemExit("Evaluation corpus is too small for the checkpoint context length.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DreamTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    tokens = tokens.to(device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    total_nats = 0.0
    total_target_tokens = 0
    total_target_bytes = 0
    max_start = len(tokens) - config.block_size - 1

    with torch.inference_mode():
        for _ in range(args.batches):
            starts = torch.randint(
                max_start + 1, (args.batch_size,), generator=generator, device="cpu"
            ).tolist()
            x = torch.stack([
                tokens[start:start + config.block_size] for start in starts
            ])
            y = torch.stack([
                tokens[start + 1:start + config.block_size + 1] for start in starts
            ])
            logits, _ = model(x)
            token_losses = F.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                y.reshape(-1),
                reduction="none"
            ).view_as(y)
            byte_lengths = torch.tensor(
                [tokenizer.token_byte_length(int(token_id))
                 for token_id in y.reshape(-1).tolist()],
                dtype=token_losses.dtype,
                device=device
            ).view_as(y)
            total_nats += float(token_losses.sum().item())
            total_target_tokens += y.numel()
            total_target_bytes += int(byte_lengths.sum().item())

        prompt_ids = tokenizer.encode(args.sample_prompt)
        if not prompt_ids:
            prompt_ids = [32]
        prompt = torch.tensor(
            [prompt_ids[-config.block_size:]], dtype=torch.long, device=device
        )
        torch.manual_seed(args.seed)
        generated = model.generate(
            prompt, max_new_tokens=args.sample_tokens, temperature=0.8, top_k=40,
            constrain_utf8=True,
            token_byte_pieces=getattr(tokenizer, "token_byte_pieces", None)
        )[0].tolist()
        sample = tokenizer.decode(generated[len(prompt[0]):])

    mean_nats_per_token = total_nats / total_target_tokens
    mean_nats_per_byte = total_nats / total_target_bytes
    result = {
        "schema": "dream-evaluation-v1",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "evaluation_corpus": str(data_path),
        "evaluation_corpus_sha256": evaluation_hash,
        "evaluation_language": args.language,
        "evaluation_parent_test_sha256": parent_test_hash,
        "evaluation_bytes": len(text.encode("utf-8")),
        "evaluation_tokens": len(tokens),
        "sampled_target_tokens": total_target_tokens,
        "sampled_target_bytes": total_target_bytes,
        "architecture": config.__dict__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": device,
        "seed": args.seed,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "mean_cross_entropy_nats_per_token": mean_nats_per_token,
        "mean_cross_entropy_nats_per_byte": mean_nats_per_byte,
        "bits_per_byte": mean_nats_per_byte / math.log(2),
        "perplexity_per_byte": math.exp(min(mean_nats_per_byte, 20)),
        "sample_prompt": args.sample_prompt,
        "sample_generation": sample,
        "limitations": [
            "Token cross-entropy is divided by the exact UTF-8 bytes represented by sampled target tokens; it is not a language-quality score.",
            "Metrics use randomly sampled context windows and can vary with seed and batch count.",
            "Metrics do not measure instruction following or factual accuracy.",
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
