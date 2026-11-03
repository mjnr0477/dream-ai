"""Evaluate a DREAM checkpoint on a held-out UTF-8 text corpus."""
import argparse
import hashlib
import json
import math
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="Held-out validation/test text")
    parser.add_argument("--manifest", default="", help="Optional corpus manifest to verify the exact held-out test split")
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
    tokenizer_name = checkpoint.get("tokenizer", "utf8-byte-v1")
    if tokenizer_name == "utf8-byte-v1":
        tokenizer = DreamByteTokenizer()
    elif tokenizer_name == DreamBPETokenizer.format_version:
        try:
            tokenizer = DreamBPETokenizer.from_dict(checkpoint["tokenizer_data"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"Invalid checkpoint BPE tokenizer: {exc}") from exc
    else:
        raise SystemExit("Unsupported checkpoint tokenizer.")
    config = DreamConfig(**checkpoint["config"])
    if config.vocab_size != tokenizer.vocab_size:
        raise SystemExit("Checkpoint model vocabulary does not match its tokenizer.")
    text = data_path.read_text(encoding="utf-8")
    evaluation_hash = sha256(data_path)
    training_meta = checkpoint.get("training") or checkpoint.get("sft_training", {})
    training_hash = training_meta.get("corpus_sha256", training_meta.get("train_sha256"))
    validation_hash = training_meta.get("validation_sha256")
    manifest_hash = None
    manifest_test_groups = set()
    if args.manifest:
        manifest_path = Path(args.manifest)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Could not read evaluation corpus manifest: {exc}") from exc
        if not isinstance(manifest, dict) or manifest.get("schema") != "dream-corpus-v1":
            raise SystemExit("Unsupported evaluation corpus manifest schema.")
        splits = manifest.get("splits")
        if not isinstance(splits, dict):
            raise SystemExit("Evaluation corpus manifest is missing split metadata.")
        test_split = splits.get("test")
        if not isinstance(test_split, dict):
            raise SystemExit("Evaluation corpus manifest is missing the held-out test split.")
        if test_split.get("sha256") != evaluation_hash:
            raise SystemExit("Evaluation corpus hash does not match the manifest held-out test split.")
        manifest_test_groups = test_split.get("source_group_ids")
        if (
            not isinstance(manifest_test_groups, list)
            or not manifest_test_groups
            or any(not isinstance(group, str) or not group.strip() for group in manifest_test_groups)
            or len(manifest_test_groups) != len(set(manifest_test_groups))
        ):
            raise SystemExit("Evaluation manifest has invalid held-out test source-group metadata.")
        for split_name in ("train", "validation"):
            split = splits.get(split_name)
            if not isinstance(split, dict):
                raise SystemExit(f"Evaluation manifest is missing the {split_name} split.")
            groups = split.get("source_group_ids")
            if (
                not isinstance(groups, list)
                or any(not isinstance(group, str) or not group.strip() for group in groups)
                or len(groups) != len(set(groups))
            ):
                raise SystemExit(f"Evaluation manifest has invalid {split_name} source-group metadata.")
            if set(groups) & set(manifest_test_groups):
                raise SystemExit(f"Evaluation manifest has source-group leakage between {split_name} and test.")
        manifest_hash = sha256(manifest_path)
    expected_manifest_hash = training_meta.get("corpus_manifest_sha256")
    sft_manifest_hash = (checkpoint.get("sft_training") or {}).get("pretraining_manifest_sha256")
    if manifest_hash is not None:
        if expected_manifest_hash and expected_manifest_hash != manifest_hash:
            raise SystemExit("Evaluation manifest does not match the pretraining checkpoint manifest.")
        if sft_manifest_hash and sft_manifest_hash != manifest_hash:
            raise SystemExit("Evaluation manifest does not match the SFT checkpoint pretraining manifest.")
    elif expected_manifest_hash or sft_manifest_hash:
        raise SystemExit("This checkpoint requires --manifest for provenance-verified evaluation.")
    if evaluation_hash in {training_hash, validation_hash}:
        raise SystemExit(
            "Evaluation corpus hash matches training or validation data; "
            "use a genuinely held-out test corpus."
        )
    sft_metadata = checkpoint.get("sft_training")
    if sft_metadata:
        expected_test_hash = sft_metadata.get("pretraining_test_sha256")
        if not expected_test_hash or evaluation_hash != expected_test_hash:
            raise SystemExit(
                "SFT checkpoint evaluation must use the exact held-out test corpus "
                "recorded in its pretraining manifest."
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
    losses = []
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
        "evaluation_corpus_sha256": evaluation_hash,
        "corpus_manifest_sha256": manifest_hash,
        "held_out_test_source_groups": len(manifest_test_groups),
        "evaluation_tokens": len(tokens),
        "tokenizer": tokenizer_name,
        "architecture": config.__dict__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": device,
        "seed": args.seed,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "mean_cross_entropy_nats_per_token": mean_loss,
        "bits_per_token": mean_loss / math.log(2),
        "perplexity_per_token": math.exp(min(mean_loss, 20)),
    }
    # Preserve the original byte-metric keys for existing byte-tokenizer consumers.
    if tokenizer_name == "utf8-byte-v1":
        result["evaluation_bytes"] = len(text.encode("utf-8"))
        result["mean_cross_entropy_nats_per_byte"] = mean_loss
        result["bits_per_byte"] = mean_loss / math.log(2)
        result["perplexity_per_byte"] = math.exp(min(mean_loss, 20))
    result.update({
        "sample_prompt": args.sample_prompt,
        "sample_generation": sample,
        "limitations": [
            "Next-token metrics do not measure instruction following or factual accuracy.",
            "A low validation loss does not certify safety, reasoning, usefulness or generalization.",
            "Sample generation is qualitative and must be reviewed by humans."
        ]
    })

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
