"""Report held-out next-byte metrics separately for each reviewed language.

The evaluator reconstructs the exact prepared test split from its source JSONL and
manifest, then evaluates language-isolated byte streams. It does not score fluency.
"""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from dream_transformer import DreamByteTokenizer, DreamConfig, DreamTransformer


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_language_test_corpora(records_path, manifest_path, test_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema") != "dream-corpus-v1":
        raise ValueError("Unsupported corpus manifest schema.")
    test_meta = manifest.get("splits", {}).get("test", {})
    test_text = Path(test_path).read_bytes()
    actual_hash = sha256_bytes(test_text)
    if actual_hash != test_meta.get("sha256"):
        raise ValueError("Test file SHA-256 does not match the manifest test split.")

    expected_ids = test_meta.get("record_ids")
    if not isinstance(expected_ids, list) or not expected_ids:
        raise ValueError("Manifest test split has no record IDs.")
    expected = set(expected_ids)
    if len(expected) != len(expected_ids) or not all(
        isinstance(item, str) and item for item in expected_ids
    ):
        raise ValueError("Manifest test record IDs are invalid.")

    grouped = defaultdict(list)
    selected = []
    seen_ids = set()
    for line_number, line in enumerate(
        Path(records_path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid source JSONL at line {line_number}.") from exc
        if row.get("id") not in expected:
            continue
        if row["id"] in seen_ids:
            raise ValueError("Duplicate record ID in source JSONL.")
        seen_ids.add(row["id"])
        if any(row.get(flag) is not True for flag in (
            "rights_confirmed", "privacy_reviewed", "quality_reviewed"
        )):
            raise ValueError("Test record is missing required review attestations.")
        language = row.get("language")
        text = row.get("text")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("Test record has an invalid language tag.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Test record has empty text.")
        selected.append(row)
        grouped[language.strip()].append(text.strip())

    if seen_ids != expected:
        raise ValueError("Source JSONL does not contain every manifest test record.")
    reconstructed = ("\n\n".join(row["text"].strip() for row in selected) + "\n").encode("utf-8")
    if reconstructed != test_text:
        raise ValueError("Source JSONL records do not reconstruct the exact held-out test file.")

    manifest_counts = test_meta.get("language_counts", {})
    actual_counts = {}
    for row in selected:
        language = row["language"].strip()
        actual_counts[language] = actual_counts.get(language, 0) + 1
    if actual_counts != manifest_counts:
        raise ValueError("Language counts do not match the manifest test split.")
    return {
        language: "\n\n".join(texts) + "\n"
        for language, texts in sorted(grouped.items())
    }, actual_hash, actual_counts


def evaluate_tokens(model, tokens, block_size, batches, batch_size, seed, device):
    max_start = len(tokens) - block_size - 1
    if max_start < 1:
        raise ValueError("Not enough bytes for one full evaluation context.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses = []
    with torch.inference_mode():
        for _ in range(batches):
            starts = torch.randint(
                max_start + 1, (batch_size,), generator=generator, device="cpu"
            ).tolist()
            x = torch.stack([tokens[i:i + block_size] for i in starts]).to(device)
            y = torch.stack([
                tokens[i + 1:i + block_size + 1] for i in starts
            ]).to(device)
            _, loss = model(x, y)
            losses.append(float(loss.item()))
    mean_loss = sum(losses) / len(losses)
    return {
        "evaluation_windows": batches * batch_size,
        "mean_cross_entropy_nats_per_byte": mean_loss,
        "bits_per_byte": mean_loss / math.log(2),
        "perplexity_per_byte": math.exp(min(mean_loss, 20)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="Exact prepared test.txt")
    parser.add_argument("--records-jsonl", required=True, help="Original reviewed corpus JSONL")
    parser.add_argument("--manifest", required=True, help="prepare_corpus.py manifest.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="eval-results/multilingual-by-language.json")
    args = parser.parse_args()

    if args.batch_size < 1 or args.batches < 1:
        raise SystemExit("batch-size and batches must be positive.")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("tokenizer") != "utf8-byte-v1":
        raise SystemExit("Unsupported checkpoint tokenizer.")
    config = DreamConfig(**checkpoint["config"])
    training_meta = checkpoint.get("training") or checkpoint.get("sft_training", {})
    training_hash = training_meta.get("corpus_sha256", training_meta.get("train_sha256"))
    validation_hash = training_meta.get("validation_sha256")
    test_hash = sha256_file(args.data)
    if test_hash in {training_hash, validation_hash}:
        raise SystemExit("Evaluation corpus matches training or validation data.")
    sft_meta = checkpoint.get("sft_training")
    if sft_meta and sft_meta.get("pretraining_test_sha256") != test_hash:
        raise SystemExit("SFT evaluation must use its exact recorded pretraining test corpus.")

    try:
        corpora, corpus_hash, language_record_counts = load_language_test_corpora(
            args.records_jsonl, args.manifest, args.data
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"Cannot verify language test split: {exc}") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DreamTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    tokenizer = DreamByteTokenizer()
    results = {}
    skipped = {}
    for index, (language, text) in enumerate(sorted(corpora.items())):
        tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long, device="cpu")
        if len(tokens) < config.block_size + 2:
            skipped[language] = {
                "utf8_bytes": len(tokens),
                "reason": "Insufficient bytes for one complete model context."
            }
            continue
        results[language] = {
            "records": language_record_counts[language],
            "utf8_bytes": len(tokens),
            **evaluate_tokens(
                model, tokens, config.block_size, args.batches,
                args.batch_size, args.seed + index, device
            )
        }

    report = {
        "schema": "dream-multilingual-evaluation-v1",
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "test_corpus_sha256": corpus_hash,
        "architecture": config.__dict__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": device,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "batches_per_language": args.batches,
        "language_results": results,
        "skipped_languages": skipped,
        "limitations": [
            "Metrics are byte-level next-token prediction, not translation quality or language fluency.",
            "Languages with too few held-out bytes are reported as skipped, never silently combined.",
            "Each language is evaluated as an isolated byte stream; boundary context is not shared across languages.",
            "Human native-speaker review remains necessary for factuality, instruction following, safety and usefulness."
        ]
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
