"""Measure DREAM BPE efficiency on text held out from tokenizer training.

The tokenizer is learned only from --data. All reported efficiency metrics are
measured on --validation-data so repeated training text cannot inflate the
reported result.
"""
import argparse
import hashlib
import json
from pathlib import Path

from dream_bpe import DreamBPETokenizer


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def benchmark(training_text, evaluation_text, vocab_size=1024, min_frequency=2):
    if not isinstance(training_text, str) or not training_text.strip():
        raise ValueError("Tokenizer training text must be non-empty")
    if not isinstance(evaluation_text, str) or not evaluation_text.strip():
        raise ValueError("Held-out evaluation text must be non-empty")
    if training_text == evaluation_text:
        raise ValueError("Tokenizer training and held-out evaluation text must differ")

    tokenizer = DreamBPETokenizer.train(
        training_text, vocab_size=vocab_size, min_frequency=min_frequency
    )
    evaluation_bytes = evaluation_text.encode("utf-8")
    bpe_ids = tokenizer.encode(evaluation_text)
    decoded = tokenizer.decode(bpe_ids)
    if decoded != evaluation_text:
        raise RuntimeError("BPE tokenizer failed lossless held-out round-trip validation")

    byte_count = len(evaluation_bytes)
    bpe_count = len(bpe_ids)
    tokenizer_json = json.dumps(
        tokenizer.to_dict(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    report = {
        "schema": "dream-tokenizer-benchmark-v2",
        "evaluation_scope": "held-out",
        "training_corpus_sha256": sha256_bytes(training_text.encode("utf-8")),
        "evaluation_corpus_sha256": sha256_bytes(evaluation_bytes),
        "training_utf8_bytes": len(training_text.encode("utf-8")),
        "corpus_utf8_bytes": byte_count,
        "byte_token_count": byte_count,
        "bpe_token_count": bpe_count,
        "vocabulary_size": tokenizer.vocab_size,
        "requested_vocabulary_size": vocab_size,
        "tokens_saved": byte_count - bpe_count,
        "token_reduction_fraction": round(
            (byte_count - bpe_count) / byte_count, 6
        ),
        "bytes_per_bpe_token": round(byte_count / bpe_count, 6),
        "round_trip_exact": True,
        "tokenizer_format": tokenizer.format_version,
        "tokenizer_sha256": sha256_bytes(tokenizer_json)
    }
    return report, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Train DREAM BPE on one corpus and evaluate it on held-out text."
    )
    parser.add_argument("--data", required=True, help="UTF-8 tokenizer training corpus")
    parser.add_argument(
        "--validation-data", required=True,
        help="Separate held-out UTF-8 corpus; never used to train tokenizer merges"
    )
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--out", default="reports/tokenizer-benchmark.json")
    parser.add_argument("--tokenizer-out", default="checkpoints/dream-bpe.json")
    args = parser.parse_args()

    training_text = Path(args.data).read_text(encoding="utf-8")
    evaluation_text = Path(args.validation_data).read_text(encoding="utf-8")
    report, tokenizer = benchmark(
        training_text, evaluation_text,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency
    )

    report_path = Path(args.out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    temporary.replace(report_path)
    tokenizer.save(args.tokenizer_out)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved tokenizer: {args.tokenizer_out}")
    print(f"Saved benchmark: {report_path}")


if __name__ == "__main__":
    main()
