"""Measure DREAM BPE token efficiency against the legacy UTF-8 byte tokenizer."""
import argparse
import json
from pathlib import Path

from dream_bpe import DreamBPETokenizer


def benchmark(text, vocab_size=1024, min_frequency=2):
    if not isinstance(text, str) or not text:
        raise ValueError("Benchmark text must be a non-empty string")
    tokenizer = DreamBPETokenizer.train(
        text, vocab_size=vocab_size, min_frequency=min_frequency
    )
    byte_count = len(text.encode("utf-8"))
    bpe_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(bpe_ids)
    if decoded != text:
        raise RuntimeError("BPE tokenizer failed lossless round-trip validation")
    bpe_count = len(bpe_ids)
    return {
        "schema": "dream-tokenizer-benchmark-v1",
        "corpus_utf8_bytes": byte_count,
        "byte_token_count": byte_count,
        "bpe_token_count": bpe_count,
        "vocabulary_size": tokenizer.vocab_size,
        "requested_vocabulary_size": vocab_size,
        "tokens_saved": byte_count - bpe_count,
        "token_reduction_fraction": round(
            (byte_count - bpe_count) / byte_count, 6
        ),
        "round_trip_exact": True,
        "tokenizer_format": tokenizer.format_version
    }, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="UTF-8 plain-text corpus")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--out", default="reports/tokenizer-benchmark.json")
    parser.add_argument("--tokenizer-out", default="checkpoints/dream-bpe.json")
    args = parser.parse_args()

    text = Path(args.data).read_text(encoding="utf-8")
    report, tokenizer = benchmark(text, args.vocab_size, args.min_frequency)
    report_path = Path(args.out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    tokenizer.save(args.tokenizer_out)
    print(json.dumps(report, indent=2))
    print(f"Saved tokenizer: {args.tokenizer_out}")
    print(f"Saved benchmark: {report_path}")


if __name__ == "__main__":
    main()
