"""Evaluate DREAM BPE token efficiency per language on held-out text.

The BPE vocabulary is trained only on --train-data. --test-data is used only
for measurement and exact UTF-8 round-trip checks; this is not a language
quality or model-intelligence benchmark.
"""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from dream_bpe import DreamBPETokenizer


LANGUAGE_TAG = re.compile(r"^[A-Za-z]{2,3}(?:[-+][A-Za-z0-9]{2,8})*$")


def load_records(path):
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Record at {path}:{line_number} must be an object")
        language = row.get("language")
        text = row.get("text")
        if not isinstance(language, str) or not LANGUAGE_TAG.fullmatch(language):
            raise ValueError(f"Invalid language tag at {path}:{line_number}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Missing non-empty text at {path}:{line_number}")
        records.append({"language": language, "text": text})
    if not records:
        raise ValueError(f"No language-tagged text records in {path}")
    return records


def _fingerprint(text):
    normalized = " ".join(text.split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def benchmark_multilingual(train_records, test_records, vocab_size=1024,
                           min_frequency=2):
    if not isinstance(train_records, list) or not train_records:
        raise ValueError("Training records must be a non-empty list")
    if not isinstance(test_records, list) or not test_records:
        raise ValueError("Test records must be a non-empty list")

    def validate(records, label):
        clean = []
        for index, row in enumerate(records, 1):
            if not isinstance(row, dict):
                raise ValueError(f"{label} row {index} must be an object")
            language, text = row.get("language"), row.get("text")
            if not isinstance(language, str) or not LANGUAGE_TAG.fullmatch(language):
                raise ValueError(f"{label} row {index} has an invalid language tag")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{label} row {index} has empty text")
            clean.append({"language": language, "text": text})
        return clean

    train = validate(train_records, "Training")
    tests = validate(test_records, "Test")
    train_hashes = {_fingerprint(row["text"]) for row in train}
    if any(_fingerprint(row["text"]) in train_hashes for row in tests):
        raise ValueError("Training and test text overlap after normalization")

    tokenizer = DreamBPETokenizer.train(
        "\n".join(row["text"] for row in train),
        vocab_size=vocab_size,
        min_frequency=min_frequency
    )
    grouped = defaultdict(list)
    for row in tests:
        grouped[row["language"]].append(row["text"])

    per_language = {}
    total_bytes = total_bpe = total_documents = 0
    for language in sorted(grouped):
        samples = grouped[language]
        original_bytes = sum(len(sample.encode("utf-8")) for sample in samples)
        encoded = [tokenizer.encode(sample) for sample in samples]
        if any(tokenizer.decode(ids) != sample for ids, sample in zip(encoded, samples)):
            raise RuntimeError(f"Lossless round-trip failed for language {language}")
        bpe_tokens = sum(map(len, encoded))
        reduction = (original_bytes - bpe_tokens) / original_bytes if original_bytes else 0.0
        per_language[language] = {
            "documents": len(samples),
            "utf8_bytes": original_bytes,
            "byte_token_count": original_bytes,
            "bpe_token_count": bpe_tokens,
            "tokens_saved": original_bytes - bpe_tokens,
            "token_reduction_fraction": round(reduction, 6),
            "round_trip_exact": True
        }
        total_bytes += original_bytes
        total_bpe += bpe_tokens
        total_documents += len(samples)

    return {
        "schema": "dream-multilingual-tokenizer-benchmark-v1",
        "purpose": "held-out tokenizer efficiency and Unicode round-trip only; not language fluency",
        "tokenizer_format": tokenizer.format_version,
        "vocabulary_size": tokenizer.vocab_size,
        "training_documents": len(train),
        "training_languages": sorted({row["language"] for row in train}),
        "test_documents": total_documents,
        "test_languages": sorted(grouped),
        "held_out_text": True,
        "overall": {
            "utf8_bytes": total_bytes,
            "byte_token_count": total_bytes,
            "bpe_token_count": total_bpe,
            "tokens_saved": total_bytes - total_bpe,
            "token_reduction_fraction": round(
                (total_bytes - total_bpe) / total_bytes, 6
            ) if total_bytes else 0.0,
            "round_trip_exact": True
        },
        "by_language": per_language
    }, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True, help="Training JSONL with language,text")
    parser.add_argument("--test-data", required=True, help="Held-out JSONL with language,text")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--out", default="reports/multilingual-tokenizer-benchmark.json")
    parser.add_argument("--tokenizer-out", default="checkpoints/dream-bpe-multilingual.json")
    args = parser.parse_args()

    report, tokenizer = benchmark_multilingual(
        load_records(args.train_data),
        load_records(args.test_data),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency
    )
    report_path = Path(args.out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(report_path)
    tokenizer.save(args.tokenizer_out)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved tokenizer: {args.tokenizer_out}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
