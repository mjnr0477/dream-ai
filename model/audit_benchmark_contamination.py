#!/usr/bin/env python3
"""Find exact and normalized text overlap between DREAM evaluation tasks and corpora.

This is a deterministic leakage screen, not a proof that a benchmark is uncontaminated.
It reports hashes and task IDs only; it never copies matched training text into the report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

CONTENT_FIELDS = (
    "text", "input_text", "target_text", "prompt", "response", "content",
    "instruction", "output", "completion", "answer", "body", "question",
)
TASK_REFERENCE_FIELDS = ("reference", "expected_answer", "expected_expression", "expected_output")
MIN_MATCH_CHARS = 40


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_text(value: str) -> str:
    """Match the corpus normalizer: NFKC, case-folding and whitespace folding."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path.name} at line {line_number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"JSONL record in {path.name} at line {line_number} must be an object")
            records.append(item)
    return records


def load_benchmark(path: Path) -> list[dict]:
    tasks = read_jsonl(path)
    if not tasks:
        raise ValueError("Benchmark contains no tasks")
    seen = set()
    for index, task in enumerate(tasks, 1):
        task_id = task.get("id")
        prompt = task.get("prompt")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"Benchmark task {index} has no valid id")
        if task_id in seen:
            raise ValueError(f"Duplicate benchmark task id: {task_id}")
        seen.add(task_id)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Benchmark task {task_id} has no valid prompt")
        rubric = task.get("rubric", [])
        if not isinstance(rubric, list) or any(not isinstance(item, str) for item in rubric):
            raise ValueError(f"Benchmark task {task_id} has an invalid rubric")
    return tasks


def task_phrases(task: dict) -> list[tuple[str, str]]:
    phrases = [("prompt", task["prompt"])]
    for index, item in enumerate(task.get("rubric", [])):
        if item.strip():
            phrases.append((f"rubric[{index}]", item))
    for field in TASK_REFERENCE_FIELDS:
        value = task.get(field)
        if isinstance(value, str) and value.strip():
            phrases.append((field, value))
    return phrases


def corpus_record_text(record: dict) -> tuple[str, list[str]]:
    segments: list[tuple[str, str]] = []
    for field in CONTENT_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            segments.append((field, value.strip()))
    messages = record.get("messages")
    if isinstance(messages, list):
        for index, message in enumerate(messages):
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    segments.append((f"messages[{index}].content", content.strip()))
    return "\n".join(value for _, value in segments), [field for field, _ in segments]


def load_corpus_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".txt":
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            raise ValueError(f"Corpus text file is empty: {path.name}")
        return [{"__content__": raw, "__fields__": ["file"]}]
    if path.suffix.lower() != ".jsonl":
        raise ValueError(f"Corpus must be .jsonl or .txt: {path.name}")
    records = read_jsonl(path)
    if not records:
        raise ValueError(f"Corpus JSONL contains no records: {path.name}")
    normalized = []
    for record in records:
        content, fields = corpus_record_text(record)
        if not content:
            continue
        normalized.append({
            "__content__": content,
            "__fields__": fields,
            "__record_id__": record.get("id", record.get("record_id")),
            "__source_group__": record.get("source_group"),
        })
    if not normalized:
        raise ValueError(f"Corpus has no recognized text fields: {path.name}")
    return normalized


def fingerprint(value: object, domain: str) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return sha256_bytes((domain + "\0" + value).encode("utf-8"))


def audit(benchmark_path: Path, corpus_paths: list[Path]) -> dict:
    benchmark_raw = benchmark_path.read_bytes()
    tasks = load_benchmark(benchmark_path)
    overlaps = []
    checked_fields = 0
    skipped_short_fields = 0
    corpus_summaries = []

    for corpus_path in corpus_paths:
        if not corpus_path.is_file():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
        corpus_raw = corpus_path.read_bytes()
        records = load_corpus_records(corpus_path)
        scanned = 0
        for record in records:
            content = record["__content__"]
            normalized_content = normalize_text(content)
            if not normalized_content:
                continue
            scanned += 1
            record_hash = sha256_bytes(content.encode("utf-8"))
            record_id_hash = fingerprint(record.get("__record_id__"), "dream-corpus-record-id-v1")
            source_group_hash = fingerprint(record.get("__source_group__"), "dream-source-group-v1")
            for task in tasks:
                for field, phrase in task_phrases(task):
                    normalized_phrase = normalize_text(phrase)
                    if len(normalized_phrase) < MIN_MATCH_CHARS:
                        skipped_short_fields += 1
                        continue
                    checked_fields += 1
                    if normalized_phrase in normalized_content:
                        overlaps.append({
                            "task_id": task["id"],
                            "benchmark_field": field,
                            "corpus_file": corpus_path.name,
                            "corpus_record_sha256": record_hash,
                            "corpus_record_id_sha256": record_id_hash,
                            "source_group_sha256": source_group_hash,
                            "match_type": "normalized_substring",
                        })
        corpus_summaries.append({
            "file": corpus_path.name,
            "sha256": sha256_bytes(corpus_raw),
            "records_scanned": scanned,
            "records_without_text_skipped": len(records) - scanned,
        })

    overlaps.sort(key=lambda item: (
        item["task_id"], item["corpus_file"], item["benchmark_field"],
        item["corpus_record_sha256"],
    ))
    return {
        "schema": "dream-benchmark-contamination-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_file": benchmark_path.name,
        "benchmark_sha256": sha256_bytes(benchmark_raw),
        "benchmark_tasks": len(tasks),
        "corpus_files": corpus_summaries,
        "corpus_records_scanned": sum(item["records_scanned"] for item in corpus_summaries),
        "task_fields_checked": checked_fields,
        "short_task_fields_skipped": skipped_short_fields,
        "minimum_match_characters": MIN_MATCH_CHARS,
        "overlap_count": len(overlaps),
        "status": "overlap_detected" if overlaps else "no_exact_normalized_overlap_found",
        "overlaps": overlaps,
        "limitations": [
            "This detects exact normalized prompt, rubric and long reference substrings only.",
            "It does not detect paraphrases, semantic similarity, shared source material or memorization.",
            "A clean report is not proof of an uncontaminated benchmark; human source review remains required.",
            "Reports contain hashes and task IDs, not matched training text or raw record identifiers.",
        ],
    }


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path, help="Benchmark JSONL file")
    parser.add_argument("--corpus", required=True, nargs="+", type=Path,
                        help="One or more training/validation/test JSONL or text files")
    parser.add_argument("--out", required=True, type=Path, help="JSON audit report path")
    args = parser.parse_args()
    try:
        report = audit(args.benchmark, args.corpus)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    write_json_atomic(args.out, report)
    print(json.dumps({
        "status": report["status"],
        "benchmark_tasks": report["benchmark_tasks"],
        "corpus_records_scanned": report["corpus_records_scanned"],
        "task_fields_checked": report["task_fields_checked"],
        "overlap_count": report["overlap_count"],
        "report": str(args.out),
    }, indent=2))
    return 1 if report["overlap_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
