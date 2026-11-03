"""Generate reproducible DREAM-native answers for a human-scored task benchmark.

This runner uses only the checkpoint's native tokenizer and transformer. It does
not call external models or automatically grade the model's answers.
"""
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


def load_tasks(path, limit=0):
    tasks = []
    seen_ids = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid benchmark JSON at line {line_number}: {exc}") from exc
        if not isinstance(item, dict):
            raise SystemExit(f"Benchmark item at line {line_number} must be an object.")
        for field in ("id", "category", "prompt"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise SystemExit(f"Benchmark item at line {line_number} is missing {field}.")
        rubric = item.get("rubric")
        if (
            not isinstance(rubric, list)
            or not rubric
            or any(not isinstance(point, str) or not point.strip() for point in rubric)
        ):
            raise SystemExit(f"Benchmark item at line {line_number} needs a non-empty rubric.")
        if item["id"] in seen_ids:
            raise SystemExit(f"Duplicate benchmark task id: {item['id']}")
        seen_ids.add(item["id"])
        tasks.append({
            "id": item["id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "rubric": rubric
        })
    if not tasks:
        raise SystemExit("Benchmark contains no tasks.")
    return tasks[:limit] if limit else tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark", default="evals/core-v1.jsonl")
    parser.add_argument("--out", default="eval-results/native-task-benchmark.json")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Use 0 for deterministic greedy generation.")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional task limit for quick smoke tests; 0 runs all tasks.")
    args = parser.parse_args()

    if type(args.max_new_tokens) is not int or not 0 <= args.max_new_tokens <= 2048:
        raise SystemExit("max-new-tokens must be between 0 and 2048.")
    if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2:
        raise SystemExit("temperature must be finite and between 0 and 2.")
    if args.top_k < 0 or args.limit < 0:
        raise SystemExit("top-k and limit must be non-negative.")

    checkpoint_path = Path(args.checkpoint)
    benchmark_path = Path(args.benchmark)
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

    try:
        config = DreamConfig(**checkpoint["config"])
        training_meta = checkpoint.get("training") or checkpoint.get("sft_training") or {}
        completed_steps = training_meta.get("completed_steps", 0)
        if type(completed_steps) is not int or completed_steps < 1:
            raise ValueError("checkpoint has no completed DREAM training steps")
        if config.vocab_size != tokenizer.vocab_size:
            raise ValueError("checkpoint model vocabulary does not match its tokenizer")
        model = DreamTransformer(config)
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Invalid DREAM checkpoint: {exc}") from exc

    tasks = load_tasks(benchmark_path, args.limit)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    results = []
    with torch.inference_mode():
        for task in tasks:
            prompt_text = f"USER: {task['prompt'].strip()}\nASSISTANT:"
            prompt_ids = tokenizer.encode(prompt_text)
            prompt_ids = prompt_ids[-config.block_size:]
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            generated = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k
            )[0].tolist()
            response = tokenizer.decode(generated[len(prompt_ids):]).strip()
            results.append({
                **task,
                "response": response,
                "response_tokens": len(generated) - len(prompt_ids),
                "human_score": None,
                "review_notes": ""
            })

    report = {
        "schema": "dream-native-task-benchmark-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "benchmark_sha256": sha256(benchmark_path),
        "checkpoint": str(checkpoint_path),
        "tokenizer": tokenizer_name,
        "architecture": config.__dict__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "completed_training_steps": completed_steps,
        "data_provenance_verified": bool(
            training_meta.get("data_provenance_verified")
            or training_meta.get("pretraining_manifest_sha256")
        ),
        "corpus_manifest_sha256": (
            training_meta.get("corpus_manifest_sha256")
            or training_meta.get("pretraining_manifest_sha256")
        ),
        "training_corpus_sha256": (
            training_meta.get("corpus_sha256")
            or training_meta.get("train_sha256")
        ),
        "validation_corpus_sha256": training_meta.get("validation_sha256"),
        "device": device,
        "seed": args.seed,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k
        },
        "task_count": len(results),
        "scoring_policy": "Human rubric scoring only; no model-based automatic grading.",
        "results": results,
        "limitations": [
            "This report records generated answers; it does not establish answer correctness.",
            "A human reviewer must score each response against its rubric.",
            "Smoke-test checkpoints trained on synthetic data are not evidence of production capability."
        ]
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({
        "report": str(output),
        "task_count": len(results),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "benchmark_sha256": report["benchmark_sha256"],
        "device": device
    }, indent=2))


if __name__ == "__main__":
    main()
