"""Run a deterministic, small task benchmark against a native DREAM checkpoint.

This is a diagnostic exact-match suite, not a comprehensive capability or safety
benchmark. It does not approve a checkpoint for production.
"""
import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from checkpoint_tokenizer import load_tokenizer
from dream_transformer import DreamConfig, DreamTransformer


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_answer(value):
    """Normalize case and punctuation while retaining letters and digits in all scripts."""
    if not isinstance(value, str):
        raise TypeError("Benchmark answers must be strings")
    output = []
    pending_space = False
    for char in value.casefold():
        category = unicodedata.category(char)
        if category[0] in ("L", "N") or category == "M":
            if pending_space and output:
                output.append(" ")
            output.append(char)
            pending_space = False
        elif char.isspace():
            pending_space = True
        elif category[0] in ("P", "S"):
            pending_space = True
    return "".join(output).strip()


def load_tasks(path):
    tasks = []
    seen = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            task = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at benchmark line {line_number}") from exc
        if not isinstance(task, dict):
            raise ValueError(f"Benchmark line {line_number} must be an object")
        for field in ("id", "category", "language", "prompt", "reference"):
            if not isinstance(task.get(field), str) or not task[field].strip():
                raise ValueError(f"Benchmark line {line_number} requires non-empty {field}")
        if task["id"] in seen:
            raise ValueError(f"Duplicate benchmark id at line {line_number}: {task['id']}")
        if "\n" in task["prompt"] or "\r" in task["prompt"]:
            raise ValueError(f"Benchmark prompt must be one line at line {line_number}")
        seen.add(task["id"])
        tasks.append(task)
    if not tasks:
        raise ValueError("Benchmark contains no tasks")
    return tasks


def summarize(results, key):
    groups = defaultdict(list)
    for result in results:
        groups[result[key]].append(result["correct"])
    return {
        name: {
            "correct": sum(values),
            "total": len(values),
            "exact_match_rate": round(sum(values) / len(values), 4),
        }
        for name, values in sorted(groups.items())
    }


def evaluate(checkpoint_path, benchmark_path, max_new_tokens=48):
    checkpoint_path = Path(checkpoint_path)
    benchmark_path = Path(benchmark_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = checkpoint.get("sft_training") or checkpoint.get("training") or {}
    completed_steps = metadata.get("completed_steps", 0)
    if not isinstance(completed_steps, int) or isinstance(completed_steps, bool) or completed_steps < 1:
        raise ValueError("Refusing to benchmark a checkpoint without completed DREAM training steps")

    config = DreamConfig(**checkpoint["config"])
    tokenizer = load_tokenizer(checkpoint, config)
    model = DreamTransformer(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    tasks = load_tasks(benchmark_path)
    results = []

    with torch.inference_mode():
        for task in tasks:
            prompt = f"USER: {task['prompt'].strip()}\nASSISTANT:"
            prompt_ids = tokenizer.encode(prompt)
            if not prompt_ids or len(prompt_ids) >= config.block_size:
                raise ValueError(f"Prompt does not fit checkpoint context: {task['id']}")
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            generated = model.generate(
                input_ids,
                max_new_tokens=min(max_new_tokens, config.block_size - len(prompt_ids)),
                temperature=0.0,
                top_k=1,
                constrain_utf8=True,
                token_byte_pieces=getattr(tokenizer, "token_byte_pieces", None),
            )[0].tolist()
            response = tokenizer.decode(generated[len(prompt_ids):]).strip()
            # These tasks request one concise answer. Ignore any later generated lines.
            response = response.splitlines()[0].strip() if response.splitlines() else ""
            correct = normalize_answer(response) == normalize_answer(task["reference"])
            results.append({
                "id": task["id"],
                "category": task["category"],
                "language": task["language"],
                "reference": task["reference"],
                "response": response,
                "correct": correct,
            })

    correct_count = sum(item["correct"] for item in results)
    return {
        "schema": "dream-core-task-evaluation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "benchmark": str(benchmark_path),
        "benchmark_sha256": sha256(benchmark_path),
        "completed_training_steps": completed_steps,
        "device": device,
        "tokenizer": type(tokenizer).__name__,
        "tasks": len(results),
        "correct": correct_count,
        "exact_match_rate": round(correct_count / len(results), 4),
        "by_category": summarize(results, "category"),
        "by_language": summarize(results, "language"),
        "results": results,
        "technical_preflight_passed": True,
        "production_release_approved": False,
        "warning": (
            "This small, authored exact-match diagnostic is not a comprehensive "
            "capability, factuality, reasoning, safety or multilingual benchmark. "
            "Do not use its score alone to claim model quality or production readiness."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark", default="evals/dream-core-v1.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--out", default="eval-results/dream-core-v1.json")
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 256:
        parser.error("--max-new-tokens must be between 1 and 256")
    report = evaluate(args.checkpoint, args.benchmark, args.max_new_tokens)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
