"""Aggregate human rubric scores from a DREAM native task benchmark report.

This tool never grades model output. A reviewer fills each rubric_scores entry
with 0 or 1 in the generated report; this script validates and aggregates those
human judgments, keeping incomplete tasks visible.
"""
import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path


def validate_report(report):
    if not isinstance(report, dict) or report.get("schema") != "dream-native-task-benchmark-v1":
        raise ValueError("Input must be a dream-native-task-benchmark-v1 report.")
    results = report.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("Report must contain a non-empty results list.")
    seen_ids = set()
    normalized = []
    for index, item in enumerate(results, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Result {index} must be an object.")
        task_id = item.get("id")
        category = item.get("category")
        rubric = item.get("rubric")
        scores = item.get("rubric_scores")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"Result {index} has an invalid task id.")
        if task_id in seen_ids:
            raise ValueError(f"Duplicate task id: {task_id}")
        seen_ids.add(task_id)
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"Task {task_id} has an invalid category.")
        if not isinstance(rubric, list) or not rubric or any(
            not isinstance(point, str) or not point.strip() for point in rubric
        ):
            raise ValueError(f"Task {task_id} has an invalid rubric.")
        if not isinstance(scores, list) or len(scores) != len(rubric):
            raise ValueError(f"Task {task_id} must have one rubric score per rubric item.")
        for score in scores:
            if score is not None and (type(score) is not int or score not in (0, 1)):
                raise ValueError(f"Task {task_id} rubric scores must be 0, 1 or null.")
        normalized.append((task_id, category, scores))
    declared_count = report.get("task_count")
    if declared_count != len(results):
        raise ValueError("task_count does not match the number of results.")
    return normalized


def aggregate(report):
    tasks = validate_report(report)
    overall_scored = 0
    overall_points = 0
    complete_tasks = 0
    categories = defaultdict(lambda: {
        "tasks": 0, "complete_tasks": 0, "scored_rubric_items": 0,
        "earned_points": 0
    })
    task_results = []
    for task_id, category, scores in tasks:
        scored = [score for score in scores if score is not None]
        earned = sum(scored)
        complete = len(scored) == len(scores)
        overall_scored += len(scored)
        overall_points += earned
        complete_tasks += int(complete)
        group = categories[category]
        group["tasks"] += 1
        group["complete_tasks"] += int(complete)
        group["scored_rubric_items"] += len(scored)
        group["earned_points"] += earned
        task_results.append({
            "id": task_id,
            "category": category,
            "rubric_items": len(scores),
            "scored_rubric_items": len(scored),
            "earned_points": earned,
            "complete": complete,
            "score_percent": round(100 * earned / len(scored), 2) if scored else None
        })

    def finalize(group):
        count = group["scored_rubric_items"]
        return {
            **group,
            "score_percent": round(100 * group["earned_points"] / count, 2) if count else None
        }

    total_items = sum(len(scores) for _, _, scores in tasks)
    return {
        "schema": "dream-human-task-score-summary-v1",
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "benchmark_sha256": report.get("benchmark_sha256"),
        "tokenizer": report.get("tokenizer"),
        "parameters": report.get("parameters"),
        "completed_training_steps": report.get("completed_training_steps"),
        "data_provenance_verified": report.get("data_provenance_verified") is True,
        "task_count": len(tasks),
        "complete_tasks": complete_tasks,
        "task_completion_percent": round(100 * complete_tasks / len(tasks), 2),
        "scored_rubric_items": overall_scored,
        "total_rubric_items": total_items,
        "scoring_coverage_percent": round(100 * overall_scored / total_items, 2),
        "earned_points": overall_points,
        "score_percent": round(100 * overall_points / overall_scored, 2) if overall_scored else None,
        "scoring_method": "Human-assigned binary rubric scores only; no model-based grading.",
        "categories": {
            name: finalize(categories[name]) for name in sorted(categories)
        },
        "tasks": task_results,
        "limitations": [
            "Scores reflect the human rubric judgments entered in the source report.",
            "Incomplete rubric items are excluded from accuracy percentages and shown in coverage.",
            "This summary does not independently verify reviewer identity, rubric quality or answer correctness."
        ]
    }


def write_json_atomic(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent,
            prefix=output.name + ".", suffix=".tmp", delete=False
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, output)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate human-assigned rubric scores for a DREAM task report."
    )
    parser.add_argument("--input", required=True, help="Human-reviewed task benchmark JSON")
    parser.add_argument("--out", default="eval-results/human-score-summary.json")
    args = parser.parse_args()
    try:
        report = json.loads(Path(args.input).read_text(encoding="utf-8"))
        summary = aggregate(report)
        write_json_atomic(args.out, summary)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Could not score DREAM task report: {exc}") from exc
    print(json.dumps({
        "summary": str(args.out),
        "tasks": summary["task_count"],
        "complete_tasks": summary["complete_tasks"],
        "scored_rubric_items": summary["scored_rubric_items"],
        "score_percent": summary["score_percent"]
    }, indent=2))


if __name__ == "__main__":
    main()
