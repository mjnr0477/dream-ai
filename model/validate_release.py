"""Cross-check DREAM checkpoint audit and held-out evaluation artifacts.

This is a technical consistency gate, not a model-quality or production certification.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read_json(path, label):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(checkpoint_path, audit_path, evaluation_path):
    checkpoint_hash = sha256(checkpoint_path)
    audit = read_json(audit_path, "checkpoint audit")
    evaluation = read_json(evaluation_path, "held-out evaluation")

    if audit.get("schema") != "dream-checkpoint-audit-v1":
        raise ValueError("Unsupported checkpoint audit schema")
    if evaluation.get("schema") != "dream-evaluation-v1":
        raise ValueError("Unsupported evaluation report schema")
    if audit.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Checkpoint audit hash does not match supplied checkpoint")
    if evaluation.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Evaluation checkpoint hash does not match supplied checkpoint")
    if audit.get("strict_state_dict_load") is not True:
        raise ValueError("Checkpoint audit did not confirm strict state loading")
    if audit.get("all_floating_tensors_finite") is not True:
        raise ValueError("Checkpoint audit did not confirm finite weights")
    steps = audit.get("completed_steps")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("Checkpoint audit has invalid training-step metadata")
    if audit.get("parameter_count") != evaluation.get("parameters"):
        raise ValueError("Audit and evaluation parameter counts do not match")

    corpus_hash = evaluation.get("evaluation_corpus_sha256")
    if not isinstance(corpus_hash, str) or len(corpus_hash) != 64:
        raise ValueError("Evaluation report has no valid held-out corpus hash")
    metrics = (
        "mean_cross_entropy_nats_per_token",
        "mean_cross_entropy_nats_per_byte",
        "bits_per_byte",
        "perplexity_per_byte",
    )
    for name in metrics:
        value = evaluation.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Evaluation metric {name} is missing or non-finite")
        if value < 0:
            raise ValueError(f"Evaluation metric {name} cannot be negative")

    return {
        "schema": "dream-technical-release-gate-v1",
        "technical_preflight_passed": True,
        "checkpoint_sha256": checkpoint_hash,
        "evaluation_corpus_sha256": corpus_hash,
        "completed_training_steps": steps,
        "parameter_count": audit["parameter_count"],
        "checks": [
            "checkpoint identity matches audit and evaluation",
            "checkpoint audit confirms strict tensor loading and finite weights",
            "evaluation report uses the same checkpoint",
            "evaluation metrics are present and finite",
            "parameter counts agree across artifacts",
        ],
        "limitations": [
            "This gate verifies artifact consistency and basic integrity only.",
            "It does not establish reasoning, coding, factuality, multilingual fluency, safety or usefulness.",
            "Human review and independent task benchmarks are required before any production-quality claim.",
            "A passing technical preflight is not production release approval.",
        ],
        "production_release_approved": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--out", default="eval-results/technical-release-gate.json")
    args = parser.parse_args()
    try:
        report = validate(args.checkpoint, args.audit, args.evaluation)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Technical release gate failed: {exc}") from exc
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
