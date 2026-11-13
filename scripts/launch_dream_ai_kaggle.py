#!/usr/bin/env python3
"""Safely launch DREAM AI's reviewed Kaggle adapter-training notebook."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "training" / "dream_adapter_improvement_notebook.ipynb"
REQUIRED_FILES = ("train.jsonl", "validation.jsonl", "manifest.json", "audit-gate.json")
KERNEL_SLUG = "dream-ai-autonomous-adapter-training"


def validate_dataset_id(value: str) -> str:
    value = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", value):
        raise ValueError("DREAM_AI_KAGGLE_DATASET must be in owner/dataset-slug form.")
    return value


def missing_required_files(csv_text: str) -> list[str]:
    rows = list(csv.DictReader(csv_text.splitlines()))
    if not rows:
        raise ValueError("Kaggle returned no dataset file rows.")
    names = []
    for row in rows:
        name = next((row[key] for key in ("name", "Name", "fileName", "filename") if row.get(key)), None)
        if name:
            names.append(Path(name).name)
    duplicates = [name for name in REQUIRED_FILES if names.count(name) > 1]
    if duplicates:
        raise ValueError("Dataset has duplicate required basenames: " + ", ".join(duplicates))
    return [name for name in REQUIRED_FILES if names.count(name) != 1]



def verify_private_dataset(metadata: dict, username: str, dataset_id: str) -> None:
    owner = dataset_id.split("/", 1)[0]
    if owner.lower() != username.strip().lower():
        raise RuntimeError("The training dataset must be owned by the authenticated Kaggle account.")
    info = metadata.get("info", metadata)
    privacy = info.get("isPrivate") if isinstance(info, dict) else None
    if privacy is not True and str(privacy).strip().lower() not in ("true", "1", "yes"):
        raise RuntimeError("Kaggle metadata did not confirm that the dataset is private; refusing to attach training data.")


def fetch_private_dataset_metadata(dataset_id: str, username: str, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "datasets", "metadata", dataset_id, "--path", str(directory)])
    metadata_path = directory / "dataset-metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"Kaggle did not produce dataset metadata for {dataset_id}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    verify_private_dataset(metadata, username, dataset_id)
    return metadata


def private_dataset_refs(csv_text: str, username: str) -> list[str]:
    # `datasets list --mine` only returns the authenticated user's datasets;
    # privacy is verified separately through the dataset metadata endpoint.
    refs = []
    for row in csv.DictReader(csv_text.splitlines()):
        ref = next((row[key] for key in ("ref", "Ref", "datasetRef", "dataset_ref") if row.get(key)), "")
        if (
            ref and "/" in ref
            and ref.split("/", 1)[0].strip().lower() == username.strip().lower()
        ):
            refs.append(ref.strip())
    return sorted(set(refs))


def format_readiness_summary(dataset_reports: list[dict]) -> str:
    """Explain why adapter training is waiting without exposing private dataset contents."""
    lines = [
        "## DREAM AI adapter-training readiness",
        "",
        "**Status: NOT STARTED**",
        "",
        "The launcher checked private datasets owned by the configured Kaggle account, but none contained exactly one copy of every required file:",
        "",
        *[f"- `{name}`" for name in REQUIRED_FILES],
        "",
        "### Dataset file checks",
    ]
    if dataset_reports:
        for report in dataset_reports:
            label = report.get("label", "Private dataset")
            if not re.fullmatch(r"Private dataset [0-9]+", label):
                label = "Private dataset"
            if report.get("inventory_error") is True:
                lines.append(f"- **{label}**: file inventory could not be verified")
                continue
            missing = [name for name in REQUIRED_FILES if name in report.get("missing", list(REQUIRED_FILES))]
            lines.append(f"- **{label}**: missing " + (", ".join(f"`{name}`" for name in missing) if missing else "no missing files"))
    else:
        lines.append("- No verifiable private datasets were found in the account inventory.")
    lines.extend([
        "",
        "Training was not launched. No model or production adapter files were changed.",
        "",
        "The required files must come from DREAM's reviewed opt-in export, human review, source-group-isolated split preparation, and matching privacy/provenance audit. Do not bypass those gates or mark review flags automatically.",
        "The notebook separately requires at least 32 human-approved examples from at least 8 independent session groups, completed rights/privacy/quality approval, a Kaggle GPU, and both Kaggle Secrets HF_TOKEN and DREAM_NOTEBOOK_ARTIFACT_TOKEN. A passing run uploads a candidate only; it does not replace the production adapter.",
    ])
    return "\n".join(lines) + "\n"


def write_step_summary(summary: str, env=os.environ) -> None:
    """Append a readiness report to GitHub Actions when running inside a workflow."""
    summary_path = env.get("GITHUB_STEP_SUMMARY", "").strip()
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write(summary)



def build_kernel_metadata(username: str, dataset_id: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", username.strip()):
        raise ValueError("KAGGLE_USERNAME is missing or invalid.")
    return {
        "id": f"{username.strip()}/{KERNEL_SLUG}",
        "title": "DREAM AI Autonomous Adapter Training",
        "code_file": "dream_adapter_improvement_notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [validate_dataset_id(dataset_id)],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def configure_kaggle_credentials(username: str, api_token: str) -> None:
    if not username.strip() or not api_token.strip():
        raise ValueError("KAGGLE_USERNAME and KAGGLE_API_TOKEN are required.")
    os.environ["KAGGLE_API_TOKEN"] = api_token.strip()


def classify_kernel_status(status: str) -> str:
    """Fail closed when Kaggle reports a state the launcher cannot recognize."""
    value = str(status or "").strip().lower()
    if any(word in value for word in ("running", "queued", "initializing", "starting")):
        return "active"
    if any(word in value for word in ("complete", "succeeded", "success")):
        return "complete"
    if any(word in value for word in ("error", "failed", "cancelled", "canceled", "timed out")):
        return "failed"
    return "unknown"


def get_kernel_status(kernel_id: str) -> str | None:
    """Return None only for a confirmed missing kernel; propagate other API failures."""
    probe = subprocess.run(
        ["kaggle", "kernels", "status", kernel_id],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = (probe.stdout or "").strip()
    if probe.returncode == 0:
        if not output:
            raise RuntimeError("Kaggle returned an empty kernel status; refusing to launch.")
        return output
    if re.search(r"not found|does not exist|no such kernel|404", output, re.IGNORECASE):
        return None
    raise RuntimeError(
        "Could not verify the existing Kaggle kernel status; refusing to submit a duplicate run. "
        + (output[-1000:] if output else "Kaggle status request failed without details.")
    )


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except subprocess.CalledProcessError as exc:
        details = "\n".join(
            part.strip() for part in (exc.stdout, exc.stderr) if part and part.strip()
        )
        for token in (os.environ.get("KAGGLE_API_TOKEN", ""), os.environ.get("KAGGLE_KEY", "")):
            if token and details:
                details = details.replace(token, "[REDACTED]")
        message = f"Kaggle command failed with exit code {exc.returncode}."
        if details:
            message += "\nKaggle reported: " + details[-4000:]
        raise RuntimeError(message) from exc


def launch(*, force_retry: bool = False) -> int:
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    api_token = os.environ.get("KAGGLE_API_TOKEN", "").strip() or os.environ.get("KAGGLE_KEY", "").strip()
    dataset_id = os.environ.get("DREAM_AI_KAGGLE_DATASET", "").strip()
    if not username or not api_token:
        missing = []
        if not username:
            missing.append("KAGGLE_USERNAME")
        if not api_token:
            missing.append("KAGGLE_API_TOKEN")
        print("WAITING_FOR_SETUP: missing " + ", ".join(missing))
        print("No training was started and no model files were changed.")
        write_step_summary(
            "## DREAM AI automatic GPU training\n\n"
            "**Status: WAITING FOR ONE-TIME SETUP**\n\n"
            "Add GitHub repository secrets KAGGLE_USERNAME and KAGGLE_API_TOKEN. "
            "No Kaggle job was submitted.\n"
        )
        return 0

    if not NOTEBOOK.is_file():
        raise FileNotFoundError(f"Training notebook not found: {NOTEBOOK}")
    configure_kaggle_credentials(username, api_token)

    own_datasets = run(
        ["kaggle", "datasets", "list", "--mine", "--csv", "--page-size", "100"],
        capture=True,
    )
    private_refs = private_dataset_refs(own_datasets.stdout, username)
    with tempfile.TemporaryDirectory(prefix="dream-ai-dataset-check-") as check_temp:
        check_root = Path(check_temp)
        if dataset_id:
            dataset_id = validate_dataset_id(dataset_id)
            fetch_private_dataset_metadata(dataset_id, username, check_root / "metadata")
            listing = run(
                ["kaggle", "datasets", "files", dataset_id, "--csv", "--page-size", "200"],
                capture=True,
            )
        else:
            matching = []
            dataset_reports = []
            for index, candidate_ref in enumerate(private_refs):
                try:
                    fetch_private_dataset_metadata(
                        candidate_ref, username, check_root / f"metadata-{index}"
                    )
                except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError):
                    continue
                candidate_files = run(
                    ["kaggle", "datasets", "files", candidate_ref, "--csv", "--page-size", "200"],
                    capture=True,
                )
                try:
                    candidate_missing = missing_required_files(candidate_files.stdout)
                except ValueError:
                    dataset_reports.append({
                        "label": f"Private dataset {index + 1}",
                        "inventory_error": True,
                    })
                    continue
                dataset_reports.append({
                    "label": f"Private dataset {index + 1}",
                    "missing": candidate_missing,
                })
                if not candidate_missing:
                    matching.append((candidate_ref, candidate_files))
            if not matching:
                print("WAITING_FOR_PRIVATE_REVIEWED_DATASET: no private dataset with all four required files was found.")
                for report in dataset_reports:
                    label = report["label"]
                    if report.get("inventory_error") is True:
                        print(f"DATASET_FILE_CHECK: {label} | inventory could not be verified")
                    else:
                        missing_text = ", ".join(report["missing"]) if report["missing"] else "none"
                        print(f"DATASET_FILE_CHECK: {label} | missing: {missing_text}")
                summary = format_readiness_summary(dataset_reports)
                print(summary)
                write_step_summary(summary)
                print("No training was started and no model files were changed.")
                return 0
            if len(matching) > 1:
                raise RuntimeError("More than one eligible private dataset exists; set DREAM_AI_KAGGLE_DATASET to select one. Dataset identifiers are omitted from logs.")
            dataset_id, listing = matching[0]
            print("AUTO_DISCOVERED_PRIVATE_DATASET: one eligible dataset found; identifier omitted from public logs.")

    missing = missing_required_files(listing.stdout)
    if missing:
        raise RuntimeError("Dataset must contain exactly one copy of: " + ", ".join(missing))

    metadata = build_kernel_metadata(username, dataset_id)
    kernel_id = metadata["id"]
    with tempfile.TemporaryDirectory(prefix="dream-ai-kaggle-") as temp:
        workdir = Path(temp)
        notebook_name = "dream_adapter_improvement_notebook.ipynb"
        (workdir / notebook_name).write_bytes(NOTEBOOK.read_bytes())
        (workdir / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

        # The current Kaggle CLI runs a notebook when it is first pushed.
        # Cell 1 checks HF_TOKEN before loading a model or starting training.
        # The owner can add that Kaggle Secret after this first registration run.
        status = get_kernel_status(kernel_id)
        if status is None:
            print("CREATING_PRIVATE_KAGGLE_KERNEL")
            run([
                "kaggle", "kernels", "push", "--path", str(workdir),
                "--accelerator", "NvidiaTeslaT4", "--timeout", "7200",
            ])
            print("KAGGLE_TRAINING_LAUNCH: FIRST_RUN_SUBMITTED")
            print("The private kernel reads HF_TOKEN and DREAM_NOTEBOOK_ARTIFACT_TOKEN from Kaggle Secrets.")
            return 0

        status_class = classify_kernel_status(status)
        print("KAGGLE_KERNEL_STATUS:", status)
        if status_class == "active":
            print("SKIPPED: the Kaggle GPU kernel is already active.")
            return 0
        if status_class == "failed" and not force_retry:
            print("AUTOMATIC_RETRY_PAUSED: the last Kaggle run failed or was cancelled.")
            print("Review the private Kaggle run logs, fix the cause, then manually dispatch with force_retry=true.")
            return 0
        if status_class == "unknown":
            raise RuntimeError("Kaggle returned an unrecognized kernel status; refusing to burn GPU quota.")

        print("DATASET_PREFLIGHT: PASSED")
        print("KAGGLE_KERNEL: private training notebook selected; identifier omitted from public logs.")
        print("PRIVATE_DATASET: one eligible dataset selected; identifier omitted from public logs.")
        print("PRODUCTION_ADAPTER: protected; notebook uploads candidates only")
        run([
            "kaggle", "kernels", "push", "--path", str(workdir),
            "--accelerator", "NvidiaTeslaT4", "--timeout", "7200",
        ])
        print("KAGGLE_TRAINING_LAUNCH: SUBMITTED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="test helper logic without Kaggle credentials")
    parser.add_argument("--force-retry", action="store_true", help="retry a failed Kaggle run after human review")
    args = parser.parse_args()
    if args.check:
        metadata = build_kernel_metadata("test-user", "test-user/test-dataset")
        assert metadata["is_private"] == "true"
        assert metadata["dataset_sources"] == ["test-user/test-dataset"]
        assert metadata["code_file"] == "dream_adapter_improvement_notebook.ipynb"
        assert classify_kernel_status("KernelWorkerStatus.RUNNING") == "active"
        assert classify_kernel_status("KernelWorkerStatus.COMPLETE") == "complete"
        assert classify_kernel_status("KernelWorkerStatus.ERROR") == "failed"
        assert classify_kernel_status("some-new-status") == "unknown"
        sample = (
            "name,size\ntrain.jsonl,10\nvalidation.jsonl,10\n"
            "manifest.json,10\naudit-gate.json,10\n"
        )
        assert missing_required_files(sample) == []
        verify_private_dataset(
            {"info": {"isPrivate": True}}, "test-user", "test-user/test-dataset"
        )
        assert private_dataset_refs(
            "ref,title,size\ntest-user/test-dataset,Reviewed,10\ntest-user/public-data,Public,10\n",
            "test-user",
        ) == ["test-user/public-data", "test-user/test-dataset"]
        readiness = format_readiness_summary([{
            "label": "Private dataset 1",
            "missing": ["train.jsonl", "audit-gate.json"],
        }])
        assert "test-user/incomplete-data" not in readiness
        assert "Status: NOT STARTED" in readiness
        assert "train.jsonl" in readiness and "audit-gate.json" in readiness
        assert "No model or production adapter files were changed." in readiness
        unreadable = format_readiness_summary([{
            "label": "Private dataset 2",
            "inventory_error": True,
        }])
        assert "file inventory could not be verified" in unreadable
        assert "Private dataset 2" in unreadable
        try:
            verify_private_dataset(
                {"info": {"isPrivate": False}}, "test-user", "test-user/test-dataset"
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("public dataset was accepted")
        try:
            validate_dataset_id("not-a-dataset-id")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid dataset id was accepted")
        print("LAUNCHER SELF-CHECK: PASSED")
        return 0
    try:
        return launch(force_retry=args.force_retry)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"AUTONOMOUS_LAUNCH_BLOCKED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
