"""Synthetic corpus-manifest fixtures for native DREAM training tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from license_policy import POLICY_VERSION, classify_record_license


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fingerprint(kind: str, value: str) -> str:
    payload = f"dream-{kind}-id-v1\0{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_test_corpus_manifest(
    root,
    train_path,
    validation_path,
    test_path=None,
    *,
    license_name="CC0-1.0",
    license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    rights_basis="original-authorship",
):
    root = Path(root).resolve()
    train_path = Path(train_path).resolve()
    validation_path = Path(validation_path).resolve()
    if test_path is None:
        test_path = root / "test.txt"
        if not test_path.exists():
            test_path.write_text(
                "Independent synthetic held-out evaluation text for DREAM. " * 12,
                encoding="utf-8"
            )
    test_path = Path(test_path).resolve()

    split_paths = {
        "train": train_path,
        "validation": validation_path,
        "test": test_path,
    }
    split_manifest = {}
    policy_counts = {}
    license_counts = {}
    source_ids = []
    all_bytes = bytearray()

    for split_name, text_path in split_paths.items():
        text_path.parent.mkdir(parents=True, exist_ok=True)
        if not text_path.is_file():
            raise FileNotFoundError(text_path)
        raw_text = text_path.read_bytes()
        all_bytes.extend(raw_text)
        record_id = _fingerprint("record", f"synthetic-{split_name}-record")
        source_group = _fingerprint("source-group", f"synthetic-{split_name}-group")
        source_ids.append(source_group)

        item = {
            "license": license_name,
            "license_url": license_url,
            "rights_basis": rights_basis,
            "rights_confirmed": True,
            "source_url": f"https://example.org/dream-synthetic/{split_name}",
            "source": "DREAM synthetic training-test fixture",
        }
        decision = classify_record_license(item)
        provenance_record = {
            "record_id_sha256": record_id,
            "source_group_sha256": source_group,
            "source": item["source"],
            "source_url": item["source_url"],
            "license": license_name,
            "license_url": license_url,
            "rights_basis": rights_basis,
            "source_date": "2026-01-15",
            "language": "en",
            "license_policy_code": decision["code"],
            "attribution_required": decision["attribution_required"],
            "rights_confirmed": True,
            "privacy_reviewed": True,
            "quality_reviewed": True,
            "approval_granted": True,
            "consent_version": None,
            "consented": None,
            "consented_at": None,
            "human_reviewed_at": None,
            "permission_scope": None,
            "permission_evidence_url": None,
        }
        provenance_path = root / f"{split_name}.provenance.jsonl"
        provenance_bytes = (
            json.dumps(provenance_record, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        provenance_path.write_bytes(provenance_bytes)
        all_bytes.extend(provenance_bytes)

        split_policy_counts = {decision["code"]: 1}
        policy_counts[decision["code"]] = policy_counts.get(decision["code"], 0) + 1
        license_counts[license_name] = license_counts.get(license_name, 0) + 1
        split_manifest[split_name] = {
            "path": text_path.name,
            "records": 1,
            "utf8_bytes": len(raw_text),
            "record_ids": [record_id],
            "source_group_ids": [source_group],
            "sha256": _sha256(raw_text),
            "source_groups": 1,
            "language_counts": {"en": 1},
            "license_counts": {license_name: 1},
            "license_policy_counts": split_policy_counts,
            "source_date_range": {"earliest": "2026-01-15", "latest": "2026-01-15"},
            "language_files": {},
            "provenance": {
                "path": provenance_path.name,
                "records": 1,
                "sha256": _sha256(provenance_bytes),
                "contains_training_text": False,
                "contains_raw_record_or_group_ids": False,
            },
        }

    manifest = {
        "schema": "dream-corpus-v1",
        "identifier_format": "sha256-domain-separated-v1",
        "input_sha256": _sha256(bytes(all_bytes)),
        "records": 3,
        "duplicate_text_records_removed": 0,
        "source_groups": 3,
        "language_counts": {"en": 3},
        "license_counts": license_counts,
        "license_policy_version": POLICY_VERSION,
        "license_policy_counts": policy_counts,
        "rights_basis_counts": {rights_basis: 3},
        "source_date_range": {"earliest": "2026-01-15", "latest": "2026-01-15"},
        "split_strategy": "synthetic source-group-disjoint test fixture",
        "language_split_coverage": {"en": {"train": 1, "validation": 1, "test": 1}},
        "splits": split_manifest,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8"
    )
    return manifest_path
