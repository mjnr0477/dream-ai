#!/usr/bin/env python3
"""Upload a verified DREAM training artifact to the authenticated DREAM HOST vault."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from urllib.parse import quote

import requests


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="Private ZIP or checkpoint file to upload")
    parser.add_argument("--host", default=os.environ.get("DREAM_HOST_URL", "https://dream-host-live.onrender.com"))
    args = parser.parse_args()

    token = os.environ.get("DREAM_NOTEBOOK_ARTIFACT_TOKEN", "").strip()
    if len(token) < 32:
        parser.error("DREAM_NOTEBOOK_ARTIFACT_TOKEN must be set to the dedicated private artifact token.")
    artifact = args.artifact.expanduser().resolve()
    if not artifact.is_file() or artifact.stat().st_size < 1:
        parser.error("Artifact file does not exist or is empty.")
    host = args.host.rstrip("/")
    if not host.startswith("https://") and not host.startswith("http://127.0.0.1"):
        parser.error("DREAM HOST must use HTTPS (loopback HTTP is allowed for local tests).")

    expected_hash = sha256_file(artifact)
    url = host + "/api/v1/notebook/artifacts/" + quote(artifact.name, safe="._-")
    with artifact.open("rb") as handle:
        response = requests.put(
            url,
            data=handle,
            headers={"Authorization": "Bearer " + token, "Content-Type": "application/octet-stream"},
            timeout=(15, 1800)
        )
    response.raise_for_status()
    result = response.json().get("artifact", {})
    if result.get("sha256") != expected_hash or result.get("size_bytes") != artifact.stat().st_size:
        raise RuntimeError("DREAM HOST returned artifact metadata that does not match the local file.")
    print("DREAM artifact stored.")
    print("Name:", result.get("name"))
    print("Size (bytes):", result.get("size_bytes"))
    print("SHA-256:", result.get("sha256"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
