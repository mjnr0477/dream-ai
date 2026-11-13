#!/usr/bin/env python3
"""Run the reviewed native DREAM checkpoint as a persistent GPU chat worker.

No Transformers, PEFT, Kaggle or Hugging Face packages are used. The worker
downloads one SHA-256-pinned instruction checkpoint from the existing DREAM R2
artifact API, validates it with model/runtime_checkpoint.py, and joins the
existing DREAM HOST chat queue.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import signal
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import requests
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from runtime_checkpoint import MAX_CHECKPOINT_BYTES, load_validated_checkpoint  # noqa: E402

HOST = os.environ.get("DREAM_HOST_URL", "https://dream-host-live.onrender.com").rstrip("/")
WORKER_TOKEN = os.environ.get("DREAM_HOST_WORKER_TOKEN", "").strip()
ARTIFACT_TOKEN = os.environ.get("DREAM_HOST_ARTIFACT_TOKEN", "").strip()
MODEL_NAME = os.environ.get("DREAM_HOST_MODEL_NAME", "dream-native-v1").strip()
ADAPTER_NAME = os.environ.get("DREAM_HOST_ADAPTER_NAME", "none").strip()
ARTIFACT_NAME = os.environ.get(
    "DREAM_NATIVE_CHECKPOINT_ARTIFACT", "dream-native-sft-checkpoint.pt"
).strip()
APPROVED_SHA256 = os.environ.get("DREAM_NATIVE_CHECKPOINT_SHA256", "").strip().lower()
CHECKPOINT_PATH = Path(os.environ.get(
    "DREAM_NATIVE_CHECKPOINT_PATH", "/var/lib/dream-ai/models/dream-native-instruct.pt"
)).expanduser().resolve()
WORKER_ID = os.environ.get("DREAM_HOST_WORKER_ID", "native-" + uuid.uuid4().hex[:16])
WORKER_NAME = "DREAM native inference worker"
POLL_SECONDS = max(0.5, min(float(os.environ.get("DREAM_HOST_WORKER_POLL_SECONDS", "1.5")), 10.0))
MAX_NEW_TOKENS = 256
INFERENCE_SECONDS = 50
STOP = threading.Event()
STATE = {"busy": False}

DEVICE = "cuda"
GPU_NAME = ""
MODEL = None
TOKENIZER = None
CONFIG = None
CHECKPOINT_METADATA: dict[str, Any] = {}


def require_configuration() -> None:
    if not HOST.startswith("https://") and not HOST.startswith("http://127.0.0.1"):
        raise RuntimeError("DREAM_HOST_URL must use HTTPS.")
    if len(WORKER_TOKEN) < 32 or len(ARTIFACT_TOKEN) < 32:
        raise RuntimeError("DREAM_HOST_WORKER_TOKEN and DREAM_HOST_ARTIFACT_TOKEN are required.")
    if WORKER_TOKEN == ARTIFACT_TOKEN:
        raise RuntimeError("The worker and artifact tokens must be distinct.")
    if not MODEL_NAME or not ADAPTER_NAME or ADAPTER_NAME.lower() != "none":
        raise RuntimeError("Native inference requires DREAM_HOST_MODEL_NAME and DREAM_HOST_ADAPTER_NAME=none.")
    if not ARTIFACT_NAME or "/" in ARTIFACT_NAME or "\\" in ARTIFACT_NAME:
        raise RuntimeError("DREAM_NATIVE_CHECKPOINT_ARTIFACT must be a simple R2 artifact name.")
    if len(APPROVED_SHA256) != 64 or any(char not in "0123456789abcdef" for char in APPROVED_SHA256):
        raise RuntimeError("Set DREAM_NATIVE_CHECKPOINT_SHA256 to the independently approved checkpoint SHA-256.")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is visible. Native inference refuses to fall back to CPU.")
    if not WORKER_ID or len(WORKER_ID) < 6:
        raise RuntimeError("DREAM_HOST_WORKER_ID is invalid.")


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": "Bearer " + token, "Content-Type": "application/json"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_json(url: str, token: str, timeout: int = 30) -> dict[str, Any]:
    response = requests.get(url, headers=auth_headers(token), timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("DREAM HOST returned an invalid JSON object.")
    return payload


def download_approved_checkpoint() -> tuple[str, int]:
    """Download only the exact approved artifact version; never use an unpinned latest."""
    artifacts = get_json(HOST + "/api/v1/notebook/artifacts", ARTIFACT_TOKEN).get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("DREAM R2 artifact list is missing.")
    matches = [
        item for item in artifacts
        if isinstance(item, dict)
        and item.get("name") == ARTIFACT_NAME
        and item.get("sha256") == APPROVED_SHA256
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "The exact approved native checkpoint was not found in DREAM R2. "
            "Refusing to load a different or newer artifact."
        )
    item = matches[0]
    size = item.get("size_bytes")
    if not isinstance(size, int) or isinstance(size, bool) or not 1 <= size <= MAX_CHECKPOINT_BYTES:
        raise RuntimeError("The approved checkpoint size is outside the native runtime limit.")

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT_PATH.is_file() and CHECKPOINT_PATH.stat().st_size == size:
        if sha256_file(CHECKPOINT_PATH) == APPROVED_SHA256:
            return APPROVED_SHA256, size

    temporary = CHECKPOINT_PATH.with_suffix(CHECKPOINT_PATH.suffix + ".part")
    digest = hashlib.sha256()
    written = 0
    response = requests.get(
        HOST + "/api/v1/notebook/artifacts/" + APPROVED_SHA256,
        headers=auth_headers(ARTIFACT_TOKEN),
        stream=True,
        timeout=(20, 1800),
    )
    response.raise_for_status()
    try:
        with temporary.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                written += len(chunk)
                if written > MAX_CHECKPOINT_BYTES or written > size:
                    raise RuntimeError("R2 checkpoint exceeded its declared or permitted size.")
                digest.update(chunk)
                output.write(chunk)
        if written != size or digest.hexdigest() != APPROVED_SHA256:
            raise RuntimeError("R2 checkpoint SHA-256 or byte count does not match the approved release.")
        os.replace(temporary, CHECKPOINT_PATH)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return APPROVED_SHA256, size


def check_host_model_configuration() -> None:
    response = requests.get(HOST + "/health", timeout=30)
    response.raise_for_status()
    status = response.json()
    if status.get("model") != MODEL_NAME or status.get("adapter") != ADAPTER_NAME:
        raise RuntimeError(
            "DREAM HOST model configuration does not match this native worker. "
            "Set DREAM_HOST_MODEL_NAME=dream-native-v1 and DREAM_HOST_ADAPTER_NAME=none "
            "on dream-host-live before enabling native inference."
        )


def load_native_model() -> None:
    global MODEL, TOKENIZER, CONFIG, CHECKPOINT_METADATA, GPU_NAME
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    GPU_NAME = torch.cuda.get_device_name(0)
    print("DREAM native GPU:", GPU_NAME, flush=True)
    check_host_model_configuration()
    digest, size = download_approved_checkpoint()
    print("Approved native checkpoint downloaded:", size, "bytes; sha256:", digest, flush=True)

    payload = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    sft = payload.get("sft_training") if isinstance(payload, dict) else None
    if not isinstance(sft, dict) or not isinstance(sft.get("completed_steps"), int) or sft["completed_steps"] < 1:
        raise RuntimeError("The approved artifact is not a completed native DREAM instruction-tuning checkpoint.")
    del payload

    MODEL, TOKENIZER, CONFIG, CHECKPOINT_METADATA = load_validated_checkpoint(CHECKPOINT_PATH)
    MODEL.to(DEVICE).eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    print(
        "DREAM native model validated and loaded:",
        {"device": GPU_NAME, "parameters": sum(p.numel() for p in MODEL.parameters()),
         "sft_steps": CHECKPOINT_METADATA.get("completed_steps"), "checkpoint_sha256": digest},
        flush=True,
    )


def clean_assistant_output(text: str) -> str:
    value = text.strip()
    kept = []
    fence_char = None
    for line in value.splitlines():
        marker = line.lstrip()
        if fence_char is None:
            if marker.startswith("```") or marker.startswith("~~~"):
                fence_char = marker[0]
            elif line.startswith(("USER:", "ASSISTANT:", "SYSTEM:")):
                break
        elif marker.startswith(fence_char * 3):
            fence_char = None
        kept.append(line)
    return "\n".join(kept).rstrip()


def generate(job: dict[str, Any]) -> tuple[str, int, int]:
    if MODEL is None or TOKENIZER is None or CONFIG is None:
        raise RuntimeError("DREAM native inference model is not loaded.")
    messages = job.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 64:
        raise ValueError("Chat messages are missing or invalid.")
    lines = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant"):
            raise ValueError("A chat message has an unsupported role.")
        content = message.get("content")
        if not isinstance(content, str) or len(content) > 12000:
            raise ValueError("Chat message content is invalid or too long.")
        lines.append(f"{message['role'].upper()}: {content.strip()}\n")
    prompt_ids = TOKENIZER.encode("".join(lines) + "ASSISTANT:")
    if not prompt_ids:
        raise ValueError("The native DREAM prompt is empty.")
    prompt_ids = prompt_ids[-CONFIG.block_size:]
    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

    raw_tokens = job.get("max_tokens", 128)
    try:
        requested_tokens = int(raw_tokens)
    except (TypeError, ValueError, OverflowError):
        requested_tokens = 128
    max_new_tokens = max(1, min(requested_tokens, MAX_NEW_TOKENS, CONFIG.block_size))
    try:
        temperature = float(job.get("temperature", 0.7))
    except (TypeError, ValueError, OverflowError):
        temperature = 0.7
    if not math.isfinite(temperature):
        temperature = 0.7
    temperature = max(0.0, min(temperature, 2.0))

    if temperature > 0:
        torch.manual_seed(secrets.randbits(63))
        torch.cuda.manual_seed_all(secrets.randbits(63))

    deadline = time.monotonic() + INFERENCE_SECONDS
    with torch.inference_mode():
        output = MODEL.generate(
            token_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=min(40, CONFIG.vocab_size),
            constrain_utf8=True,
            token_byte_pieces=getattr(TOKENIZER, "token_byte_pieces", None),
            deadline_monotonic=deadline,
        )
    torch.cuda.synchronize()
    generated = output[0, len(prompt_ids):].tolist()
    content = clean_assistant_output(TOKENIZER.decode(generated))
    if not content:
        raise RuntimeError("Native DREAM produced an empty answer.")
    return content, len(prompt_ids), len(generated)


def post(path: str, payload: dict[str, Any], timeout: int = 30) -> requests.Response:
    response = requests.post(
        HOST + path, json=payload, headers=auth_headers(WORKER_TOKEN), timeout=timeout
    )
    response.raise_for_status()
    return response


def registration_payload() -> dict[str, Any]:
    return {
        "workerId": WORKER_ID,
        "name": WORKER_NAME,
        "metadata": {
            "device": GPU_NAME,
            "model": MODEL_NAME,
            "adapter": ADAPTER_NAME,
            "engine": "dream-native-transformer",
            "checkpoint_sha256": APPROVED_SHA256,
            "completed_training_steps": CHECKPOINT_METADATA.get("completed_steps"),
        },
    }


def register_until_ready() -> None:
    payload = registration_payload()
    while not STOP.is_set():
        try:
            print("Registering native DREAM inference worker with DREAM HOST.", flush=True)
            response = post("/api/v1/workers/register", payload, timeout=90)
            response.raise_for_status()
            print("Native DREAM inference worker registered:", WORKER_ID, flush=True)
            return
        except requests.RequestException as exc:
            print("DREAM HOST unavailable during registration; retrying:", str(exc)[:250], flush=True)
            STOP.wait(5)
    raise RuntimeError("Native DREAM worker stopped before registration.")


def heartbeat_loop() -> None:
    while not STOP.wait(15):
        try:
            response = post("/api/v1/workers/heartbeat", {
                "workerId": WORKER_ID,
                "name": WORKER_NAME,
                "busy": STATE["busy"],
                "metadata": registration_payload()["metadata"],
            }, timeout=20)
            response.raise_for_status()
        except Exception as exc:
            print("Native DREAM heartbeat failed; will retry:", str(exc)[:250], flush=True)


def run_jobs() -> None:
    register_until_ready()
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    while not STOP.is_set():
        try:
            response = post("/api/v1/workers/jobs/claim", {"workerId": WORKER_ID}, timeout=35)
            if response.status_code == 204:
                STOP.wait(POLL_SECONDS)
                continue
            if response.status_code == 409:
                print("DREAM HOST rejected a job claim; check model configuration.", flush=True)
                STOP.wait(5)
                continue
            response.raise_for_status()
            job = response.json()["job"]
            job_id = job.get("id")
            STATE["busy"] = True
            print("Native DREAM processing job:", job_id, flush=True)
            try:
                content, prompt_tokens, completion_tokens = generate(job)
                result = post("/api/v1/workers/jobs/" + str(job_id) + "/result", {
                    "workerId": WORKER_ID,
                    "content": content,
                    "promptTokens": prompt_tokens,
                    "completionTokens": completion_tokens,
                }, timeout=90)
                result.raise_for_status()
                print("Native DREAM completed job:", job_id, flush=True)
            except Exception as exc:
                # Do not log user prompts or generated text.
                print("Native DREAM job failed:", type(exc).__name__, str(exc)[:250], flush=True)
                try:
                    post("/api/v1/workers/jobs/" + str(job_id) + "/fail", {
                        "workerId": WORKER_ID,
                        "error": "Native DREAM inference failed. Retry the request.",
                    }, timeout=20)
                except Exception:
                    traceback.print_exc()
            finally:
                STATE["busy"] = False
        except requests.RequestException as exc:
            print("DREAM HOST connection lost; retrying:", str(exc)[:250], flush=True)
            STOP.wait(5)
        except Exception as exc:
            print("Native DREAM worker loop error:", type(exc).__name__, str(exc)[:250], flush=True)
            STOP.wait(5)


def main() -> None:
    require_configuration()
    load_native_model()
    run_jobs()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopping native DREAM inference worker.", flush=True)
    finally:
        STOP.set()
