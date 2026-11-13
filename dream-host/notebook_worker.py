#!/usr/bin/env python3
"""Operator-owned Python cell worker for DREAM NOTEBOOK.

This worker executes trusted notebook code in per-notebook child interpreters.
It is not a hardened multi-tenant sandbox: run it only on a machine controlled
by the notebook owner, with access restricted to the owner's private host token.
"""
from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path

import requests

HOST = os.environ.get("DREAM_HOST_URL", "https://dream-host-live.onrender.com").rstrip("/")
TOKEN = os.environ.get("DREAM_HOST_NOTEBOOK_WORKER_TOKEN", "").strip()
WORKER_ID = os.environ.get("DREAM_HOST_NOTEBOOK_WORKER_ID", "notebook-" + uuid.uuid4().hex[:16])
WORKER_NAME = os.environ.get("DREAM_HOST_NOTEBOOK_WORKER_NAME", "DREAM NOTEBOOK compute worker")
TRUSTED_MODE = os.environ.get("DREAM_HOST_NOTEBOOK_TRUSTED_MODE", "").strip() == "1"
POLL_SECONDS = max(0.5, min(float(os.environ.get("DREAM_HOST_NOTEBOOK_POLL_SECONDS", "1.5")), 10.0))
MAX_EXECUTION_SECONDS = max(30, min(int(os.environ.get("DREAM_HOST_NOTEBOOK_MAX_EXECUTION_SECONDS", "1800")), 1800))
MAX_NOTEBOOK_KERNELS = max(1, min(int(os.environ.get("DREAM_HOST_NOTEBOOK_MAX_KERNELS", "4")), 16))
KERNEL_IDLE_SECONDS = max(60, min(int(os.environ.get("DREAM_HOST_NOTEBOOK_KERNEL_IDLE_SECONDS", "1800")), 86400))
OUTPUT_LIMIT = 120_000
STOP = threading.Event()
KERNELS: dict[str, dict] = {}
KERNEL_LOCK = threading.Lock()

if not HOST.startswith("https://") and not HOST.startswith("http://127.0.0.1"):
    raise RuntimeError("DREAM_HOST_URL must use HTTPS (loopback HTTP is allowed for local tests).")
if len(TOKEN) < 32:
    raise RuntimeError("Set DREAM_HOST_NOTEBOOK_WORKER_TOKEN to the private notebook worker token configured on DREAM HOST.")
if not TRUSTED_MODE:
    raise RuntimeError(
        "Notebook cells execute trusted Python code. Set DREAM_HOST_NOTEBOOK_TRUSTED_MODE=1 "
        "only on a machine controlled by the notebook owner."
    )


KERNEL_BOOTSTRAP = r'''
import ast
import contextlib
import io
import json
import sys
import traceback

class LimitedBuffer(io.StringIO):
    def __init__(self, limit=120000):
        super().__init__()
        self.limit = limit
        self.used = 0
    def write(self, value):
        text = str(value)
        remaining = max(0, self.limit - self.used)
        if remaining:
            clipped = text[:remaining]
            self.used += len(clipped)
            return super().write(clipped)
        return len(text)

namespace = {"__name__": "__main__"}
execution_count = 0
for line in sys.stdin:
    try:
        request = json.loads(line)
        code = request.get("code", "")
        execution_count += 1
        output = LimitedBuffer()
        errors = LimitedBuffer()
        display = ""
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                tree = ast.parse(code, filename="<dream-notebook-cell>", mode="exec")
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last_expression = tree.body.pop()
                    if tree.body:
                        module = ast.Module(body=tree.body, type_ignores=[])
                        exec(compile(ast.fix_missing_locations(module), "<dream-notebook-cell>", "exec"), namespace, namespace)
                    value = eval(compile(ast.fix_missing_locations(ast.Expression(last_expression.value)), "<dream-notebook-cell>", "eval"), namespace, namespace)
                    if value is not None:
                        display = repr(value)
                else:
                    exec(compile(ast.fix_missing_locations(tree), "<dream-notebook-cell>", "exec"), namespace, namespace)
        except BaseException:
            errors.write(traceback.format_exc())
        response = {
            "stdout": output.getvalue()[-120000:],
            "stderr": errors.getvalue()[-120000:],
            "display": display[:120000],
            "executionCount": execution_count
        }
    except BaseException:
        response = {
            "stdout": "",
            "stderr": traceback.format_exc()[-120000:],
            "display": "",
            "executionCount": execution_count
        }
    sys.__stdout__.write(json.dumps(response, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()
'''


def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"}


def post(path: str, payload: dict, timeout: int = 30) -> requests.Response:
    response = requests.post(HOST + path, json=payload, headers=auth_headers(), timeout=timeout)
    response.raise_for_status()
    return response


def detect_compute() -> dict[str, str]:
    metadata = {
        "role": "notebook",
        "device": "CPU",
        "deviceType": "cpu",
        "python": sys.version.split()[0],
        "workerVersion": "0.1.0"
    }
    try:
        import torch
        metadata["torch"] = torch.__version__
        if torch.cuda.is_available():
            metadata["device"] = torch.cuda.get_device_name(0)
            metadata["deviceType"] = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            metadata["device"] = "Apple Metal"
            metadata["deviceType"] = "mps"
    except Exception:
        pass
    return metadata


def child_environment(workspace: Path) -> dict[str, str]:
    # Deliberately do not inherit DREAM_HOST_* tokens, HF_TOKEN, or other
    # operator environment variables into notebook code.
    allowed = (
        "PATH", "LANG", "LC_ALL", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH",
        "CUDA_HOME", "CUDA_PATH", "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "PYTHONPATH", "CONDA_PREFIX", "VIRTUAL_ENV"
    )
    environment = {key: os.environ[key] for key in allowed if key in os.environ}
    environment["DREAM_HOST_URL"] = HOST
    # Pass only dedicated, explicitly configured secrets to trusted notebook
    # kernels. Never pass DREAM_HOST_NOTEBOOK_WORKER_TOKEN or other HOST keys.
    for key in ("DREAM_NOTEBOOK_HF_TOKEN", "DREAM_NOTEBOOK_ARTIFACT_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            environment[key] = value
    environment["HOME"] = str(workspace)
    environment["TMPDIR"] = str(workspace)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def start_kernel(notebook_id: str) -> dict:
    folder = tempfile.mkdtemp(prefix="dream-notebook-")
    workspace = Path(folder)
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", KERNEL_BOOTSTRAP],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=workspace,
        env=child_environment(workspace),
        text=True,
        encoding="utf-8",
        bufsize=1,
        start_new_session=True
    )
    return {"process": process, "workspace": workspace, "last_used": time.time(), "execution_count": 0}


def cleanup_idle_kernels() -> None:
    now = time.time()
    with KERNEL_LOCK:
        expired = [
            (notebook_id, kernel)
            for notebook_id, kernel in KERNELS.items()
            if now - kernel["last_used"] > KERNEL_IDLE_SECONDS
        ]
        for notebook_id, kernel in expired:
            KERNELS.pop(notebook_id, None)
            stop_kernel(kernel)


def get_kernel(notebook_id: str) -> dict:
    cleanup_idle_kernels()
    with KERNEL_LOCK:
        kernel = KERNELS.get(notebook_id)
        if kernel and kernel["process"].poll() is None:
            kernel["last_used"] = time.time()
            return kernel
        if kernel:
            stop_kernel(kernel)
        if len(KERNELS) >= MAX_NOTEBOOK_KERNELS:
            raise RuntimeError(
                f"This worker already has {MAX_NOTEBOOK_KERNELS} active notebook kernels. "
                "Wait for an idle kernel to expire or restart the worker."
            )
        kernel = start_kernel(notebook_id)
        KERNELS[notebook_id] = kernel
        return kernel


def stop_kernel(kernel: dict) -> None:
    process = kernel.get("process")
    if process and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
    try:
        import shutil
        shutil.rmtree(kernel["workspace"], ignore_errors=True)
    except Exception:
        pass


def execute_cell(job: dict) -> dict:
    notebook_id = str(job.get("notebookId", "")).strip()
    code = job.get("code")
    if not notebook_id or not isinstance(code, str) or not code.strip():
        raise ValueError("Notebook job is missing its notebook ID or Python source.")
    if len(code) > 20_000:
        raise ValueError("Notebook cell exceeds the 20,000 character worker limit.")

    kernel = get_kernel(notebook_id)
    process = kernel["process"]
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("Notebook kernel pipes are unavailable.")
    process.stdin.write(json.dumps({"code": code}, ensure_ascii=False) + "\n")
    process.stdin.flush()

    line_queue: queue.Queue[str] = queue.Queue(maxsize=1)
    reader = threading.Thread(target=lambda: line_queue.put(process.stdout.readline()), daemon=True)
    reader.start()
    timeout_seconds = max(30, min(MAX_EXECUTION_SECONDS, int(job.get("timeoutMs", 180_000) / 1000)))
    try:
        line = line_queue.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        stop_kernel(kernel)
        with KERNEL_LOCK:
            KERNELS.pop(notebook_id, None)
        raise TimeoutError("Notebook cell exceeded the configured execution limit.") from exc
    if not line:
        stop_kernel(kernel)
        with KERNEL_LOCK:
            KERNELS.pop(notebook_id, None)
        raise RuntimeError("Notebook kernel exited before returning a result.")

    result = json.loads(line)
    for key in ("stdout", "stderr", "display"):
        result[key] = str(result.get(key, ""))[:OUTPUT_LIMIT]
    kernel["execution_count"] = int(result.get("executionCount") or (kernel["execution_count"] + 1))
    kernel["last_used"] = time.time()
    return result


def register_until_ready() -> None:
    payload = {
        "workerId": WORKER_ID,
        "name": WORKER_NAME,
        "metadata": detect_compute()
    }
    while not STOP.is_set():
        try:
            response = post("/api/v1/notebook/workers/register", payload, timeout=30)
            print("Registered with DREAM NOTEBOOK:", WORKER_ID, payload["metadata"], flush=True)
            return
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in (401, 403):
                raise RuntimeError("DREAM HOST rejected the notebook worker token. Check the worker secret.") from exc
            print("Notebook worker registration rejected; retrying:", str(exc)[:250], flush=True)
            STOP.wait(5)
        except requests.RequestException as exc:
            print("DREAM HOST unavailable; retrying notebook worker registration:", str(exc)[:250], flush=True)
            STOP.wait(5)
    raise RuntimeError("Notebook worker stopped before registering.")


def heartbeat_loop() -> None:
    while not STOP.wait(15):
        try:
            response = post("/api/v1/notebook/workers/heartbeat", {
                "workerId": WORKER_ID,
                "busy": STATE["busy"]
            }, timeout=15)
            response.raise_for_status()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 409:
                try:
                    register_until_ready()
                except Exception as recovery_error:
                    print("Notebook worker re-registration failed:", str(recovery_error)[:250], flush=True)
            else:
                print("Notebook heartbeat rejected:", str(exc)[:200], flush=True)
        except requests.RequestException as exc:
            print("Notebook heartbeat connection lost:", str(exc)[:200], flush=True)


STATE = {"busy": False}


def main() -> None:
    register_until_ready()
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    print("DREAM NOTEBOOK worker ready. Keep this process running while notebooks execute.", flush=True)
    while not STOP.is_set():
        cleanup_idle_kernels()
        try:
            response = post("/api/v1/notebook/workers/jobs/claim", {"workerId": WORKER_ID}, timeout=25)
            if response.status_code == 204:
                STOP.wait(POLL_SECONDS)
                continue
            response.raise_for_status()
            job = response.json().get("job")
            if not isinstance(job, dict):
                raise RuntimeError("DREAM HOST returned an invalid notebook job.")
            STATE["busy"] = True
            print("Running notebook cell:", job.get("notebookId"), job.get("cellId"), flush=True)
            try:
                result = execute_cell(job)
                post("/api/v1/notebook/workers/jobs/" + job["id"] + "/result", {
                    "workerId": WORKER_ID,
                    **result
                }, timeout=30).raise_for_status()
            except Exception as exc:
                print("Notebook cell worker error:", str(exc)[:300], flush=True)
                try:
                    post("/api/v1/notebook/workers/jobs/" + job["id"] + "/fail", {
                        "workerId": WORKER_ID,
                        "error": str(exc)[:1800]
                    }, timeout=20).raise_for_status()
                except Exception:
                    traceback.print_exc()
            finally:
                STATE["busy"] = False
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 409:
                try:
                    register_until_ready()
                except Exception as recovery_error:
                    print("Notebook worker recovery error:", str(recovery_error)[:250], flush=True)
            else:
                print("Notebook claim request rejected:", str(exc)[:250], flush=True)
                STOP.wait(3)
        except requests.RequestException as exc:
            print("DREAM HOST connection lost; retrying:", str(exc)[:250], flush=True)
            STOP.wait(5)
        except Exception as exc:
            print("Notebook worker loop error:", str(exc)[:300], flush=True)
            STOP.wait(3)


def shutdown(_signum=None, _frame=None) -> None:
    STOP.set()
    with KERNEL_LOCK:
        for kernel in list(KERNELS.values()):
            stop_kernel(kernel)
        KERNELS.clear()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    main()
