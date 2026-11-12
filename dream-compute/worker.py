#!/usr/bin/env python3
"""DREAM COMPUTE worker: join a DREAM COMPUTE control plane from a CPU or GPU machine."""

import os
import time
import uuid
import threading
import traceback

import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

HOST = os.environ.get("DREAM_COMPUTE_URL", "").rstrip("/")
TOKEN = os.environ.get("DREAM_COMPUTE_WORKER_TOKEN", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
BASE_MODEL = os.environ.get("DREAM_COMPUTE_MODEL", "Qwen/Qwen3-1.7B")
ADAPTER = os.environ.get("DREAM_COMPUTE_ADAPTER", "msjnr0477/dream-ai-adapter")
WORKER_ID = os.environ.get("DREAM_COMPUTE_WORKER_ID", "dc-" + uuid.uuid4().hex[:18])
WORKER_NAME = os.environ.get("DREAM_COMPUTE_WORKER_NAME", "DREAM COMPUTE worker")
STOP = threading.Event()
STATE = {"busy": False}

if not HOST.startswith("https://") and not HOST.startswith("http://127.0.0.1"):
    raise RuntimeError("DREAM_COMPUTE_URL must be an HTTPS URL (loopback HTTP is allowed for local testing).")
if len(TOKEN) < 32:
    raise RuntimeError("Set DREAM_COMPUTE_WORKER_TOKEN to the worker token from DREAM COMPUTE.")

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else ("Apple Metal" if DEVICE == "mps" else "CPU")
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

def headers():
    return {"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"}

def post(path, payload, timeout=30):
    response = requests.post(HOST + path, json=payload, headers=headers(), timeout=timeout)
    response.raise_for_status()
    return response

print("DREAM COMPUTE worker starting:", WORKER_ID, "device:", GPU_NAME, flush=True)
print("Loading model:", BASE_MODEL, "adapter:", ADAPTER, flush=True)
load_options = {"torch_dtype": DTYPE, "low_cpu_mem_usage": True}
if HF_TOKEN:
    load_options["token"] = HF_TOKEN
tokenizer_source = ADAPTER if ADAPTER and ADAPTER.lower() != "none" else BASE_MODEL
tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, token=HF_TOKEN or None)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_options)
if ADAPTER and ADAPTER.lower() != "none":
    if PeftModel is None:
        raise RuntimeError("Install peft to load the configured DREAM AI adapter.")
    model = PeftModel.from_pretrained(model, ADAPTER, token=HF_TOKEN or None)
model = model.to(DEVICE)
model.eval()
print("Model loaded and ready on", GPU_NAME, flush=True)

METADATA = {
    "device": GPU_NAME,
    "deviceType": DEVICE,
    "model": BASE_MODEL,
    "adapter": ADAPTER or "none",
    "torchVersion": torch.__version__
}

def generate(job):
    messages = job.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Job messages are missing or invalid.")
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=6144)
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    max_tokens = max(1, min(int(job.get("max_tokens", 512)), 1024))
    temperature = max(0.0, min(float(job.get("temperature", 0.7)), 2.0))
    top_p = max(0.01, min(float(job.get("top_p", 0.9)), 1.0))
    options = {
        "max_new_tokens": max_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.05
    }
    if temperature <= 0:
        options["do_sample"] = False
    else:
        options.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    with torch.inference_mode():
        output = model.generate(**inputs, **options)
    prompt_count = int(inputs["input_ids"].shape[-1])
    generated = output[0][prompt_count:]
    content = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not content:
        raise RuntimeError("The model returned an empty response.")
    return content, prompt_count, int(generated.shape[-1])

def register_until_ready():
    payload = {"workerId": WORKER_ID, "name": WORKER_NAME, "metadata": METADATA}
    while not STOP.is_set():
        try:
            response = post("/api/v1/workers/register", payload, timeout=60)
            print("Registered with DREAM COMPUTE:", WORKER_ID, flush=True)
            return
        except requests.RequestException as exc:
            print("Control plane unavailable; retrying registration:", str(exc)[:250], flush=True)
            STOP.wait(5)
    raise RuntimeError("Worker stopped before it registered.")

def heartbeat_loop():
    while not STOP.wait(15):
        try:
            post("/api/v1/workers/heartbeat", {
                "workerId": WORKER_ID, "busy": STATE["busy"], "metadata": METADATA
            }, timeout=20)
        except requests.RequestException as exc:
            print("Heartbeat reconnect:", str(exc)[:200], flush=True)

def main():
    register_until_ready()
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    while not STOP.is_set():
        try:
            response = post("/api/v1/workers/jobs/claim", {"workerId": WORKER_ID}, timeout=35)
            if response.status_code == 204:
                time.sleep(1.5)
                continue
            if response.status_code == 409:
                time.sleep(2)
                continue
            response.raise_for_status()
            job = response.json()["job"]
            STATE["busy"] = True
            print("Processing:", job.get("id"), flush=True)
            try:
                content, prompt_tokens, completion_tokens = generate(job)
                post("/api/v1/workers/jobs/" + job["id"] + "/result", {
                    "workerId": WORKER_ID,
                    "content": content,
                    "promptTokens": prompt_tokens,
                    "completionTokens": completion_tokens
                }, timeout=40)
                print("Completed:", job.get("id"), flush=True)
            except Exception as exc:
                print("Job failed:", str(exc)[:400], flush=True)
                try:
                    post("/api/v1/workers/jobs/" + job["id"] + "/fail", {
                        "workerId": WORKER_ID, "error": str(exc)[:1000]
                    }, timeout=20)
                except Exception:
                    traceback.print_exc()
            finally:
                STATE["busy"] = False
        except requests.RequestException as exc:
            print("Control plane connection lost; retrying:", str(exc)[:250], flush=True)
            time.sleep(5)
        except Exception as exc:
            print("Worker loop failed:", str(exc)[:250], flush=True)
            time.sleep(5)

try:
    main()
except KeyboardInterrupt:
    print("Stopping DREAM COMPUTE worker.", flush=True)
finally:
    STOP.set()
