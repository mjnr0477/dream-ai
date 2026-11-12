import os
import time
import uuid
import threading
import traceback

import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

HOST = os.environ.get("DREAM_HOST_URL", "").rstrip("/")
WORKER_TOKEN = os.environ.get("DREAM_HOST_WORKER_TOKEN", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
BASE_MODEL = os.environ.get("DREAM_MODEL_BASE", "Qwen/Qwen3-1.7B")
ADAPTER = os.environ.get("DREAM_MODEL_ADAPTER", "msjnr0477/dream-ai-adapter")
WORKER_ID = os.environ.get("DREAM_HOST_WORKER_ID", "kaggle-" + uuid.uuid4().hex[:16])
WORKER_NAME = "Kaggle GPU worker"
STOP = threading.Event()
STATE = {"busy": False}

if not HOST.startswith("https://"):
    raise RuntimeError("DREAM_HOST_URL must be the HTTPS URL of your Render DREAM HOST service.")
if len(WORKER_TOKEN) < 32:
    raise RuntimeError("Set the DREAM_HOST_WORKER_TOKEN Kaggle secret to the worker token from Render.")
if not HF_TOKEN:
    raise RuntimeError("Set the HF_TOKEN Kaggle secret before starting the worker.")
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU is available in this Kaggle session.")

def auth_headers():
    return {"Authorization": "Bearer " + WORKER_TOKEN, "Content-Type": "application/json"}

def post(path, payload, timeout=30):
    return requests.post(HOST + path, json=payload, headers=auth_headers(), timeout=timeout)

print("Loading DREAM AI model on", torch.cuda.get_device_name(0), flush=True)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER, token=HF_TOKEN)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    token=HF_TOKEN
)
model = PeftModel.from_pretrained(base, ADAPTER, token=HF_TOKEN, torch_device="cpu")
model = model.to("cuda")
model.eval()
print("Model ready:", BASE_MODEL, "+", ADAPTER, flush=True)

def heartbeat_loop():
    while not STOP.wait(15):
        try:
            response = post("/api/v1/workers/heartbeat", {
                "workerId": WORKER_ID,
                "name": WORKER_NAME,
                "busy": STATE["busy"],
                "metadata": {
                    "device": torch.cuda.get_device_name(0),
                    "model": BASE_MODEL,
                    "adapter": ADAPTER
                }
            }, timeout=20)
            response.raise_for_status()
        except Exception as exc:
            print("Heartbeat reconnect:", str(exc)[:250], flush=True)

def generate(job):
    messages = job.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("The queued job did not contain valid messages.")
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=6144)
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    max_new_tokens = max(1, min(int(job.get("max_tokens", 512)), 1024))
    temperature = max(0.0, min(float(job.get("temperature", 0.7)), 2.0))
    top_p = max(0.01, min(float(job.get("top_p", 0.9)), 1.0))
    options = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.05
    }
    if temperature <= 0:
        options["do_sample"] = False
    else:
        options.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    with torch.inference_mode():
        output = model.generate(**inputs, **options)
    prompt_length = inputs["input_ids"].shape[-1]
    generated = output[0][prompt_length:]
    content = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not content:
        raise RuntimeError("The model returned an empty answer.")
    return content, int(prompt_length), int(generated.shape[-1])

def register_until_ready():
    payload = {
        "workerId": WORKER_ID,
        "name": WORKER_NAME,
        "metadata": {
            "device": torch.cuda.get_device_name(0),
            "model": BASE_MODEL,
            "adapter": ADAPTER
        }
    }
    while not STOP.is_set():
        try:
            print("Registering with DREAM HOST:", HOST, flush=True)
            registration = post("/api/v1/workers/register", payload, timeout=90)
            registration.raise_for_status()
            print("Registered. Worker ID:", WORKER_ID, flush=True)
            return
        except requests.RequestException as exc:
            print("DREAM HOST is waking or temporarily unavailable; retrying registration:", str(exc)[:300], flush=True)
            STOP.wait(5)
    raise RuntimeError("Worker stopped before registration completed.")

def main():
    register_until_ready()

    heartbeat = threading.Thread(target=heartbeat_loop, daemon=True)
    heartbeat.start()

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
            print("Processing job:", job.get("id"), flush=True)
            try:
                content, prompt_tokens, completion_tokens = generate(job)
                result = post("/api/v1/workers/jobs/" + job["id"] + "/result", {
                    "workerId": WORKER_ID,
                    "content": content,
                    "promptTokens": prompt_tokens,
                    "completionTokens": completion_tokens
                })
                result.raise_for_status()
                print("Completed job:", job["id"], flush=True)
            except Exception as exc:
                print("Job failed:", str(exc)[:500], flush=True)
                try:
                    post("/api/v1/workers/jobs/" + job["id"] + "/fail", {
                        "workerId": WORKER_ID,
                        "error": str(exc)[:1000]
                    }, timeout=20)
                except Exception:
                    traceback.print_exc()
            finally:
                STATE["busy"] = False
        except requests.RequestException as exc:
            print("DREAM HOST connection lost; retrying:", str(exc)[:300], flush=True)
            time.sleep(5)
        except Exception as exc:
            print("Worker loop error:", str(exc)[:300], flush=True)
            time.sleep(5)

try:
    main()
except KeyboardInterrupt:
    print("Stopping DREAM HOST worker.", flush=True)
finally:
    STOP.set()
