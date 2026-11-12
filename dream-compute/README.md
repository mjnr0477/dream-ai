# DREAM COMPUTE Worker

This is the provider-independent Python worker agent for DREAM COMPUTE.

## Kaggle GPU setup

1. In the Kaggle notebook's Secrets panel, add:
   - `DREAM_COMPUTE_WORKER_TOKEN` — the worker token configured on the Render service.
   - `HF_TOKEN` — your Hugging Face read token.
2. Select a GPU accelerator if available.
3. Start a fresh notebook session (the model should only be loaded once), then run this cell:

```python
from kaggle_secrets import UserSecretsClient
import os
import subprocess
import sys

secrets = UserSecretsClient()
os.environ["DREAM_COMPUTE_URL"] = "https://dream-compute-live.onrender.com"
os.environ["DREAM_COMPUTE_WORKER_TOKEN"] = secrets.get_secret("DREAM_COMPUTE_WORKER_TOKEN")
os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")

# Qwen runs in standard PyTorch FP16 on a T4; torchao is not required.
# Remove a stale optional torchao package if it conflicts with the installed PyTorch.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "requests", "transformers", "peft", "accelerate"
])

worker_url = "https://raw.githubusercontent.com/mjnr0477/dream-ai/dream-compute-platform/dream-compute/worker.py"
subprocess.check_call(["curl", "-fsSL", worker_url, "-o", "/kaggle/working/dream_compute_worker.py"])
subprocess.check_call([sys.executable, "/kaggle/working/dream_compute_worker.py"])
```

The worker loads the configured Qwen base model and DREAM adapter, registers itself, sends a heartbeat every 15 seconds, claims jobs and reports results. It works with CUDA, Apple Metal, or CPU. CPU fallback may be slow and requires enough RAM.

If model loading fails, capture the complete first traceback. Do not repeatedly upgrade PyTorch or install a random torchao build: torchao must match the exact PyTorch/CUDA build, and it is unnecessary for the standard FP16 path above.

## Other compute machines

Install compatible `torch`, `transformers`, `peft`, `accelerate`, and `requests`, set the same environment variables, and run:

```bash
python worker.py
```

DREAM COMPUTE is the control plane, not a source of free GPU capacity. A machine must stay online and meet the model's RAM/VRAM requirements.
