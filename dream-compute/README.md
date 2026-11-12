# DREAM COMPUTE Worker

This is the provider-independent Python worker agent for DREAM COMPUTE.

## Kaggle setup

1. In the Kaggle notebook's Secrets panel, add:
   - `DREAM_COMPUTE_WORKER_TOKEN` — the worker token configured on the Render service.
   - `HF_TOKEN` — your Hugging Face read token.
2. Select a GPU accelerator if available.
3. Run:

```python
from kaggle_secrets import UserSecretsClient
import os
import subprocess
import sys

secrets = UserSecretsClient()
os.environ["DREAM_COMPUTE_URL"] = "https://dream-compute-live.onrender.com"
os.environ["DREAM_COMPUTE_WORKER_TOKEN"] = secrets.get_secret("DREAM_COMPUTE_WORKER_TOKEN")
os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")

subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "requests", "transformers", "peft", "accelerate"
])

!curl -fsSL https://raw.githubusercontent.com/mjnr0477/dream-ai/dream-compute-platform/dream-compute/worker.py -o /kaggle/working/dream_compute_worker.py
!python /kaggle/working/dream_compute_worker.py
```

The worker loads the configured Qwen base model and DREAM adapter, registers itself, sends a heartbeat every 15 seconds, claims jobs and reports results. It works with CUDA, Apple Metal, or CPU. CPU fallback may be slow and requires enough RAM.

## Other compute machines

Install `torch`, `transformers`, `peft`, `accelerate`, and `requests`, set the same environment variables, and run:

```bash
python worker.py
```

DREAM COMPUTE is the control plane, not a source of free GPU capacity. A machine must stay online and meet the model's RAM/VRAM requirements.
