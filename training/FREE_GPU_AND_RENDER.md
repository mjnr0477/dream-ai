# DREAM compute and deployment path

Checked 6 November 2026 against provider documentation.

## Recommended no-cost training machine: Kaggle

Kaggle's official GPU guidance says free NVIDIA Tesla P100 access is available with a weekly GPU quota of 30 hours or sometimes higher depending on demand and resources. This is a quota, not a permanently allocated GPU. Kaggle notebooks are therefore suitable for bounded training runs, not for an always-on model server.

For the existing Kaggle kernel `mathiassimiyu/notebook21501a7fa8`, the repository helper `scripts/kaggle-run-dream-training.sh` pulls its metadata (including the private dataset attachment), replaces the code with the current DREAM training notebook, enables GPU and Internet, pushes the run, waits for completion and downloads the artifact ZIP. It uses the Kaggle CLI's local credentials; the token is never committed or sent to the repository. It refuses to run if the existing kernel does not have exactly one dataset source attached.

Official references:
- https://www.kaggle.com/docs/efficient-gpu-usage
- https://www.kaggle.com/getting-started/524433
- https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md

The Kaggle CLI can upload and run a notebook with `kaggle kernels push -p <folder>`; `kaggle kernels output <owner/kernel> -p <folder>` retrieves generated artifacts. Both calls require the user's own Kaggle API credentials. Never commit `kaggle.json`, API tokens or credentials to this repository.

## Alternative: Lightning AI starter credits

Lightning AI currently advertises up to 30 free starter credits, with up to about 80 GPU hours on interruptible machines depending on machine/pricing. Its pricing page says the free account starts with 5 credits and adding a card unlocks 25 more. It also restarts free Studios every four hours. This is an optional alternative, not a guaranteed no-card 80-hour offer.

Official reference: https://lightning.ai/pricing

## Not suitable for sustained training

- Hugging Face ZeroGPU is shared, quota-limited GPU time intended for interactive Spaces; the free user allowance is minutes per day, not a long-running trainer. It is useful for demos/inference experiments, not DREAM's training run.
- Google Colab free GPU access is not guaranteed and usage limits fluctuate.
- Render free web services have 0.1 CPU and 512 MB RAM, spin down after 15 idle minutes, and do not provide GPU hardware. Their local filesystem is ephemeral. Render should host the web application only, not model training.

Official references:
- https://huggingface.co/docs/hub/spaces-zerogpu
- https://research.google.com/colaboratory/intl/en-GB/faq.html
- https://render.com/docs/free
- https://render.com/docs/compute-plans

## Required end-to-end sequence

1. Train DREAM's native checkpoint in a Kaggle GPU notebook using the approved corpus.
2. Save a Kaggle notebook version and verify that `dream-training-artifacts.zip` actually contains a checkpoint, its audit/evaluation reports and the exact corpus manifest/test split.
3. Run the verified self-improvement notebook against independently reviewed, explicitly approved SFT records. Candidate checkpoints are promoted only after audit and evaluation gates pass.
4. Store the resulting checkpoint in a private artifact location with its SHA-256 report.
5. Deploy the native inference runtime on a host with sufficient CPU/RAM and persistent model-file access. Render's free web process should remain the Node chat UI/API and call the model runtime over HTTPS using the existing `DREAM_MODEL_URL`, `DREAM_MODEL_NAME` and `DREAM_MODEL_TOKEN` settings.
6. Confirm the Render `/api/health` and `/api/capabilities` endpoints report the real native checkpoint before inviting users to chat.

## Important model limitation

DREAM currently implements a small transformer initialized from random weights. The Kaggle pipeline's pretraining and instruction-tuning stages are real weight updates, but a small model trained from a small corpus will not become a general-purpose frontier assistant just by repeating those updates. Repeated self-training on unverified model-generated answers risks reinforcing errors. Use independently checked, rights-cleared examples and held-out evaluation; do not turn on blind self-training.
