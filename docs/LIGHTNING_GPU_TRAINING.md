# DREAM AI training without Kaggle compute

## Design

The Lightning worker runs the existing DREAM-specific LoRA continuation workflow as a managed T4 GPU job. The original Kaggle notebook remains unchanged as the reference and rollback. The new Python trainer uses the same base model, consent checks, audit gate, source-group isolation, validation comparison, adapter-only optimizer, and candidate-only release rules.

The worker trains only DREAM's LoRA adapter on Qwen/Qwen3-1.7B. It does not train the Qwen foundation model and does not replace the separate native DREAM-from-scratch research model.

Each job downloads the approved dataset and current adapter from the private model repository, resumes only from a checkpoint bound to the exact dataset/audit/parent-adapter hashes, then uploads progress and a separate candidate. The active adapter root is never overwritten or automatically promoted.

## One-time account setup

1. Create or use a Lightning AI account and a Teamspace. Create a programmatic API key and note the Lightning User ID and Teamspace name.
2. In GitHub, open the dream-ai repository settings, then Secrets and variables, Actions, and add these repository secrets:
   - LIGHTNING_API_KEY
   - LIGHTNING_USER_ID
   - LIGHTNING_TEAMSPACE
   - HF_TOKEN (write access to the private msjnr0477/dream-ai-adapter model repository)
3. Move the four files from DREAM's reviewed opt-in export into one directory: train.jsonl, validation.jsonl, manifest.json, and audit-gate.json. Run the migration helper once:

   python -m pip install --upgrade huggingface_hub
   python scripts/publish_dream_reviewed_training_data.py --source /path/to/reviewed-export

The helper checks the audit gate and actual train/validation hashes, requires the model repository to be private, and uploads only to training-data/active. It does not submit a GPU job or change model weights.

4. Merge this branch into main after reviewing it. Then open GitHub Actions, choose DREAM AI Lightning GPU training, and manually run it once. This tests the account, T4 job, data download and candidate upload path.
5. Only after checking the Lightning account's remaining credits and billing settings, set the repository variable DREAM_AI_AUTO_TRAINING_ENABLED to true to enable one daily job. Leave it unset or false to keep the schedule off.

## Important compute and cost boundary

Lightning currently advertises up to 80 free GPU hours to start, depending on its credit terms, and shows additional GPU use as pay-as-you-go. This is limited trial compute, not unlimited or guaranteed permanent free hosting. Prices and free allocations can change. Check the Lightning billing/credits screen before enabling scheduled work.

The workflow keeps the daily schedule disabled unless the explicit repository variable DREAM_AI_AUTO_TRAINING_ENABLED=true is set. Manual dispatch is still possible. Each training update is limited to 240 seconds by default; model loading, evaluation and uploads add runtime beyond that training budget.

## One-time dataset migration

The workflow now checks the private model repository first. If the four reviewed files are already at the configured training-data path, it skips Kaggle completely. If they are missing, it can use the Kaggle dataset API once to transfer the reviewed files into the private model repository. This transfer runs on the GitHub Actions CPU runner; it does not start a Kaggle notebook or use a Kaggle GPU.

For that one-time transfer, add temporary GitHub Actions secrets KAGGLE_USERNAME and KAGGLE_KEY (or KAGGLE_API_TOKEN). The workflow automatically finds the single private dataset owned by that account that contains all four expected files. If more than one matches, set the repository variable DREAM_AI_KAGGLE_DATASET to the exact owner/dataset-slug. The migration checks that the source dataset is private and that the audit gate and train/validation hashes match before uploading. After the first successful migration, remove the temporary Kaggle secrets. Future training jobs use Lightning for GPU compute and the private model repository for data/checkpoints; they do not call Kaggle.

If the reviewed dataset is already available in a local directory instead, the separate upload helper remains available: python scripts/publish_dream_reviewed_training_data.py --source /path/to/reviewed-export

## Safety properties

- The training script checks human-approved consent and all existing audit/provenance requirements.
- Resume is rejected when the exact dataset, audit gate or parent adapter changes.
- Candidate and continuation progress are written to separate folders in the private model repository.
- The live adapter root is unchanged. Evaluation and human review are required before any promotion.
- HF is used only as the private dataset/model/checkpoint store in this design, not as GPU compute.

## Official Lightning documentation

- Jobs SDK: https://lightning.ai/docs/sdk/stable/examples/jobs
- Batch job SDK: https://support.lightning.ai/docs/platform/inference/batch-jobs/sdk/
- Pricing and free GPU allocations: https://support.lightning.ai/pricing/
