# DREAM model baseline candidate

Status: provisional engineering baseline, not yet a trained DREAM checkpoint.

## Candidate: Qwen3.5-2B

Official model card: https://huggingface.co/Qwen/Qwen3.5-2B

Reasons to evaluate it first:

- The published model card lists the checkpoint under Apache 2.0.
- It contains 2 billion language-model parameters and publishes the weights and configuration.
- The publisher explicitly lists task-specific fine-tuning and research/development as intended use cases.
- The publisher reports MMLU-Pro scores of 55.3 in non-thinking mode and 66.5 in thinking mode for Qwen3.5-2B. The same table reports 29.7 and 42.3 for Qwen3.5-0.8B. These are publisher-reported results, not independent DREAM evaluations.
- It is a multimodal checkpoint, but the first DREAM training and evaluation stage will be text-only until image input is implemented and tested.

## Comparison candidate

Qwen3.5-0.8B is the smaller fallback for constrained local/edge hardware. Its smaller size does not make it automatically better: the model card's published language benchmarks are notably lower than the 2B checkpoint. We should compare both on the DREAM evaluation set before deciding which checkpoint to ship.

## Ownership boundary

Using an open-weight checkpoint does not mean DREAM created the original foundation model. DREAM ownership begins with our independently curated and licensed training data, reproducible fine-tuning, model adapters/weights, evaluations, inference runtime, versioning and release process. We must retain the upstream license and attribution obligations.

## Compute reality

The Render free web service is for the Node application and cannot be treated as a GPU training host. Fine-tuning and model serving need a separate machine or hosted notebook with suitable RAM/VRAM. The web app and model runtime remain separate services. No paid GPU or external AI API has been provisioned by this decision.

## Gate before selecting a release model

1. Run the same DREAM evaluation cases against both candidate sizes.
2. Record latency, peak memory, task scores and failure modes.
3. Review all outputs manually against the rubrics.
4. Confirm the exact checkpoint license and redistribution terms.
5. Only then choose a checkpoint for DREAM-specific fine-tuning.
