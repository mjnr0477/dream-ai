# DREAM verified self-improvement

DREAM can run repeatable instruction-tuning cycles on its own native checkpoint. The controller is an orchestrator around DREAM's existing PyTorch trainer; it does not create compute, invent trusted labels, or make the model a frontier assistant automatically.

## What the cycle does

1. Starts from a trained DREAM checkpoint.
2. Accepts only SFT JSONL records that pass the existing rights, privacy, quality and explicit-approval checks in `model/sft.py`.
3. Fine-tunes the native DREAM model using assistant-response-only loss.
4. Audits the candidate checkpoint for exact tensor structure, finite weights and positive training-step metadata.
5. Evaluates the base and candidate on the exact same held-out pretraining test file and deterministic evaluation settings.
6. Runs the authored DREAM task diagnostic benchmark on the candidate, and on the base too when the base is already instruction-tuned.
7. Promotes the candidate only if the technical release preflight passes, held-out bits-per-byte does not regress beyond the configured tolerance, and (when a task baseline exists) exact-match task score does not regress.
8. Keeps the previous active checkpoint as a rollback copy. A rejected candidate never replaces the active checkpoint.

The controller never trains on the held-out test split. The pretraining manifest and exact test-file hash are checked by the SFT trainer. Continued SFT must retain the same pretraining held-out test corpus.

## Run one improvement cycle

From the repository root, with PyTorch installed and a suitable GPU available:

```sh
python model/self_improve.py \
  --checkpoint checkpoints/dream-bpe-instruct.pt \
  --train-data data/sft-train.jsonl \
  --validation-data data/sft-validation.jsonl \
  --pretraining-manifest data/prepared/manifest.json \
  --pretraining-test-data data/prepared/test.txt \
  --out checkpoints/dream-active.pt \
  --steps 100 \
  --batch-size 2 \
  --grad-accum-steps 4
```

For the first instruction-tuning cycle, `--checkpoint` may be the audited pretraining checkpoint (for example `checkpoints/dream-bpe.pt`). Later cycles may use the promoted instruction checkpoint as the next `--checkpoint`. Each cycle should use new, independently reviewed examples where possible; reusing the same examples repeatedly risks overfitting.

The SFT files must contain actual human-reviewed records with valid HTTPS source and license URLs, an accepted rights basis, strict source dates, `rights_confirmed: true`, `privacy_reviewed: true`, `quality_reviewed: true` and `approval_granted: true`. The SFT trainer rejects duplicate content/IDs and source-group leakage against validation and the pretraining held-out test groups. Never mark records approved unless the checks were genuinely completed.

## Outputs

Given `--out checkpoints/dream-active.pt`, a successful cycle writes:

- `checkpoints/dream-active.pt`: promoted active checkpoint;
- `checkpoints/dream-active.previous.pt`: exact prior checkpoint for rollback;
- `checkpoints/dream-active.pt.self-improve.json`: hashes, evaluation metrics and promotion decision.

If any training, audit, evaluation or gate step fails, the active checkpoint is left unchanged. The controller exits non-zero rather than reporting success.

## What "self-improvement" means here

This is controlled, data-driven continual instruction tuning. DREAM can repeatedly train its own weights using approved examples and test candidate versions without overwriting a known-good checkpoint. It is not unrestricted autonomous learning:

- Model-generated answers are not trusted training labels. They must be independently checked and approved before being used as SFT targets.
- The starter exact-match benchmark and next-token loss are diagnostics, not proof of reasoning, factuality, safety, general intelligence or usefulness.
- Passing the automated gates is not production approval. Expand the benchmark, add human review, and assess safety and privacy before broad release.
- GPU time, memory, storage and network access must be supplied by the machine or notebook running the job. The Render free web service is not a model-training GPU.
- Tool use and external actions remain separate capabilities; model training does not itself grant access to accounts, websites or devices.

## Tests

Run:

```sh
python -m unittest model.test_self_improve -v
python -m unittest discover -s model -p 'test_*.py' -v
```
