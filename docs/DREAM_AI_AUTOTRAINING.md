# DREAM AI autonomous adapter training — retired

The scheduled adapter-training workflow has been retired. It no longer launches third-party GPU jobs, uploads training data to an external model repository, or modifies adapter candidates.

No training is currently being performed by this workflow. The last preflight stopped before training because the dedicated DREAM AI data source and required compute credentials were not available.

Do not re-enable this schedule by adding provider tokens. A future training implementation must use the native DREAM architecture on a verified operator-controlled GPU, read only human-approved records, enforce one total session budget of at most five minutes, preserve optimizer/RNG checkpoints on persistent storage, and keep all outputs candidate-only.

The previous adapter and any existing checkpoints remain untouched. A successful GitHub Actions workflow or synthetic CI smoke test is not evidence that DREAM has trained or improved.
