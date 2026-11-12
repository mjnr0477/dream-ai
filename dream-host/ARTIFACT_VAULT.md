# DREAM AI Artifact Vault

The native from-scratch model can publish its checkpoint bundle to DREAM HOST without using Hugging Face as a model or artifact repository.

## Storage design

The vault is content-addressed. Uploads stream into a temporary file, are SHA-256 hashed, and are then moved into an object directory. Metadata is stored separately. The API requires a dedicated artifact token, enforces a configurable upload limit (512 MiB default, 1 GiB maximum), rejects unsafe names and never exposes artifacts publicly.

Required DREAM HOST environment variables:

- `DREAM_HOST_ARTIFACT_TOKEN`: a new random secret of at least 32 characters, distinct from every chat and notebook token.
- `DREAM_HOST_ARTIFACTS_DIR`: an absolute path on a genuinely persistent mounted filesystem.
- Optional `DREAM_HOST_ARTIFACT_MAX_BYTES`: upload limit in bytes (default 536870912; maximum 1073741824).

**Render Free limitation:** the current Render Free service has an ephemeral filesystem and does not provide a persistent disk. Do not set `DREAM_HOST_ARTIFACTS_DIR` to an ordinary Render path and assume it is durable. The vault stays disabled until a persistent disk or separately operated persistent storage mount is available. A restart or redeploy on ephemeral storage can erase model artifacts.

## Publish from Kaggle

Add a private Kaggle Secret named `DREAM_ARTIFACT_TOKEN` containing the dedicated artifact token. The native from-scratch Kaggle notebook includes an optional final upload cell that reads this secret directly. It does not expose the chat worker or notebook worker tokens to notebook code.

After the native training notebook creates `/kaggle/working/dream-training-artifacts.zip`, run the optional final cell. The cell uploads the bundle and checks that the returned SHA-256 matches the local file. If the vault is not enabled, the upload will return an HTTP error. Keep the ZIP private and download it from the Kaggle Output panel until persistent storage is configured.

From a repository checkout, the same upload can be performed with:

```python
!python model/publish_artifact.py /kaggle/working/dream-training-artifacts.zip
```

The publisher uses `DREAM_NOTEBOOK_ARTIFACT_TOKEN` from the process environment, calculates the local SHA-256, and refuses to report success unless the vault returns the same digest and byte count.

## Authenticated API

All endpoints require `Authorization: Bearer <DREAM_HOST_ARTIFACT_TOKEN>`.

- `PUT /api/v1/notebook/artifacts/:name` — stream an artifact into the vault.
- `GET /api/v1/notebook/artifacts` — list artifact metadata.
- `GET /api/v1/notebook/artifacts/:sha256` — download a content-addressed artifact.

## Connect a reviewed checkpoint to native inference

The vault is storage, not an inference host. After a checkpoint has been reviewed and approved, place the selected checkpoint on the native model runtime's persistent filesystem and set `DREAM_CHECKPOINT` to its path. The runtime already loads that checkpoint through `model/serve.py` and exposes its SHA-256 and training metadata at `/health`.

Do not automatically promote a training output. Run the checkpoint audit, held-out evaluations and release review first. A technically valid checkpoint is not proof of a capable or safe assistant.
