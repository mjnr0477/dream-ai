# DREAM AI Permanent Compute Architecture

**Status:** architecture decision and implementation contract; not a production deployment.
**Goal:** remove dependence on temporary notebook sessions and single-host GPU services while keeping DREAM's model, data, identity, and serving interface under DREAM's control.
**Non-goal:** claiming unlimited free compute exists. GPU time, durable storage, networking, and operations must be supplied by actual contributors, grants, or owned hardware.

## Executive decision

DREAM AI should own a provider-independent control plane and a portable model-serving contract. Compute should be supplied by a federation of independently operated worker nodes, not by a notebook session or one free-tier vendor.

The durable target has five separately replaceable layers:

1. **Control plane (DREAM-owned):** API, authentication/authorization for workers, job queue, worker registry, health and capacity scheduler, audit log, release gates.
2. **Inference workers (contributor-operated):** machines running a pinned DREAM model artifact behind the existing OpenAI-compatible chat interface. A peer network such as OpenHydra may be evaluated as a transport/routing adapter, not treated as guaranteed hosted capacity.
3. **Training workers (contributor-operated):** reproducible workers that fetch an approved training manifest, verify dataset/checkpoint hashes, train only an explicitly assigned task, and submit signed results. Hivemind is a candidate library for decentralized parameter averaging, but it is not a drop-in trainer for DREAM's existing loop; a compatibility prototype and convergence tests are mandatory.
4. **Artifact store (DREAM-controlled):** versioned object storage for datasets, tokenizer, checkpoints, optimizer state, evaluation reports and signed release manifests. At least three independently administered replicas are required before calling the store resilient. Git repositories are for code and small metadata, not multi-gigabyte model weights.
5. **Recovery and governance:** automatic worker replacement, checkpoint rollback, reproducible builds, model evaluation, contributor revocation, incident response, and a human-approved production release.

## What current research actually establishes

### OpenHydra

Project documentation describes a desktop/agent that connects an AI engine already running on a user's computer (Ollama, vLLM, LM Studio, llama.cpp, Exo, etc.) to a peer-to-peer network, with an OpenAI-compatible local endpoint. Its sharing model is reciprocity: users contribute their own machines and receive priority in return.

This is useful as an optional peer transport and potential inference worker source. It is **not** a hosted GPU service that DREAM can simply call from Render without any peers, and its public availability does not guarantee that enough peers will be online, capable, or serving DREAM's chosen model. Every peer must be treated as untrusted until identity, model hash, privacy, rate limits, output checks and abuse controls are implemented.

References:
- https://openhydra.co/
- https://openhydra.co/docs

### Hivemind

Hivemind is an open-source PyTorch library for decentralized deep learning. It provides a DHT, decentralized optimizer and parameter averaging for peers that run compatible training code. Its quickstart demonstrates multiple peers, including a same-machine example; the project does not provide DREAM with a pre-existing pool of volunteer GPUs.

It is a candidate for an experimental distributed-training backend. DREAM must first prove that its own transformer, optimizer, checkpoint format, data partitioning and failure recovery work correctly under peer churn. Hivemind must not be described as a production training fleet or guaranteed capacity.

References:
- https://github.com/learning-at-home/hivemind
- https://github.com/learning-at-home/hivemind/blob/master/docs/user/quickstart.md

### Cloudflare Workers AI

Cloudflare documents a free allocation of 10,000 Neurons per day, with paid usage above the allocation on the Workers Paid plan. This is a bounded inference allowance, not persistent GPU ownership, not a training cluster, and not an unlimited free production guarantee.

DREAM may keep it as an optional, quota-aware fallback only if the operator explicitly accepts the service's terms and limits. It must never be the only production runtime.

Reference:
- https://developers.cloudflare.com/workers-ai/platform/pricing/

## Non-negotiable production architecture

### Worker contract

Each worker must expose or implement:
- authenticated worker identity and revocable credentials;
- advertised hardware, memory, model hash, software/runtime version and available capacity;
- a bounded lease with heartbeat, cancellation and deadline;
- signed job/result metadata;
- model artifact hash verification before loading;
- request and output size limits, per-worker concurrency limits and timeouts;
- no access to the control-plane database or other workers' secrets;
- explicit data-handling policy; private user prompts must not be routed to unknown peers without informed authorization.

A worker is **eligible** only after a real inference canary succeeds. Merely registering, resolving DNS, opening a TCP connection, or returning a health page is not readiness.

### Scheduler and failover

- Route only to workers that passed the canary for the exact model digest and API contract.
- Use bounded retries with idempotency keys; never duplicate a non-idempotent training job blindly.
- Maintain a local queue and a clear unavailable response when no eligible capacity exists.
- Track per-worker latency, failure rate, memory pressure, quota and last successful canary.
- Quarantine workers that return malformed output, wrong model hashes, suspicious results or repeated timeouts.
- Keep at least two independently operated inference workers in different failure domains before promising high availability. Three is the target.

### Training and checkpoint integrity

- Training data is versioned and licensed; each run records dataset manifest, tokenizer digest, code commit, configuration, random seed, worker identities and completed steps.
- Checkpoints are written atomically, hashed, validated by a restore test and replicated to three independent storage locations.
- Candidate checkpoints are evaluated against held-out tests and safety/regression suites.
- Training updates never overwrite the last approved production checkpoint.
- A failed or malicious worker cannot directly publish a production model.
- Release requires reproducible evaluation, human review and a signed immutable manifest.

### Storage and control-plane persistence

Do not store model weights in ephemeral container filesystems or assume Git LFS/free hosting is durable model storage. The production storage contract requires:
- object storage with documented retention and export;
- three replicas in independently controlled accounts/providers or physical sites;
- periodic checksum audits and restore drills;
- encrypted backups and separately held recovery credentials;
- exportable metadata and a documented migration path to another S3-compatible store.

A free quota can help bootstrap but is not a permanence guarantee. Storage ownership and ongoing capacity must be confirmed before launch.

## Required milestones and acceptance gates

| Gate | Deliverable | Pass condition |
|---|---|---|
| A. Control-plane correctness | Worker/job/artifact schemas and authenticated registry | Unit/integration tests cover authorization, replay, expiry, cancellation and malformed workers |
| B. Portable inference | One pinned DREAM checkpoint served by a worker | Real canary, streaming response, model hash and load test pass |
| C. Peer inference | OpenHydra adapter prototype | Two independently operated machines complete repeated inference; disconnect and malicious-peer tests pass |
| D. Distributed training | Hivemind compatibility prototype | Multiple independent workers train the same run; convergence is compared with a single-worker baseline; peer churn restores safely |
| E. Durable artifacts | Replicated object store | Three copies, checksum audit, full restore drill and provider migration test pass |
| F. Production resilience | Scheduler, failover and monitoring | Kill any one worker/storage replica without losing approved model or queued durable jobs |
| G. Public launch | Capacity and operations commitment | Minimum capacity, privacy policy, abuse controls, incident owner and recovery objectives documented |

No milestone is complete because code was written. Each requires recorded test evidence.

## Current DREAM repository facts

- The repository already has a provider-independent model adapter and an OpenAI-compatible request path.
- The native decoder-only transformer is initialized from random weights. Its documented default size is 4,864,000 parameters.
- The README explicitly says no production-trained native DREAM checkpoint is being claimed.
- The existing app and model runtime are separate.
- The existing Cloudflare provider configuration is a bounded inference option, not a training runtime.
- A configured provider is not necessarily ready. Runtime health must not report readiness without a successful inference canary.

## Immediate implementation sequence

1. Correct provider readiness semantics and tests (configuration must not be reported as successful inference).
2. Define worker, lease, job, artifact and signed checkpoint schemas in the DREAM repository.
3. Implement the worker registry and scheduler against a local deterministic test harness first.
4. Add an OpenHydra adapter behind a feature flag; keep it disabled until real multi-machine tests pass.
5. Prototype Hivemind against DREAM's native training code in a separate experimental package and compare convergence/recovery.
6. Select and provision durable replicated object storage only after a real no-cost grant or contributor commitment is secured.
7. Run restore, failure-injection, security and sustained-load tests before enabling production traffic.

## Resource reality and release rule

There is no software-only trick that creates free, permanent GPU capacity. A durable no-payment plan requires a real compute grant or a community of independently operated machines that agrees to contribute capacity over time. The OpenToken proposal is a potential source, not secured capacity until an award and resource allocation are confirmed.

Until those resources and the acceptance gates exist, DREAM can have a durable architecture and honest degraded-mode behavior, but it cannot truthfully claim independent, always-on GPU inference or autonomous large-scale training.
