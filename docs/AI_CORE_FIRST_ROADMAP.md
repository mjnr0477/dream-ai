# DREAM AI — AI-core-first development policy

## Product sequencing

DREAM AI is being developed as an intelligence product first. New DREAM user registration, sign-in, account profiles, cross-device account memory, subscriptions, billing and account-management features are explicitly deferred. The initial product remains guest-first. This is a sequencing decision, not a removal of the existing Plugins catalogue or the separately permissioned connector work already in the repository.

Do not start account-system implementation merely to make the product appear more complete. Revisit it only after the core capability and reliability gates below have been demonstrated and the owner explicitly prioritizes it.

## What 'ahead' must mean

Do not claim DREAM is ahead of other AI systems because of branding, a long feature list, a large source catalogue, a training run, or a few impressive examples. Compare it with relevant contemporary systems using repeatable, independently authored tests, equivalent tools and context budgets, blinded human review where subjective judgment is required, and published sample counts. Report strengths and weaknesses by capability; never hide weak categories inside one aggregate score.

## Core workstreams

1. **Independent model capability.** Improve the native DREAM model through lawful, high-quality, multilingual pretraining and instruction tuning. Keep the hosted adapter and native from-scratch model clearly distinguished. A small synthetic CI checkpoint is a pipeline test, not evidence of useful intelligence.
2. **Reasoning and verification.** Build reliable problem decomposition, mathematical and code execution checks, tool-result verification, contradiction handling, long-context synthesis, and explicit uncertainty. Use bounded agents and least-privilege tools; do not let generated plans grant themselves permissions.
3. **Broad, fresh knowledge.** Expand fault-tolerant public-source retrieval across science, education, engineering, public data, news, geography and other domains. Preserve source links, timestamps, provenance, licenses and provider status. Public retrieval is not private-account access and does not automatically grant training rights.
4. **Forecasting and decision support.** Build domain-specific inputs for weather, sport, markets and other events. Track the timestamped forecast, horizon, complete outcome set, resolved result and baseline. Evaluate chronological held-out forecasts with Brier score, log loss, calibration and baseline skill before claiming accuracy.
5. **Creation and multimodality.** Build dependable writing, coding, structured file creation, data analysis, image/audio/video understanding and generation where the model/runtime and safe execution environment actually support them. Listing a capability is not the same as implementing it.
6. **Safe tool use.** Connect useful public services and user-authorized integrations with clear scopes, audit trails, approval for consequential actions, robust error handling and graceful operation when a provider is unavailable.
7. **Measured self-improvement.** Discover candidate data, record provenance, check rights and privacy, deduplicate, test for contamination, train isolated candidates, evaluate on held-out benchmarks and promote only after independent release review. Never allow the model to silently approve its own training data or release its own candidate.

## Evidence gates before an 'ahead' claim

- **Knowledge:** demonstrate current-answer quality with source-grounded factuality, citation correctness, temporal accuracy and source-failure tests.
- **Reasoning:** demonstrate multi-step correctness, mathematical accuracy, coding task success, reliable tool use and calibrated uncertainty on held-out tests.
- **Forecasting:** publish sample counts, forecast horizons, baselines, Brier/log-loss results and calibration by domain. Compare only like-for-like events and periods. A forecast is not validated by a few correct guesses.
- **Languages:** report per-language results and qualified human review. Prompt coverage or tokenizer efficiency alone is not language proficiency.
- **Safety and reliability:** pass adversarial, privacy, prompt-injection, permission-boundary, recovery, latency and availability tests.
- **Comparative performance:** use a maintained benchmark suite against relevant alternatives with identical task conditions. Claim an advantage only where measured results support it.

## Current verified constraints

- The hosted chat route has previously used a DREAM-specific adapter on Qwen3-1.7B; tuning that adapter does not train the Qwen base model.
- DREAM's native transformer is an independently designed from-scratch research architecture, but no production-quality generally capable native checkpoint is being claimed.
- Existing public-source connectors return a mixture of headlines, snippets, abstracts, metadata and selected public records. They do not constitute unrestricted internet access or comprehensive structured weather, sports and market feeds.
- The current general evaluation datasets are small and include tasks that require human scoring. Structural validation and synthetic CI tests do not establish frontier-level intelligence.
- Free compute and anonymous public endpoints have real capacity, rate, licensing and commercial-use limits. Do not solve these limits by inventing API access or weakening data-rights checks.

## Change-control rule

Preserve working chat, Plugins, public research, Gmail's existing consent boundary, training safeguards and the separate Dream Education, Dream Wi-Fi and VYBORA projects. Prefer small, tested increments. Do not add DREAM user-account, login, billing or subscription scope until the core gates have evidence and the owner gives a new explicit instruction.