# Keyless information sources and autonomous learning

Catalog revision: 2026-11-09. The machine-readable inventory is `data/keyless-source-catalog.json`.

## Requirement

DREAM's public information discovery must not require a user to enter third-party search or data API keys. Prefer anonymous, read-only public endpoints; identify rate limits and terms; keep credentials out of the browser and repository. This requirement applies to information providers. GPU-job orchestration and private artifact storage may still need the platform credentials already configured for those services.

There is no single anonymous endpoint that provides unrestricted access to the entire internet, every social network, paywalled publications, private accounts or every current fact. DREAM should use a federated source network and report which providers actually returned evidence.

## Already connected in the codebase

The current research tools in `lib/dream-tools.js` fan out to these public providers:

| Source | What DREAM can retrieve now | Boundary |
|---|---|---|
| Wikipedia MediaWiki API | Search snippets and short article introductions | Text is generally CC BY-SA; the current commercial training policy holds ShareAlike material for review. |
| Wikidata | Public entity labels, descriptions and structured records | Structured data is CC0; still retain provenance and screen personal/sensitive records. |
| GDELT DOC 2.0 | Recent news headlines, publisher URLs, languages and seen dates | GDELT data is open; publisher article content is not automatically cleared for training. |
| Crossref REST | DOI, article title, authors, references and scholarly metadata | Crossref says almost all deposited metadata is not subject to copyright; abstracts/full text need separate checks. |
| OpenAlex | Works, authors, institutions, topics and citation metadata | Basic use works without a key; anonymous quotas are lower and cursor paging is restricted. |
| Europe PMC | Biomedical publication records and abstracts; links to open-access full text | Each full-text article has its own license; “open access” is not one universal license. |
| Google News RSS | Headline discovery, publisher links and dates | Headline/feed access is not a training license for the publisher's article. |
| Bluesky public AppView | Searchable public posts | Public visibility is not author permission for model training. |
| Stack Exchange API | Public technical Q&A, question excerpts, tags, scores and answer counts | Anonymous shared-IP quota; use for retrieval with attribution, not automatic training. |
| World Bank Indicators API | Keyless country and global statistics for population, GDP, inflation, education, health, poverty and development | Each returned value carries its observation year; it is not a live measurement. Check source-specific licensing exceptions before training. |

The main entry points are `lib/research.js`, `lib/research-global.js`, `lib/research-multisource.js`, `lib/research-news.js` and `lib/research-platforms.js`.

## Additional anonymous sources to connect

The machine-readable catalog tracks the endpoints, returned fields and training boundaries. High-value next integrations are:

1. **CKAN public-data portals** — a common read API used by government and civic open-data catalogues. CKAN is software, not a blanket data license; check each dataset and downloadable resource.
2. **Hugging Face public dataset viewer** — discover public dataset metadata, license fields, schemas, splits and sample rows without a user token for non-gated datasets. Never treat the declared license field alone as proof that all underlying records are cleared.
3. **Common Crawl** — free public crawl indexes and WARC/WAT/WET archives, useful for discovering public web pages and building a search index. Archive access is not a license to train on the archived page. Check source-level rights, opt-outs and robots signals before retention or training.
4. **NCBI PubMed E-utilities** — public bibliographic search and abstracts; no API key is required at ordinary usage rates (NCBI documents a limit of three requests per second without a key). The linked article's rights still apply.
5. **GitHub public REST API** — public repository metadata, releases, issues and source files can be read without authentication, but the anonymous limit is 60 requests per hour. Public visibility is not a license: require a verified compatible repository/file license before considering code for training.
6. **Open Library** — useful book and edition metadata. Open Library asks users to register the application use case and says its APIs are not intended as a third-party data backend, so use it conservatively for discovery rather than bulk harvesting.

These should be integrated as independent, fault-tolerant providers with per-source timeouts, backoff, cache/ETag support, retrieval timestamps, canonical source URLs, language and source status. A failed provider must not make DREAM claim that it searched the whole internet.

## Sources that should not be silently enabled

- Google Search grounding, Brave Search LLM Context, YouTube Data API and X recent-post search are currently optional integrations that require a key/token. They must remain disabled when no credential is configured; the user will not be asked to supply one.
- Meta, Instagram, Threads, TikTok and LinkedIn do not provide unrestricted anonymous access to all platform content. Use only approved public endpoints or user-authorized integrations; do not scrape private feeds or bypass restrictions.
- Reddit content has distinct commercial-use and AI-training terms. Do not ingest it into training without explicit rights-holder/platform permission.
- Amazon Creators API is an affiliate catalogue integration, not a general search engine, and has separate eligibility and approval requirements.

## Training: retrieval is not training

A question answered from live search does not need a training run. It needs retrieval, evidence ranking, source validation and citations. Training changes the model's weights and must use a separate corpus.

DREAM's existing training policy requires source URLs, license URLs, rights basis, date, privacy and quality checks, deduplication, source-group-separated train/validation/test splits and reproducible hashes. The current pipeline intentionally refuses unreviewed or unknown-rights material. Do not set human-review flags automatically merely because an endpoint is public.

A safe autonomous loop is:

1. Discover candidate datasets and source records from the keyless catalogue.
2. Fetch only public data; enforce rate limits, robots/opt-out rules and source-specific terms.
3. Store metadata and provenance first. Classify each record as retrieval-only, candidate-for-review, or approved under a documented source/license policy.
4. Run deduplication, language detection, quality checks, contamination checks and privacy/PII screening. Quarantine uncertain records.
5. Prepare reproducible corpus candidates and source-group-isolated splits. Never merge training, validation and test sources.
6. Train resumable native-model and adapter experiments only on corpora that pass the configured rights/privacy/quality gates.
7. Compare candidates on held-out, independently authored tests for reasoning, factuality, coding, safety and multilingual performance. Keep the production model unchanged unless a separate release decision passes.

Automated checks can reduce review work, but cannot honestly certify unclear legal rights, consent or every privacy issue. Candidate discovery can run unattended; disputed or unknown rights must remain quarantined.

## What the current training controller actually does

`.github/workflows/dream-ai-autotrain.yml` is the single daily **adapter-training controller** (04:17 Kenya time) and supports manual dispatch. The former duplicate trigger has been retired. The controller:

1. Requires `DREAM_AI_SUPABASE_URL` as a repository variable and `DREAM_AI_SUPABASE_SERVICE_ROLE_KEY` as a repository secret for a dedicated DREAM AI database. If either is absent, it skips database access and training. It has no fallback to Dream Education or Mariona.
2. Reads only DREAM AI training examples with explicit v3 consent, approved status and human-review fields. It requires at least 32 qualifying examples across at least 8 independent source groups; legacy records and single-session experiments are excluded.
3. Builds reproducible train/validation splits and audit files, then uploads the inputs to an owner-controlled private Kaggle dataset. The workflow checks Kaggle's dataset metadata and stops if it cannot confirm the dataset is private.
4. Reuses the existing Kaggle username/key secrets. A GitHub `HF_TOKEN` secret is optional when the private Kaggle notebook already has its own linked `HF_TOKEN` secret.
5. Launches a bounded, resumable GPU adapter experiment and uploads the result under a candidate path in the private Hugging Face adapter repository.

This workflow **does not automatically approve data**, scrape restricted services, train the Qwen foundation model, or promote a candidate into production. Candidate weights remain inactive until a separate evaluation and release decision passes. The dedicated DREAM AI database URL and key are not configured by default, so the workflow safely performs no database reads or GPU training until that independent store is set up.

The workflow also does not yet autonomously discover, license-screen and prepare every new training corpus. The public-source network above improves live retrieval; model training remains a separate process requiring source-level rights, consent, quality, privacy, deduplication and contamination checks.

The from-scratch native DREAM transformer uses separate training notebooks and scripts. The adapter workflow does not train or certify the native model. Tiny CI training tests validate code paths only; they are not evidence of production-level intelligence.

## References

- Crossref anonymous/public REST access: https://www.crossref.org/documentation/retrieve-metadata/rest-api/access-and-authentication/
- OpenAlex keyless basic API use and anonymous quota limits: https://help.openalex.org/api/authentication/
- World Bank API authentication: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
- World Bank open-data licensing and exceptions: https://datacatalog.worldbank.org/public-licenses
- Bluesky public unauthenticated reads: https://docs.bsky.app/
- Common Crawl free access and source-level rights boundaries: https://www.commoncrawl.org/get-started and https://www.commoncrawl.org/terms-of-use
- Hugging Face dataset viewer and public/gated dataset access: https://huggingface.co/docs/dataset-viewer/quick_start
- NCBI E-utilities keyless usage limits: https://eutilities.github.io/site/API_Key/usageandkey/
- GitHub anonymous REST API limits: https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api
- Europe PMC article-level open-access licenses: https://dev.europepmc.org/downloads/openaccess

This catalogue is deliberately broad but not exhaustive: sources, endpoints, terms, quotas and availability change. Verify the official provider documentation before enabling each connector.
