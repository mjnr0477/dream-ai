# Keyless information sources and autonomous learning

Catalog revision: 2026-11-08. The machine-readable inventory is `data/keyless-source-catalog.json`.

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

The main entry points are `lib/research.js`, `lib/research-global.js`, `lib/research-multisource.js`, `lib/research-news.js` and `lib/research-platforms.js`.

## Additional anonymous sources to connect

The catalog records each endpoint, what it can return and its training boundary. High-value next integrations are:

1. **World Bank Indicators API and Data360** — country and global development indicators across economics, population, education, health and poverty. The World Bank's default license for its own open datasets is CC BY 4.0, but individual datasets can have exceptions or third-party terms. Check each dataset's metadata before reuse.
2. **CKAN public-data portals** — a common read API used by many government and civic open-data catalogues. The CKAN API itself does not grant rights; check each dataset and downloadable resource.
3. **Hugging Face public dataset viewer** — discover public dataset metadata, license fields, schemas, splits and sample rows without a user token for non-gated datasets. Never treat the declared license field alone as proof that all underlying records are cleared.
4. **Common Crawl** — free public crawl indexes and WARC/WAT/WET archives, useful for discovering public web pages and building a search index. Archive access is not a license to train on the archived page. Check source-level rights, opt-outs and robots signals before retention or training.
5. **NCBI PubMed E-utilities** — public bibliographic search and abstracts; no API key is required at ordinary usage rates (NCBI documents a limit of three requests per second without a key). The linked article's rights still apply.
6. **Stack Exchange API** — anonymous public Q&A search without a key; responses include a remaining-quota field and can request a backoff interval. Respect shared-IP throttles and cache repeated queries. Posts carry attribution/share-alike requirements, so this connector is for retrieval rather than automatic training. [API docs](https://api.stackexchange.com/docs), [throttles](https://api.stackexchange.com/docs/throttle).
7. **GitHub public REST API** — public repository metadata, releases, issues and source files can be read without authentication, but the anonymous limit is 60 requests per hour. Public visibility is not a license: require a verified compatible repository/file license before considering code for training.
8. **Open Library** — useful book and edition metadata. Open Library asks users to register the application use case and says its APIs are not intended as a third-party data backend, so use it conservatively for discovery rather than bulk harvesting.

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

## What the current daily workflow actually does

`.github/workflows/dream-ai-autonomous-training.yml` schedules a daily training trigger and calls `scripts/launch_dream_ai_kaggle.py`. The launcher checks for an eligible private Kaggle dataset with the exact reviewed adapter-training files, then submits `training/dream_ai_adapter_self_train_kaggle.ipynb`. The notebook continues a DREAM-specific adapter on its selected foundation model and uploads candidate/progress artifacts; it does not train the foundation model and does not automatically promote a candidate into production.

The workflow currently does **not** automatically discover, download, license-screen and prepare new public training data. It also depends on the existing Kaggle credentials and Kaggle-side Hugging Face secret for job execution and candidate uploads. Those are infrastructure credentials, not public-source API keys. Missing credentials or a missing eligible reviewed dataset must stop safely rather than pretend training occurred.

The separate native model notebook trains DREAM's from-scratch transformer, but it is not the same daily adapter-training job. The native model still requires substantial, high-quality corpus data and much more compute than the tiny CI smoke tests.

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
