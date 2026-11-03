# DREAM training data

This directory is reserved for datasets created or lawfully licensed for DREAM.

Do not put private user conversations, passwords, personal records, copyrighted books
without permission, or scraped personal data here.

## Data acceptance rules

Every dataset release must include:
- source and collection date;
- written permission or an explicit license permitting the intended use;
- language and subject labels;
- personal-data review and redaction record;
- deduplication and quality-review report;
- train/validation/test split with no cross-split leakage;
- dataset version and SHA-256 manifest.

## First target domains

1. General language and instruction following.
2. Mathematics with worked, independently checked solutions.
3. Programming examples with runnable tests and license provenance.
4. Teaching explanations and learner misconceptions.
5. Kiswahili and English, with additional Kenyan languages only where data quality and permission are adequate.
6. Kenyan curriculum and local-context content with authoritative source citations and version dates.

## Important

The repository intentionally does not ship a toy corpus and claim that it makes a useful assistant. A byte-level transformer needs a large, clean corpus and substantial compute. Begin with a small smoke-test corpus only when it is clearly labeled as such; never evaluate or market it as production intelligence.
