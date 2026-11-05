# OASST1 candidate import for DREAM

DREAM includes a conservative importer for the OpenAssistant Conversations dataset (OASST1). The upstream dataset card identifies OASST1 as Apache-2.0 and describes 88,838 messages in its ready_for_export export. The source export is pinned to revision:

https://huggingface.co/datasets/OpenAssistant/oasst1/tree/56e16363c12104182b8ad8173591b4d1e31c1faa

## Run in Colab

Download the upstream 2023-04-12_oasst_ready.messages.jsonl.gz file from the pinned dataset revision, then run:

```bash
python data/import_oasst1_candidates.py \
  --input data/2023-04-12_oasst_ready.messages.jsonl.gz \
  --output data/oasst1-candidates.jsonl \
  --report data/oasst1-candidate-report.json
```

## Important: candidates are not approved training data

Every exported row has rights_confirmed, privacy_reviewed, quality_reviewed, and approval_granted set to false. This is intentional. The upstream Apache-2.0 label is useful provenance, but it does not replace DREAM's own review of licensing obligations, user-generated content, privacy, quality, and intended use. The candidate file must fail DREAM's training gate until a human reviews and explicitly approves the records. Never bulk-change these flags merely to make training run.

The importer deliberately omits OASST user IDs. It uses message-tree IDs as source groups so messages from one conversation tree cannot be split across training and evaluation. The manifest later fingerprints these identifiers.

OASST1 is a general multilingual conversation dataset, not a Swahili corpus. Its published language list does not include Swahili. It can be one reviewed seed source, not DREAM's complete pretraining corpus and not evidence of Kenyan-language capability. Find and separately review suitable Swahili, English, Sheng, and other target-language sources before making language claims.

This importer does not guarantee absence of personal information or harmful content. Automated labels are imperfect; human privacy and quality review remains mandatory.
