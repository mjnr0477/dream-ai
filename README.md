# DREAM AI

DREAM AI is an independent AI research and application project, separate from Dream Education, Dream Wi-Fi and VYBORA.

## African-language evaluation expansion

`evals/african-languages-v1.jsonl` adds 20 targeted evaluation prompts across Kiswahili, Yoruba, Hausa, Amharic, Igbo, Somali, Kinyarwanda, Luganda, Zulu and Xhosa. Each language has a multi-step arithmetic task and a concept-explanation task, so the set checks more than simple translation.

Run its structural checks with:

```sh
npm run validate:african-evals
```

All cases are explicitly marked `pending_native_review`. The prompts and rubrics are engineering drafts, not certified translations. Before using this set to make language-quality claims or as supervised training data, have qualified native speakers review and correct every prompt and rubric, record the review provenance, and score model outputs independently. A passing validator proves only dataset structure and coverage; it does not prove model fluency or competence.

