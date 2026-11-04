# 易经八卦权重空间练习集

`weight-space-practice-iching-300.jsonl` contains 300 Chinese objective-choice practice records for weight-space experiments. The current version is registered locally as `ds-eval-upload-14aee30e5c7ec6ec0959` using `objective-choice-jsonl-v1`.

## Import evidence

- 300 source rows; the existing importer preview accepted 300 and rejected 0.
- Registered role: `evaluation_only`; this does not make it the final held-out benchmark.
- Final model-effect reporting must continue to use the original 81-question suite, `ds-eval-upload-54ca42ca021d295d8390`.
- Registration and lineage evidence are under `data/processed/user-upload/ds-eval-upload-14aee30e5c7ec6ec0959/`.
- First registration `ds-eval-upload-e084610fbfbcfe252c90` was superseded after a lexical near-duplicate was found. It remains locally registered because the app has no deletion operation; use the v2 ID above.

## Files

- `weight-space-practice-iching-300.jsonl`: importer-compatible source rows.
- `weight-space-practice-iching-300.provenance.json`: question IDs, rationale, topic, pinned source revisions, attribution, overlap review, and registration identities.
- `generate_iching_practice.py`: deterministic generator (seed `20261103`).

## Source and review limits

The questions are newly worded from pinned Zhouyi/Wikisource text and trigram/hexagram tables. Wikisource attribution and CC BY-SA 4.0 notices are recorded in the provenance file; the imported source license is marked `user_declared_unverified`. A normalized exact-stem and character 3-gram comparison found no exact matches or high lexical-overlap pairs after the revision. The 81-question suite and practice set share the same subject, so lexical comparison cannot prove semantic non-overlap. Answer keys are based on source facts and verified construction rules; no independent subject-matter expert review is claimed.
