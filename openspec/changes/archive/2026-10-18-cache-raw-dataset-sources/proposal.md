# Proposal

## Why

The data builders currently stream Hugging Face rows directly into each build, so repeating a build may fetch and process the same pinned source again. A local, verifiable raw-source cache will let users iterate on cleaning and eligibility rules without repeatedly downloading the upstream dataset, while preserving source revision and reproducibility evidence.

## What Changes

- Add an explicit command/workflow to materialize a pinned Hugging Face dataset split as local raw JSONL plus a cache manifest.
- Allow FineMath and OpenR1 builders to consume a validated cache offline, while retaining their existing direct-streaming behavior when cache use is not requested.
- Key and validate cache entries by repository, config, revision, split, row count, and content digest; never silently reuse incomplete or mismatched data.
- Record cache provenance in generated processing reports and dataset manifests.
- Keep source cache outside generated processed datasets and document storage, reuse, verification, and refresh behavior.

## Capabilities

### New Capabilities

- `dataset-source-cache`: Explicitly materialize, verify, and reuse immutable local copies of pinned upstream dataset splits.

### Modified Capabilities

None.

## Impact

- Affected code: `src/onegpu_posttrain/data/sources.py`, data-build CLI in `src/onegpu_posttrain/cli.py`, processing/manifest evidence, and related tests.
- User-facing CLI and docs: add source-cache command/options and explain that cache creation fetches the full selected split once; subsequent builds can run offline from a verified cache.
- Storage: raw JSONL caches may be large and remain local/ignored; no database, external cache service, or new required dependency is introduced.
- Existing streaming commands remain behavior-compatible unless the user explicitly opts into a source cache.
