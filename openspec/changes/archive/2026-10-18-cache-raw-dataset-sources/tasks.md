# Tasks

## 1. Source Cache Format and Materialization

- [x] 1.1 Implement source-cache identity/manifest models and streaming JSONL materialization with row count, byte count, SHA256, completion state, temporary-file cleanup, and atomic manifest publication; verify complete, interrupted, empty, and refresh-preserves-prior-cache cases with unit tests.
- [x] 1.2 Add `cache-source` CLI for a pinned dataset config and split, with an overrideable cache root and explicit refresh; verify help, successful fake-source caching, and nonzero incomplete-fetch behavior in CLI tests, and document the prefetch command and disk-space implications in Chinese operator docs.

## 2. Offline Builder Reuse and Provenance

- [x] 2.1 Add validated `--source-cache` consumption to FineMath and OpenR1 builds, reject identity/digest/complete-state mismatches without remote fallback, and record acquisition mode/cache identity in processing reports and dataset manifests; verify offline reuse and provenance with integration tests.
- [x] 2.2 Preserve direct streaming and existing `--input-jsonl` behavior, reject conflicting source-input options, and verify regression cases plus cache tampering and source-revision mismatch errors.
- [x] 2.3 Update the Chinese data-build guide with cache creation, reuse, refresh, offline operation, storage location, and troubleshooting; run each documented command against CLI help or fixtures.

## 3. End-to-End Contract Verification

- [x] 3.1 Run focused source-cache/data-build tests, Ruff, and strict OpenSpec validation; verify a verified cache can support a fresh build with network access disabled and that lineage/readiness validation sees the recorded source evidence without machine-specific absolute paths.
