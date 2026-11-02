# Design

## Context

The CLI currently streams a pinned FineMath/OpenR1 Hugging Face split for each build. It also accepts local JSONL, but that path has no source-cache manifest or verification. Keep the existing stream as the default; source caching is an explicit, full-split materialization and opt-in build input.

## Goals / Non-Goals

**Goals:**

- Materialize the complete configured source split once, verify it, and let subsequent builds run offline from that cache.
- Keep cache identity tied to repo/config/revision/split and content SHA256; retain cache provenance in build evidence.
- Make failed downloads and refreshes unable to replace a previously verified cache.

**Non-Goals:**

- Automatically cache every streamed build or silently download a full split when a cache is missing.
- Cache benchmark datasets, tokenizer/model weights, or processed/canonical outputs.
- Resume interrupted downloads or introduce a database/cache service.

## Decisions

1. **Explicit two-step workflow.** Add `cache-source --config <yaml> [--split train] [--cache-dir data/raw/cache] [--refresh]`; builders gain `--source-cache <manifest-or-entry>`. This makes the first full-source download visible and deliberate, and keeps existing target-driven streaming behavior unchanged. `--source-cache` and `--input-jsonl` are mutually exclusive.

2. **Identity-addressed, content-verified entries.** Derive a full SHA256 entry key from normalized repo, config, pinned revision, and split. For Windows path-length headroom, use a 24-hex directory prefix while retaining and validating the full key in the manifest. Store under `data/raw/cache/<key-prefix>/`; the manifest points to an immutable JSONL blob named with a collision-checked prefix of its content SHA256 and records the full digest, row count, byte count and complete status. Cache reads first validate source coordinates, completion, file existence, row count and digest. No network fallback occurs when an explicitly requested cache is absent or invalid.

3. **Atomic publication and refresh.** Download to a temporary file in the entry directory, compute row count/digest while writing, and only after clean end-of-source publish the immutable blob and atomically replace the small manifest pointer. On failure, do not publish a complete manifest or change the current pointer; clean temporary files where possible. Previous content blobs are preserved, so a failed refresh cannot destroy a working cache.

4. **Evidence without external-path coupling.** Add optional source-acquisition metadata to processing reports and dataset manifest metadata: mode (`remote_stream` or `verified_cache`), cache key, manifest/content digest, and row count. Preserve existing source repo/config/revision fields. Do not use a machine-specific absolute cache path as an identity or lineage parent.

5. **Local-only storage.** Default cache root is `data/raw/cache`, already excluded from version control. The CLI can override it. Cache manifests and blobs are user data and are not committed or copied into each processed build root.

## Risks / Trade-offs

- **Full raw datasets can consume substantial disk space** → The explicit cache command clearly states that it reads the entire split; expose row/byte counts and let users choose another cache root.
- **Interrupted materialization leaves a partial file** → Never treat a temporary file as a cache entry; use completion metadata and digest validation, with cleanup on failure.
- **Pinned upstream revision can still yield unexpected source changes through upstream tooling** → Content SHA256 identifies the actual cached bytes and is recorded separately from the requested revision.
- **Cache may be moved or deleted** → Cache use fails with a clear path/identity error; existing direct streaming remains available when the user intentionally omits `--source-cache`.
