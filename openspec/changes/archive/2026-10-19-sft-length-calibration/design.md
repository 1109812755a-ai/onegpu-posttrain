# Design

## Context

The smoke SFT recipe is intentionally fixed at 1,024 tokens, and the data builder already treats rendered length as an eligibility gate rather than silently truncating. DeepMath is cached locally as Parquet, but it is not yet an official Mini dataset and needs an explicit adapter before project-formatted examples can be built. See `proposal.md` and the capability contract for motivation and observable behavior.

## Goals / Non-Goals

**Goals:**
- Make local-source length retention reproducible and auditable before allocating GPU time.
- Keep source/sampling identity, tokenizer/template identity, quality coverage, and hardware results in one calibration bundle.
- Run bounded resource probes only after candidate data can be built and the user explicitly enables GPU probing.

**Non-Goals:**
- Changing the official Mini target, quality gates, or `recipes/smoke/20-sft.yaml` default.
- Claiming that a higher token cap improves model quality, or treating a few-step probe as an efficacy experiment.
- Downloading datasets or adding a mandatory data-quality framework.

## Decisions

1. **Add a separate local calibration path, not a new official dataset build.** Read the locally cached DeepMath Parquet snapshot and adapt its question, answer, and reasoning-trace columns into temporary SFT candidates. Use stable source-row/trace identities and snapshot/content digests. Missing cache is a clear stop condition; do not fall back to Hub access. This keeps the exploratory data from accidentally satisfying Mini readiness.

2. **Select a deterministic sample by content identity, not by taking the head of each shard.** Pin source snapshot, sample size, and seed; choose rows with a stable hash-based ordering across all shards, then preserve that same source pool/order for all candidate budgets. This reduces shard-order bias and makes reruns comparable. The report states sample size and selection method; a bounded sample remains exploratory, not a population estimate.

3. **Measure the fully rendered training sequence using the project tokenizer/template path.** Count the final prompt-plus-answer representation using the pinned Qwen tokenizer and actual SFT chat template. Record both trace-level denominators and distinct-problem coverage because DeepMath can contain multiple traces per question. Reject over-budget traces without truncation; emit a separate manifest per candidate length and do not mutate existing source/build outputs.

4. **Separate length retention, semantic verification, and resource feasibility.** Reuse available project verification where the source answer format is supported. If verification cannot be performed, retain explicit `not_measured` coverage and do not call those candidates verified. First produce CPU-only retention/quality evidence; an explicit bounded GPU option may then run short SFT probes. Keep parent checkpoint, seed, optimizer settings, sample pool, update budget, and software environment fixed; eligible records can differ by cap, so record each dataset digest/count and do not interpret probe loss or outcomes as a model-quality comparison.

5. **Use immutable, namespaced calibration artifacts.** Write under a new calibration/run identity, never into `data/processed/mini-v1` or an existing run directory. Store the resolved protocol, per-cap manifests, report, and per-probe `run.json`/resource measurements together. Failed/OOM attempts remain evidence and do not overwrite successful results.

6. **Keep candidate budgets explicit and non-defaulting.** Start with 1,024/2,048/4,096 tokens as an exploratory matrix. The calibration command requires an output identity and candidate set; it does not rewrite the smoke recipe or select a new project-wide default. Any future default change requires review of the calibration evidence and a separate change.

## Risks / Trade-offs

- [DeepMath trace fields or answer conventions may not map cleanly to the existing SFT schema] → Validate required columns and representative records before building; report unsupported/malformed counts and stop rather than guessing field semantics.
- [A bounded deterministic sample can still misrepresent the full corpus] → Publish denominators, per-shard/sample coverage, seed and selection method; label rates exploratory and rerun with a larger sample before any policy decision.
- [The 4,096-token probe may OOM or create a large checkpoint] → Require an explicit GPU-probe flag, cap steps, avoid retaining checkpoints unless needed, use a unique run directory, and preserve OOM as a valid result.
- [Different caps naturally produce different eligible datasets] → Fix source pool/order and training controls, record per-cap dataset hashes/counts, and limit conclusions to retention and resource feasibility.
- [GPU utilization and peak memory vary with background processes] → Capture device/driver and pre/post-run memory snapshots; mark measurements as unavailable or contaminated when reliable attribution is not possible.

## Migration Plan

No existing artifact or training default is migrated. Add the calibration command and schemas as an isolated opt-in path; validate first on CPU/local cached inputs, then run explicitly requested short GPU probes into new artifact directories. Rollback consists of not invoking/removing the new calibration outputs; no official Mini data or existing run is rewritten.
