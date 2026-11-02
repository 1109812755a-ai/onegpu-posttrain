# Design

## Context

See [proposal.md](proposal.md) for motivation and scope. The required product behavior already exists: `SourceDefinition` accepts a verified local source cache and benchmark JSONL; `build-policy-data` shares one deterministic source-pool/split protocol across policy builds; policy reports contain selection, quality, coverage and lineage evidence; experiment registration/comparison checks exact identities and declared controls.

The available OpenR1 cache is pinned to `e4e141ec9dea9f8326f4d347be56105859b2bd68`, with 93,733 rows and a complete content digest. The expected Qwen3-0.6B-Base tokenizer revision is present locally. A policy source profile and a dedicated non-empty benchmark-index JSONL are not currently present; existing `contamination.jsonl` files are match reports, not reusable benchmark indexes. The source license and attribution terms must be reviewed from the pinned source before declaring the profile usable.

The previous Mini report is a diagnostic baseline, not an A/B arm: it records 375 SFT examples from 38,140 assigned problems and combines several failure causes under the legacy `sft_no_eligible_correct_trace` counter. Its candidate correct-trace median length is 4,747 tokens against the old builder's 1,024-token cap. Do not infer the new policy build's yield from those aggregates.

## Goals / Non-Goals

**Goals:**

- Produce full-source, offline SFT data evidence for one rule difference while keeping the source pool and problem split identical.
- Preserve exploratory status, denominators, label uncertainty, length/coverage diagnostics, source terms and complete data lineage.
- Permit a bounded model comparison only after its data, parent, budget and evaluation prerequisites pass explicit checks.

**Non-Goals:**

- Change builder behavior, data-policy semantics, frozen Mini gates, shared specs, UI, or training/evaluation APIs.
- Obtain missing datasets/tokenizers from the network, call a generation provider, or alter the raw source cache.
- Promote `report_only` data to an official Mini dataset or claim that increased yield means improved quality.
- Require training to run if data review shows the arms are unsafe, insufficient, or not comparable.

## Decisions

### 1. Reuse the pinned cache and require local-only preflight

Use the existing source-cache manifest and verify its complete identity, row count and digest before building. Reuse the locally cached tokenizer only if its repository and revision exactly match the policy configuration. Build the contamination index only from already-local benchmark material whose pinned revisions and terms can be verified; do not mistake prior contamination-hit reports for an index. If any prerequisite is absent or unverifiable, stop before policy construction and record a blocker. Any network acquisition requires a separate explicit operator decision.

This is preferred over allowing the builder to fall back to Hugging Face because implicit network access would make the experiment slower and weaken the promised reproducibility. The source profile must record source revision, cache identity, benchmark identity, license/attribution declaration and tokenizer identity.

### 2. Compare one policy field and keep both arms exploratory

Use SFT arm A with `correctness: filter` and arm B with `correctness: report_only`. Both policies use `usage: exploratory`; otherwise the `usage` field itself would be a second policy difference. Keep `completeness: filter`, `answer_parse: filter`, the tokenizer/template, `max_length: 1024`, split seed/weights, and integrity safeguards identical. Unknown or incorrect labels in B remain explicit; they are never described as verified.

The 1,024-token limit is held fixed to isolate the chosen policy field and avoid silently changing hardware/recipe requirements. A separate length-cap experiment would be a different variable and is out of scope.

### 3. Exhaust the same source before comparing outputs

The policy builder stops when `target_count` is reached, which can make A and B observe different source prefixes. Set the same target above the maximum possible source-row count (100,000 for this 93,733-row cache), leave `max_scan_rows` unset, and require both reports to show 93,733 raw rows and `source_exhausted`. This forces full source traversal and retains the same deterministic problem assignment in both arms; the large target is a scan-control value, not an acceptance quota.

Interpret `stop_reason` and observed `raw_rows` against the verified cache manifest. The current `partial` report flag can also be true when the source is exhausted but the deliberately high target is unmet; do not call such an arm a truncated source scan. If a consumer cannot distinguish target shortfall from incomplete source consumption, stop and capture that as a separate behavior change rather than patching code inside this research change.

### 4. Separate data-only findings from model-effect claims

First build both datasets into new experiment-scoped directories and review the same-pool/split IDs, counts, diagnostics, denominators, quality state, coverage, length and lineage. Record an immutable experiment registration before any training run.

Classify the study as `controlled_quality_ablation` only if the policy diff contains exactly the declared correctness field and the training contract has a common parent hash, identical resolved recipe, seed, actual budget, evaluation dataset identities and evaluation protocol. The baseline may be the retained M1-CPT checkpoint; if that artifact cannot be verified, both arms must use the same verified Base checkpoint. Never train B from A's output.

Select the common training budget only after both dataset counts and the local training profile are known. Require each arm to supply the full declared budget without silent truncation or unintended repeated sampling; record token exposure and length distribution as secondary outcomes. If no safe matched budget is feasible, preserve the data comparison as a `pipeline_effect`/data-only result and keep model effect `NOT_EVALUATED` or `NOT_COMPARABLE` as appropriate.

Use the same fixed held-out evaluation sets and scoring configuration for both models. The established smoke matrix may verify execution but its small denominators support only descriptive observations. Missing local evaluation data is a stop condition, not a reason to download or change the evaluation protocol silently.

### 5. Keep artifacts additive and local

Keep experiment configuration and new artifacts under a unique path such as `artifacts/data-experiments/openr1-sft-policy-ab-<id>/`, with separate `data-a/` and `data-b/` roots and immutable registration/report outputs. Reference large caches by manifest and digest instead of copying them. Do not overwrite historical Mini builds or demo artifacts. Include source/license and benchmark provenance sufficient for review; publish no source text or derived benchmark content unless redistribution terms permit it.

## Risks / Trade-offs

- **The relaxed arm can include incorrect or unknown targets.** Keep it exploratory, expose correctness denominators, enforce format/length/integrity checks and gate training on review; never promote it to Mini.
- **Local contamination inputs may be unavailable or have unclear terms.** Stop before build; do not substitute a prior match report or silently fetch data.
- **The high scan target may produce a target-shortfall flag despite full source exhaustion.** Verify row count and `source_exhausted` against the manifest and preserve the distinction; request a separate fix if downstream reporting treats it as an incomplete scan.
- **A/B dataset sizes and token exposure may differ.** Register the actual common budget and report residual distribution differences; otherwise downgrade to pipeline-effect/data-only interpretation.
- **Single-seed or small evaluation results can be noisy.** Mark them descriptive, retain all denominators and failures, and do not claim general improvement.
- **The full-source build can consume disk and time.** Reuse the local cache, write only new outputs, estimate available space before building and stop if the workspace cannot safely hold both arms.

## Migration Plan

No schema or code migration is required. Verify prerequisites, create new operator profiles and experiment artifacts, build/validate both data arms offline, review and preregister, then conditionally run the two training/evaluation branches. Rollback means stopping before the next gated step; keep all prior outputs immutable and label incomplete attempts clearly. Never delete or rewrite the shared source cache as part of this study.

## Open Questions

- Whether an M1-CPT checkpoint with verifiable local hash is retained, and whether the preflighted training budget fits both arms, can be decided after data builds; otherwise use the same verified Base parent or stop before training.
- Whether an already-local benchmark cache can produce the required dedicated contamination index must be answered during preflight. If not, the study remains blocked pending an explicit source-acquisition decision.
