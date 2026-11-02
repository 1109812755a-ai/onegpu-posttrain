# Design

## Context

See `proposal.md` for motivation. The project already has a target-aware OpenR1 builder with a configurable `max_raw_rows`, a non-zero CLI result for unmet build targets, GRPO run metrics including completion clipping, a generic `run-model-eval` command, and file-based experiment/run/model/evaluation lineage. The frozen v0.1.1 evaluation protocol remains authoritative; this change adds verification and evidence around it rather than redefining it.

The OpenSpec root currently has no main specs. This change therefore establishes three durable capabilities in its delta specs: data build evidence, unified model evaluation, and GRPO calibration evidence.

## Goals / Non-Goals

**Goals:**

- Turn the four dev7 gates into reproducible checks and saved artifacts, including hardware-run evidence where required.
- Preserve current data eligibility, evaluation and lineage semantics.
- Make reports distinguish measured, unavailable, failed and incomplete evidence.

**Non-Goals:**

- No change to preference correctness/completion/length/contamination gates, evaluation datasets or answer parsing.
- No Mini training, OVAT study, new scheduler, database, new runtime dependency or full UI work.
- No retroactive modification of historical Smoke artifacts or claims.

## Decisions

### 1. Verify the existing OpenR1 scan contract before changing its implementation

The builder already checks all configured targets, supports `max_raw_rows`, emits stop reason/target/counter data, and the CLI returns a failure code when targets remain unmet. First add/retain focused fixtures proving that it keeps consuming rows after an early eligible pair, stops at the explicit cap or true source exhaustion, preserves train/eval disjointness, and reports actionable counters. Inspect the saved real build report to explain the observed shortfall. Change implementation only if a fixture or real-stream evidence shows a contract defect; do not relax eligibility or add a hidden scan cutoff.

Alternative considered: increase quotas by weakening gates or silently sampling until a target is filled. Rejected because it would change the frozen dataset contract and make the evidence less trustworthy.

### 2. Treat GRPO length calibration as a controlled experiment, not a config-only change

Use the existing GRPO profiles and reward setup as the baseline. Run candidate `max_completion_length_candidate` values 512, 768, 1024 and 1536 with fixed M2 parent, prompt IDs, seed, generation settings, profile, training/completion budget and all other resolved configuration. Keep a shared `experiment_id` and use a distinct `run_id` per candidate; record the resolved config/hash and environment for each run. Preserve OOM/failed candidates as explicit outcomes. Compare actual generated lengths and clipping ratio with verifier/format reward, wall time and peak VRAM. Change a recipe default only after the saved comparison supports that decision; if no candidate clears the clipping target, report the best supported candidate without declaring the target passed.

Alternative considered: select a higher completion limit from intuition or training logs alone. Rejected because it cannot establish impact on the reference 8GB GPU or reward validity.

### 3. Reuse the frozen evaluator and create a complete M2/M3 evaluation matrix

Use the existing generic model-evaluation flow and frozen evaluation configuration for M2-SFT and the four M3 merged models. Keep GSM8K, MATH-500, and the same internal-holdout IDs, prompt/decoder settings, parser and limits. Save the ordinary per-model `eval.json` and evaluation lineage artifacts, then produce a matrix/comparison report keyed by model and protocol. Include explicit missing/failed/partial statuses; never coerce absent metrics to zero. Keep preference and GRPO-specific metrics as separate evidence, not substitutes for task metrics.

Alternative considered: create a new evaluator or change the benchmark limits for the M3 models. Rejected because the current evaluator and frozen protocol are already the intended comparison surface.

### 4. Keep repository hygiene separate from experiment semantics

Run the configured Ruff check on `src` and `tests`, fix findings without unrelated behavior changes or broad suppressions, then run the full test suite and `compileall -q src plugins`. Record the command/version and outcomes in the project status. If Ruff is not installed in the environment, report that as an environment limitation rather than claiming a pass.

Alternative considered: suppress existing findings globally or treat lint success as a substitute for tests. Rejected because neither establishes regression safety.

### 5. Reconcile project status from artifacts, not from memory

Inspect the completed `implement-lineage-v1` change and actual lineage artifacts, then update stale checkboxes/status wording in implementation status and milestone/design status only where the repository evidence supports completion. Preserve historical Smoke reports as immutable evidence; add a dated correction or current-status note instead of rewriting old experimental outcomes.

Alternative considered: mark all lineage work complete solely because the OpenSpec change has 20/20 tasks. Rejected because task completion does not prove every required artifact exists in the current workspace.

## Risks / Trade-offs

- **Longer completion limits may OOM or consume substantial GPU time** → keep runs sequential, preserve OOM as a result, and do not automatically choose an unsupported default.
- **The streamed source may not yield enough eligible preference pairs within the configured cap** → expose the true shortfall and diagnostics; defer any sampling-policy change to a separately versioned design decision.
- **Historical M3 checkpoints may be unavailable or incompatible with current evaluation dependencies** → record each missing/failed run explicitly and do not block the report from showing available results.
- **Ruff or optional evaluation tooling may be absent locally** → record the exact environment limitation and leave the relevant acceptance gate open.
- **Legacy smoke lineage may be incomplete** → report completeness according to current validator results; do not fabricate or silently backfill provenance.

## Migration Plan

No data or model migration is required. New calibration/evaluation attempts use fresh run output directories and append new evidence. Existing historical artifacts remain unchanged. If a candidate recipe default is selected, preserve the previous recipe in the change history and record the chosen value and evidence in the resolved run configuration. Rollback consists of reverting only the changed recipe/reporting code; saved run artifacts remain inspectable.

## Open Questions

None that change the agreed scope or acceptance criteria. Hardware-runtime duration and whether every candidate completes are empirical outcomes and must be reported, not decided in advance.
