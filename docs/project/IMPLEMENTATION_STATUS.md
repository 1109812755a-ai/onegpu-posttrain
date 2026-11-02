# OneGPU-PostTrain Implementation Status

Package version: `0.1.0.dev6`  
Status updated: 2026-10-20  
Reference execution: Windows / Python 3.10.11 / RTX 4060 Laptop 8GB

## Completed

- [x] Design revision v0.1.1 freeze for product release v0.1 and smoke pipeline implementation.
- [x] FineMath and OpenR1 smoke data builders, split/decontamination gates, and ms-swift adapters.
- [x] Real RTX 4060 Base evaluation, PEFT-CPT and M1 merge, Full-CPT probe, SFT and M2 merge.
- [x] Real DPO, original SimPO (`cpo_alpha=0`) and common held-out preference scoring.
- [x] Real GRPO-standard (`beta=0.04`) and GRPO-lowmem (`beta=0`) execution, metrics, comparison and merges.
- [x] 20-step smoke runs for both GRPO profiles: standard mean reward 0.0325; lowmem 0.02375. Observed completion clipping averaged 90% and 91.25%, respectively; these are diagnostic signals, not effect-size claims.
- [x] Unified GSM8K / MATH-500 / internal / domain-PPL evaluation completed for Base, M1-CPT and M2-SFT.
- [x] M2, DPO and SimPO evaluated on the same 3 held-out preference pairs. The sample is too small for ranking.
- [x] Full local unit suite: 54 passed. `compileall`, example-data validation and `pip check` passed.
- [x] 2026-10-16 data-evidence slice: full CPU suite 65 passed; touched data/report modules and related tests pass Ruff. The 54-test/compile/pip checks above remain the historical dev6 baseline.
- [x] 2026-10-17 dev7 verification: Ruff 0.17.0 `ruff check src tests` clean; full suite 112 passed, 1 skipped; `python -m compileall -q src plugins` passed. UI API tests require local loopback; sandboxed socketpair stalled, but the complete suite passed outside the sandbox.
- [x] 2026-10-17 early read-only Data UI slice: real/explicit demo scopes, processing and quality evidence, lineage-linked runs, safe artifact summaries, filters and local UI/API. This is a preview; the remaining five pages and user-task UX walkthrough remain open.
- [x] Real execution report: [`runs/smoke/RUN_REPORT_zh.md`](../../runs/smoke/RUN_REPORT_zh.md).
- [x] 2026-10-18 `data-policy-experiments` implementation: versioned exploratory/baseline SFT and Preference policy, same-pool/split preview/build, experiment preregistration and conservative comparison, opt-in local policy editor/report view, Chinese guide and CPU-owned-fixture walkthrough. Focused Python tests: 36 passed; browser-state unit tests: 2 passed; `pytest --ignore=tests/test_ui_api.py`: 181 passed; Ruff, compileall and Vite production build passed. The existing `test_ui_api.py` TestClient suite stalls on sandbox loopback initialization and was not included in this run. Fixture lineage validation passed with 4 runs/4 artifacts and 0 warnings. Fixture report is `NOT_EVALUATED`; it is not model-effect evidence.
- [x] 2026-10-19 real OpenR1 SFT policy A/B: preregistered `correctness: filter` vs `report_only`, trained both independently from the same hashed M1-CPT parent for 20 optimizer steps (seed 42), and evaluated on the same local held-out set (20 examples). Both runs and the comparison tool report `COMPARABLE` with complete control checks; accuracy was A 0/20 vs B 1/20 (+0.05), parse rate 19/20 for both. This is descriptive smoke evidence only (single seed, tiny denominator); B contains 327/694 upstream incorrect-labeled examples and all artifacts remain `exploratory`. RTX 4060 sampled peak VRAM was 7,919/7,949 MiB, near device capacity. Full report: [`final-summary.md`](../../artifacts/data-experiments/openr1-sft-policy-ab-20261019-01/final-summary.md); detailed data report: [`data-quality-comparison.md`](../../artifacts/data-experiments/openr1-sft-policy-ab-20261019-01/data-quality-comparison.md).

## Current phase: Mini exploratory end-to-end path; formal readiness and launcher evaluation remain open

Dev7 hardening and M2/M3 smoke evaluation gates are complete with documented limitations. The separate 5K / 2,048-token DeepMath SFT pilot remains blocked: its short probe passed, but the complete run OOMed at optimizer step 19/624. Do not retry unchanged. After that pilot summary was written, a reduced exploratory path was also completed: 5M-token CPT produced M1; a 307-example, 1,024-token SFT subset produced M2; DPO, original SimPO, GRPO-standard and GRPO-lowmem branches were trained and merged; and all five models were evaluated under the same smoke-scale `math-eval-v1` matrix. This proves the reduced-budget pipeline and its model/evaluation lineage can be traversed, but it is not an official Mini baseline or evidence of algorithm gains. The matrix uses only GSM8K 2, MATH-500 10, internal holdout 5 and domain-PPL 5; preference training used 4 pairs, and the GRPO runs were only 3 steps with 100% completion clipping. The actual evidence is in [`Mini alignment summary`](../../runs/mini/alignment-existing-data-v1/mini_alignment_summary.json).

The local Launcher frontend has also completed a real 38/38-step SFT run and recorded its training/model lineage. Its follow-up benchmark evaluation was attempted twice and failed while EvalScope wrote a review-cache file under a deeply nested Windows path (`FileNotFoundError`); the failed evaluation runs and lineage were preserved without fabricated scores. The Launcher evaluation workflow therefore remains open for a Windows path-length fix and successful rerun. Mini data readiness remains **NOT READY** (22 blockers); CPT review is `legacy_unknown`, DeepMath reasoning review is pending, and OpenR1 preference remains 4/2,000. All these runs remain exploratory. Do not infer model gains or mark official baseline readiness.

## Mini data readiness implementation (2026-10-17)

- [x] Add versioned `mini-data-readiness@1` budgets, read-only readiness evaluation, independent budget/quality/processing/lineage dimensions, JSON/Markdown reports, and the `validate-mini-data` CLI. Operator workflow: [`MINI_READINESS_CLI_zh.md`](../implementation/data/MINI_READINESS_CLI_zh.md).
- [x] Full-source Mini data evidence retained at `data/processed/mini/2026-10-17-build-02`; the readiness assessment is **NOT READY**, not a Mini training baseline. FineMath CPT train **5,000,368 / 5,000,000 tokens (PASS)**; CPT holdout 50,078 tokens; OpenR1 SFT **375 / 5,000 examples (shortfall)**; preference **4 / 2,000 pairs (shortfall)**; GRPO **2,000 / 2,000 prompts (PASS)**; internal holdout 500 examples; preference eval 1 pair.
- [x] FineMath protocol v2 uses chunk-content SHA256 in record IDs while retaining URL-keyed train/holdout assignment. Fresh build-02 confirms the IDs pass structural uniqueness; byte-identical duplicate-row rejection remains covered by tests. The old build-01 artifacts remain immutable historical v1 evidence.
- [x] Full OpenR1 scan completed with source revision `e4e141ec9dea9f8326f4d347be56105859b2bd68`, 93,733 rows, and stop reason `source_exhausted`. Saved processing report records stage yields/rejections; no eligibility gate was relaxed to meet targets.
- [x] Build-02 readiness assessment `mini-check-e47dcde35bbd46c0aaff17a22ccaa5ab` reports 9 blockers: two budget shortfalls and seven `quality_not_evaluated` states. Current `structural-evidence-v1` covers schema validity, unique IDs and train/eval separation; independent semantic correctness and near-duplicate checks remain unmeasured. No post-hoc threshold was added; model effect is **NOT EVALUATED**.
- [x] `validate-lineage --root data/processed/mini/2026-10-17-build-02 --require-complete` passes: 9 runs, 14 artifacts, 0 errors, 0 warnings. This is provenance integrity only, not data acceptance or model-effect evidence.

Evidence: [`Mini readiness report`](../../data/processed/mini/2026-10-17-build-02/reports/mini-readiness/mini-check-e47dcde35bbd46c0aaff17a22ccaa5ab/mini_readiness.md), [`OpenR1 processing report`](../../data/processed/mini/2026-10-17-build-02/openr1/reports/processing_report.md), [`Mini milestone`](milestones/v0.1-mini.md).

- [x] Verify target-driven OpenR1 preference scanning continues to a requested count or explicit scan cap; fixture tests cover target continuation/source exhaustion, and a pinned real-source 5,000-row capped build saved rejection evidence and exited nonzero for unmet targets. This does not establish full-source exhaustion or preference-target completion; see [dev7 data evidence](../implementation/data/DATA_EVIDENCE.md).
- [x] Complete the unified M2/M3 RTX 4060 smoke evaluation for GSM8K (5), MATH-500 (25), internal holdout (20) and domain PPL (20); all five runs are complete and protocol/config/dataset lineage is consistent. Results are deliberately labeled smoke-only; see [matrix report](../implementation/evaluation/UNIFIED_EVALUATION_MATRIX.md).
- [x] Execute the prepared controlled GRPO calibration sequentially at 512 / 768 / 1024 / 1536 tokens. All candidates passed; none reached clipping below 30%. The 1536 candidate had the lowest clipping (55%) but mean accuracy reward 0, 7,417 MiB sampled peak VRAM and 1,746 s runtime, so it is not promoted to the default recipe. Full results and limitations: [GRPO completion calibration](../implementation/training/GRPO_COMPLETION_CALIBRATION.md).
- [x] Run the same GSM8K, MATH-500, internal-holdout and domain-PPL protocol across M2-SFT and all four M3 branches; all five evals and the status-aware matrix are complete. See [matrix report](../implementation/evaluation/UNIFIED_EVALUATION_MATRIX.md).
- [x] Clear repository Ruff findings: Ruff 0.17.0, `ruff check src tests` passes with zero findings and no broad ignores.
- [x] Draft the file-based Lineage v1 contract, stable ID rules and data/model/run/evaluation graph in [`docs/design/LINEAGE_DESIGN.md`](../design/LINEAGE_DESIGN.md).
- [x] Draft the stage-specific data-quality and detailed processing/effect contracts in [`docs/design/DATA_QUALITY_DESIGN.md`](../design/DATA_QUALITY_DESIGN.md) and [`docs/design/DATA_PROCESSING_EFFECT_DESIGN.md`](../design/DATA_PROCESSING_EFFECT_DESIGN.md).
- [x] Review/freeze the lineage schema and implement dataset manifests/per-run `lineage.json`; existing historical dangling references remain documented limitations.
- [x] Emit typed processing reports for both builders, ordered steps, decision accounting, available before/after evidence, benchmark-index provenance, stop reasons, and hashed manifest report references. See [implementation](../implementation/data/DATA_EVIDENCE.md).
- [x] Emit initial per-stage static quality reports: schema/duplicate-ID/global-overlap checks, lengths/source summaries, source-label consistency and explicit unavailable semantic metrics. Dataset IDs now reflect content and configuration; legacy manifest paths remain compatible.
- [x] Add versioned stage-quality review profiles, digest-bound offline review bundles, explicit reviewer decisions, usage-aware official readiness, optional answer verification and CLI integration. See [review workflow](../implementation/data/QUALITY_REVIEW_zh.md) and [build-02 review](../implementation/data/MINI_BUILD02_REVIEW_zh.md).
- [ ] Complete trustworthy references and human review for official Mini; current build-02 remains `not_ready` with legacy usage, zero independent-answer coverage, and SFT/Preference shortfalls. Model-effect experiments remain pending.
- [ ] Reconcile this status and run artifacts after each gate; keep smoke feasibility separate from quality claims.
- [ ] Complete browser-based task UX review of the Data/policy page: discoverability, filter/context retention, keyboard flow, narrow-screen use, and clear loading/empty/error/missing-evidence states. A reproducible walkthrough is documented, but a browser automation driver is not installed in this workspace.

Lineage verification note: all tasks in `implement-lineage-v1` are checked. The isolated dev7 evaluation root validates with 0 errors; its warnings reflect references outside that root (model/data/benchmark evidence). A broader `validate-lineage --root runs` scan still reports 10 dangling references in GRPO calibration records whose parent/checkpoint paths are not retained; these are preserved as limitations, not relabeled as complete evidence.

## Remaining v0.1 release gates

- [ ] Promote smoke recipes to Mini data budgets: 5M CPT tokens, 5K SFT examples, 2K verified preference pairs, and 2K verifiable GRPO prompts.
- [ ] Complete the four required OVAT experiments: CPT data size, SFT learning rate, DPO vs. original SimPO, and GRPO reward design.
- [ ] Run the representative SFT learning-rate point on seeds 42 / 43 / 44.
- [ ] Publish result provenance with configs, dataset manifests/sample IDs, environment, checkpoint lineage/hashes, evaluation protocol and resource metrics.
- [ ] After the Mini baseline and required OVAT contracts stabilize, decide whether to run the optional matched-budget cleaning study; formal reusable data-quality ablations are planned for v0.3 and are not a v0.1 release gate.

## Frozen interpretation rules

- Full-CPT and GRPO-standard OOM are valid recorded feasibility outcomes.
- GRPO-lowmem uses `beta=0`, no reference model and KL=N/A by design.
- Smoke runs validate execution and instrumentation; they do not establish algorithm quality.
- Conclusions must be bounded to the tested model, data, budget, seed and evaluation protocol.
- Do not compare loss values across different objectives as if they shared a scale.

## Next stages

See [`DEVELOPMENT_PLAN.md`](DEVELOPMENT_PLAN.md) for phase gates and [`milestones/`](milestones/) for the execution and release plans.
