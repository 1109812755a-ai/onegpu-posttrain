# Implementation Changelog

## 2026-10-19 — OpenR1 SFT Policy A/B (OpenSpec)

- Built and reviewed full-source offline A/B datasets from the same pinned OpenR1 pool and split; the only policy difference is correctness `filter` vs `report_only`.
- Pre-registered the experiment before training with hashed M1-CPT parent, recipe, seed, 20-step budget, held-out dataset and scoring protocol.
- Ran two independent 20-step RTX 4060 LoRA SFT branches from the common parent, then evaluated both on the same 20-example local holdout. The project comparator returned `COMPARABLE` with no compatibility issues.
- Observed accuracy A 0/20 vs B 1/20 (+0.05), parse rate 19/20 in both. This is a one-question, single-seed smoke observation—not evidence of statistically reliable or generalizable improvement.
- The `report_only` arm includes 327/694 upstream incorrect-labeled targets; independent semantic correctness remains unmeasured. Both datasets and models remain `exploratory` and are not eligible for Mini promotion or deployment.
- See [`final-summary.md`](../../artifacts/data-experiments/openr1-sft-policy-ab-20261019-01/final-summary.md), [`data-quality-comparison.md`](../../artifacts/data-experiments/openr1-sft-policy-ab-20261019-01/data-quality-comparison.md), and [`registration.json`](../../artifacts/data-experiments/openr1-sft-policy-ab-20261019-01/registration.json). Historical Mini artifacts, frozen gates and source cache were not targeted.

## 2026-10-18 — Data Policy Experiments (OpenSpec)

- Added versioned SFT/Preference data policies with exploratory/baseline use, filter/report-only rules, local tokenizer identity, source pool/split snapshots and offline preview/build CLI.
- Added experiment preregistration and conservative pipeline-effect/controlled-ablation comparison. Data quality, counts and coverage remain separate from model evaluation; incompatible evidence preserves raw scores without a comparable delta.
- Added a Data UI policy editor behind `serve-ui --enable-policy-editor`, bounded local preview, new draft/experiment plan writes, comparison-report browsing and Chinese operation guide. Training remains CLI-owned.
- Added an offline project-owned fixture walkthrough; it registers an experiment and emits `NOT_EVALUATED` without downloads, external model calls or GPU training.
- Mini data readiness remains **NOT READY**; this feature does not relax official quality/budget gates or establish a model improvement.

## Unreleased — 2026-10-16 Data Evidence (package remains dev6)

- Added Pydantic schemas for processing steps/reports, quality metrics/checks/reports and hashed manifest report references.
- Instrumented OpenR1 and FineMath without changing their selection rules; track first rejection, quota exclusions, explicit stop reason, available pre/post metrics and skipped benchmark checks.
- Both artifact builders emit JSON/Markdown processing and initial per-stage static quality reports. Dataset IDs include content/config/source/tokenizer/benchmark-index provenance; existing manifest filenames and legacy-reader support are retained.
- Expose upstream-label evidence separately from independent semantic correctness; empty rates and unmeasured checks remain unavailable. Structural failures are visible in reports and CLI acceptance.
- Added 11 meaningful regression cases; full CPU suite 65 passed. Touched data/report modules and related tests pass Ruff; full-repository lint has 36 remaining findings.
- See [Data Evidence implementation](../implementation/data/DATA_EVIDENCE.md). No new GPU training/model-effect result is claimed.

## 0.1.0.dev6

- Added GRPO-standard (`beta=0.04`, reference enabled) and GRPO-lowmem (`beta=0`, reference disabled) command contracts.
- Enforced completion-level batching: `per_device_train_batch_size=1`, `gradient_accumulation_steps=4`, `generation_batch_size=4`, `G=2`.
- Added explicit no-silent-truncation GRPO contract with `max_length=512`, `max_completion_length=512`, `truncation_strategy=delete`.
- Added ms-swift external reward plugin with `onegpu_accuracy` and project-specific `onegpu_boxed_format`; weights are `1.0 / 0.1`.
- Replaced regex-only boxed extraction with balanced-brace parsing so nested LaTeX such as `\\boxed{\\frac{1}{2}}` is handled consistently.
- Added monitored GRPO execution with PASS/OOM/FAILED artifacts and stage merge to `M3-GRPO-standard` / `M3-GRPO-lowmem`.
- Added `trainer_state.json` extraction for reward, reward std, zero-variance fraction, KL, completion length/clipping and per-reward metrics.
- Added GRPO profile comparison JSON/Markdown reports.
- Added whole-pipeline `smoke_summary.json/.md`.
- Added `grpo_smoke*.sh` and full `smoke_pipeline*.sh` orchestration.
- Added direct `python -m onegpu_posttrain.cli` entry support.
- Expanded tests from 38 to 50.
- Real RTX 4060 8GB execution completed on 2026-10-15: PEFT-CPT, Full-CPT probe, SFT, DPO, original SimPO, GRPO-standard and GRPO-lowmem all produced PASS artifacts and merged checkpoints where applicable.
- Base, M1-CPT and M2-SFT completed unified task evaluation; M2/DPO/SimPO completed common held-out preference scoring. M3 task evaluation remains outstanding.
- Recorded OpenR1 target shortfalls (197/200 SFT, 9/200 preference pairs, 3/50 preference-eval pairs) and high GRPO clipping in `runs/smoke/RUN_REPORT_zh.md`.
- Added the missing pinned `msgspec==0.22.0` training dependency after the real GRPO trainer exposed it.
- Final local suite: 54 passed; Ruff reported 45 findings and is scheduled for dev7 cleanup.

## Planned: 0.1.0.dev7 — Smoke Hardening (not yet implemented)

- Verify target-driven preference scanning to configured counts or explicit `max_raw_rows` / source exhaustion, with rejection counters and no weakened data gates.
- Calibrate GRPO completion lengths at 512 / 768 / 1024 / 1536 tokens and select a supported profile with clipping below 30% (aim below 20%).
- Run common task evaluation for DPO, original SimPO, GRPO-standard and GRPO-lowmem M3 checkpoints.
- Clear the 45 current Ruff findings and pass unit, compile and lint checks.
- Track acceptance evidence in `docs/project/milestones/v0.1-dev7-hardening.md`.

## 0.1.0.dev5

- Added a held-out `preference_eval.jsonl` derived only from the OpenR1 top-level holdout pool.
- Enforced zero `problem_id` overlap between preference training pairs and preference evaluation pairs.
- Added DPO LoRA smoke command compiler using `swift rlhf --rlhf_type dpo`, explicit `beta=0.1` and sigmoid loss.
- Added SimPO-original smoke command compiler using `beta=2.0`, `simpo_gamma=1.0`, and explicit `cpo_alpha=0.0`.
- Added DPO and SimPO latest-checkpoint merge to standalone `M3-DPO` and `M3-SimPO`.
- Added a framework-independent public preference evaluator based on response-token mean conditional log-probability.
- Added `policy_pair_accuracy` and `policy_pair_margin` on frozen held-out pairs.
- Added SFT / DPO / SimPO alignment comparison JSON and Markdown reports.
- Added alignment plan/execute scripts and implementation documentation.
- Expanded tests from 33 to 38.

## 0.1.0.dev4

- Added standard full-parameter CPT command compiler using `swift pt --tuner_type full`.
- Froze the diagnostic Full-CPT profile to five steps, FP16, sequence length 512, gradient checkpointing, and standard AdamW.
- Added monitored subprocess execution with sampled `nvidia-smi` GPU memory metadata.
- Added diagnostic `PASS / OOM / FAILED` semantics; OOM is preserved as a valid Full-CPT feasibility result.
- Added PEFT-CPT vs Full-CPT resource/feasibility JSON and Markdown reports.
- PEFT-CPT training now uses the same monitored execution contract for future apples-to-apples resource reporting.
- Added `swift sft` LoRA command compiler for merged `M1-CPT`.
- Froze SFT smoke to 20 steps, max length 1024, rank 8, alpha 32, batch 1 and gradient accumulation 8.
- Added SFT latest-checkpoint discovery, LoRA merge and explicit standalone `M2-SFT` output.
- Added SFT plan/execute scripts and stage-end evaluation / CPT-vs-SFT comparison orchestration.
- Added Full-CPT and SFT implementation documents.
- Expanded tests from 29 to 33.

## 0.1.0.dev3

- Added `swift pt` PEFT-CPT command compiler using pinned model revision and local ms-swift native data.
- Default smoke profile uses non-quantized LoRA; optional BNB 4-bit QLoRA fallback is retained but disabled.
- Added 20-step CPT smoke recipe for feasibility rather than effect-size claims.
- Added subprocess execution logging and runtime environment capture.
- Added deterministic latest `checkpoint-N` discovery.
- Added `swift export --merge_lora true` stage merge with explicit `M1-CPT` output directory.
- Added generic stage evaluation for Base or merged checkpoints.
- Added token-weighted FineMath holdout perplexity evaluation.
- Added EvalScope report discovery and top-level score collection.
- Added Base vs CPT comparison JSON/Markdown report.
- Fixed `adapt_smoke_data.sh` to match actual FineMath/OpenR1 builder output paths.
- Added `cpt_smoke_plan.sh` and real `cpt_smoke.sh` orchestration.
- Expanded tests from 21 to 29.

## 0.1.0.dev2

- Added Canonical -> ms-swift adapters for CPT, SFT, DPO/SimPO and GRPO.
- GRPO adapter preserves reward-side extra columns.
- Added adapter provenance report with SHA-256 hashes.
- Added EvalScope Base evaluation planner/runner for GSM8K + MATH-500.
- Added internal math holdout evaluator.
- Expanded tests from 13 to 21.

## 0.1.0.dev1

- Implemented FineMath and OpenR1 smoke data builders.
- Added decontamination, length gates, preference construction and data reports.
