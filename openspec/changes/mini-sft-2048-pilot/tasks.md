# Tasks

## 1. Mini CPT recipe and parent model

- [x] 1.1 Add a versioned `recipes/mini/10-peft-cpt.yaml` using the existing 5,000,368-token FineMath Mini artifact, pinned Qwen3-0.6B-Base, LoRA, micro-batch 1, gradient checkpointing, and one complete pass over the dataset; add tests that resolve the expected source, parent, seed, and training budget without changing the Smoke recipe.
- [x] 1.2 Preflight local dataset/manifest/quality-review state, base checkpoint hash, free GPU resources, and resolved CPT config; record that the data is pending/legacy and the run must be exploratory, and verify missing or inconsistent inputs stop before training without downloads.
- [x] 1.3 Run the full 5M-token PEFT-CPT Mini pass from Base, merge the adapter into a distinct M1 artifact, and verify completion status, actual seen-token/update counts, peak VRAM, environment, checkpoint hash, and complete lineage; on OOM/failure preserve evidence and stop without automatic quantization or recipe relaxation. Evidence: `runs/mini/cpt-finemath-5m-s42-v3/` (PASS, 413 updates, 5,000,368 tokens, 7,055 MiB sampled peak) and merged M1 `run-m1-cpt-merge-d679a3f662ff`.

## 2. Deterministic DeepMath Mini SFT data

- [x] 2.1 Implement an offline DeepMath Mini SFT builder for pinned revision `5cf055d1fe3d7a2eb19719ac020211469736ae44`; verify missing snapshot/license evidence fails closed and manifests record source-file hashes, attribution/license reference, tokenizer/template revision, and content identity.
- [x] 2.2 Split by problem identity before selection, exclude known evaluation/holdout overlap, apply complete rendered 2,048-token eligibility without truncation, and deterministically select at most 5,000 training examples; test split isolation, boundary lengths, stable IDs, reproducible ordering/hashes, under-target shortfall, and rejection accounting.
- [x] 2.3 Emit a new immutable DeepMath SFT cohort with ordered processing report, canonical/native datasets, source-row/problem/trace provenance, hashes, quality evidence, and lineage; verify repeated builds reconcile counts and never modify calibration, OpenR1, or build-02 artifacts.
- [x] 2.4 Add `deepmath.sft_train` to a new versioned Mini readiness profile while preserving `openr1.sft_train`; test each cohort's count/review/lineage is reported independently and no implicit cross-source sum satisfies 5K.

## 3. SFT quality review and readiness

- [x] 3.1 Produce digest-bound SFT quality evidence for schema, duplicate IDs, rendered lengths, answer-check method/coverage, topic/difficulty distribution, and available contamination checks; verify exact answer agreement, semantic verification, and human reasoning review remain distinct measures with explicit denominators/statuses.
- [x] 3.2 Generate a new review bundle using the versioned SFT review profile; verify pending/missing human review, verifier, or contamination evidence leaves the cohort exploratory and Mini readiness not ready, while an approved review can only apply to the exact matching dataset digest.
- [x] 3.3 Add tests for exploratory usage, pending legacy CPT review, and independently reported DeepMath/OpenR1 SFT cohorts; document reviewer actions and ensure no readiness PASS is inferred from the 5K count alone.

## 4. Mini SFT training and unified evaluation

- [x] 4.1 Add a distinct `recipes/mini/20-sft.yaml` pinned to the new Mini CPT M1, 5K target cohort, 2,048 max length, micro-batch 1, LoRA rank 8, and gradient checkpointing; test resolved config and preserve `recipes/smoke/20-sft.yaml` at 1,024.
- [x] 4.2 Recheck GPU availability and run a bounded probe with the exact Mini SFT parent/config; verify OOM or insufficient resources records a stopped/failed probe and prevents full training without silent truncation, quantization, shorter length, or model substitution. Evidence: `runs/mini/sft-deepmath-2048-probe-v2/` PASS, 3 steps at 2,048 tokens, sampled peak 7,945/8,188 MiB; exact parent is v3 M1.
- [ ] 4.3 If the exact-config probe passes, run one complete pass over the selected SFT cohort from the new M1, merge to a distinct M2 artifact, and verify actual exposure, run status, peak VRAM, environment, parent/data/config hashes, and complete model lineage; keep run usage exploratory while required data reviews remain pending. Attempt `run-mini-sft-deepmath-5k-2048-s42-e3ee24cc8136` stopped with OOM at step 19/624; no checkpoint or M2 was produced. Do not mark complete until a separately reviewed feasible run succeeds.
- [ ] 4.4 Evaluate the new M1 and M2 under the same available unified GSM8K, MATH-500, internal-holdout, and domain-PPL protocol; verify shared evaluation identities/configs and report stage deltas, costs, data-quality state, and limitations without claiming a causal 2,048-vs-1,024 effect. M1 only is complete at `runs/mini/eval-m1-cpt-v1/`; M2 is blocked on SFT OOM.

## 5. Documentation and end-to-end validation

- [x] 5.1 Update Chinese Mini workflow/status documentation with the CPT→SFT parent chain, DeepMath source cohort, 5K/2,048 settings, pending-review semantics, 8GB stop conditions, and reproducible commands; verify every linked artifact and command exists.
- [x] 5.2 Run focused data-builder, quality-review, readiness, recipe/runner, lineage, and evaluation tests plus Ruff/compile checks; run `openspec validate mini-sft-2048-pilot --strict` and verify historical artifacts and Smoke recipes remain unchanged.
- [ ] 5.3 Write a machine-readable and Chinese summary linking source/data manifests, quality reviews, CPT/SFT experiment/run IDs, checkpoints, evaluation outputs, and resource evidence; verify lineage validation reports no errors and readiness is labeled according to actual review decisions. The summary is updated with CPT, probe, SFT OOM, and M1 evaluation evidence, but full SFT/M2/M2 evaluation and a clean end-to-end lineage validation remain incomplete.

## Workflow follow-up

- The existing FineMath CPT quality review is pending and its usage is `legacy_unknown`; keep all dependent runs exploratory unless a maintainer records an approved review bound to the exact dataset digest.
- The current 2,048-token RTX 4060 probe sampled 7,945/8,188 MiB; the full run later OOMed at step 19/624. Failure evidence is preserved in `runs/mini/sft-deepmath-5k-s42-v1/`; do not retry unchanged or silently alter length/precision/model.
- Archive the change only after implementation tasks and evidence review are complete; do not archive while a required task is blocked or quality evidence is misrepresented.
