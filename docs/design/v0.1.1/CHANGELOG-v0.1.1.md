# Design Changelog — v0.1 → v0.1.1

## P0

- Rename to OneGPU-PostTrain.
- Separate PEFT-CPT from Full-CPT.
- Add Verifiable Math Core + Length Eligibility Gate.
- Move benchmark decontamination before split.
- Add high-quality length-controlled preference pairs.
- Freeze SimPO-original (`cpo_alpha=0`).
- Rework GRPO batch semantics and standard/lowmem profiles.
- Fix stage merge boundary.
- Add common policy pair evaluator.
- Enforce revision pinning + artifact hashes.

## P1

- Add General Text PPL.
- Treat FineMath-only CPT as domain stress test.
- Use nested CPT subsets.
- Use document-level CPT holdout.
- Version tokenizer/prompt templates.
- Separate answer correctness/parse/format.
- Make GRPO KL profile-dependent.
- Ban cross-algorithm loss comparison.
- Use accuracy@1 as deterministic main metric.
- Add license attribution to Release Gate.
- Freeze SFT 3-seed representative test.
- Add explicit experiment budget_type.
