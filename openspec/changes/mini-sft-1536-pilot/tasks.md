# Tasks

## 1. Bounded SFT feasibility

- [x] 1.1 Add a distinct 1,536-token probe recipe bound to the existing calibration manifest, M1-CPT, and seed 42; verify resolved inputs and hashes before launch.
- [x] 1.2 Run a maximum three-step GPU probe and record status, peak VRAM, duration, environment, and dataset identity; verify OOM/failure stops the change without full training.

## 2. Calibration dataset and model lineage

- [x] 2.1 Implement idempotent registration for the existing calibration candidate: verify source revision plus canonical/native SHA256, emit a canonical dataset manifest and truthful `data_build` registration Run without rewriting the JSONL; cover success, reuse, and mismatch cases with tests.
- [x] 2.2 Verify training preflight resolves canonical/native bytes to the registered dataset ID only on exact digest match, resolves the actual M1 model, enforces the measured max length without inventing a DataPolicy, and passes strict validation for the registered dataset root; actual adapter lineage is verified with the full run in 3.1.

## 3. Exploratory Mini SFT and evaluation

- [x] 3.1 Run the 1,536-token full-epoch attempt on the actual 4,820 examples; preserve its OOM at step 20/599, complete input/config/environment evidence, and absence of any checkpoint/adapter/M2; verify strict lineage reports the failure accurately.
- [x] 3.2 Derive a 1,024-token child cohort from the registered 1,536 dataset without upstream rescanning; verify exact length filtering, parent/child manifests, processing counts/rejection reasons, native/canonical hashes, and strict local lineage validation.
- [x] 3.3 Train one exploratory epoch on the derived cohort with a distinct 1,024 recipe and output directory; merge M2 only on successful training, record actual resources and status, and verify complete data/model lineage or truthful OOM/failure lineage.
- [x] 3.4 If M2 is produced, run the existing unified M1/M2 evaluation protocol; write a concise Chinese summary with actual count, exploratory/review limitations, and results; finish with strict lineage validation and OpenSpec strict validation.
