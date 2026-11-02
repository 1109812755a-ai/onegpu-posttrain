# Tasks

## 1. Local DeepMath Source and Deterministic Sampling

- [x] 1.1 Add an offline-only DeepMath Parquet reader/adapter with explicit cache path, required-column validation, stable source-row/trace IDs, and actionable missing-cache behavior; verify unit tests cover valid, malformed, and missing snapshots without network calls.
- [x] 1.2 Implement seeded hash-based bounded sampling across shards and persist source snapshot, sample identity, seed, and denominators; verify repeated runs select identical record IDs and do not depend on shard-head ordering.
- [x] 1.3 Document local-cache prerequisites and exploratory-only status in the data/calibration CLI guide; verify every documented command is offline and fails clearly when source data is absent.

## 2. Rendered-Length Candidate Bundles

- [x] 2.1 Reuse the pinned Qwen tokenizer and final SFT chat-template rendering to measure prompt-plus-trace lengths and report per-trace and distinct-problem distributions; verify fixture token counts against direct tokenizer rendering.
- [x] 2.2 Build separate 1,024/2,048/4,096 eligibility candidates without truncation and emit digest-bound manifests; verify boundary lengths, deterministic IDs, and distinct manifest/content hashes across different gates.
- [x] 2.3 Add calibration report schema for denominators, per-cap retention, sample coverage, tokenizer/template identity, and explicit unavailable fields; verify schema validation and JSON serialization tests.

## 3. Verification Coverage and Data-Quality Semantics

- [x] 3.1 Adapt supported DeepMath answer/trace records to existing verification interfaces without assuming unsupported answer semantics; verify correct, incorrect, malformed, and unsupported cases with fixtures.
- [x] 3.2 Report verifier coverage and measured/not-measured quality fields separately from length eligibility; verify a length-only run cannot label candidates as verified or quality-passing.
- [x] 3.3 Document interpretation limits and ensure calibration outputs cannot be written into or accepted as official Mini inputs; verify path isolation and Mini-readiness regression tests.

## 4. Bounded SFT Hardware Probes

- [x] 4.1 Add an explicit opt-in bounded probe for each candidate using the same parent, source pool/order, seed, and training controls, with unique outputs and a fixed update budget; verify default execution remains CPU-only and existing smoke recipe is unchanged.
- [x] 4.2 Capture outcome, duration, peak VRAM, device/environment identity, and OOM/failure details with unavailable states; verify success and simulated OOM paths preserve separate candidate records.
- [x] 4.3 Document that these probes measure hardware feasibility only, not model efficacy; verify the report and CLI help contain that limitation.

## 5. End-to-End Calibration Validation

- [x] 5.1 Run offline end-to-end calibration on a small owned fixture and validate all three candidate reports, manifests, and data-quality labels; verify repeatability and count reconciliation.
- [x] 5.2 Run OpenSpec validation and the targeted test suite, then confirm `recipes/smoke/20-sft.yaml` and official Mini artifacts are unchanged; record the exact commands and outcomes in the implementation summary.
