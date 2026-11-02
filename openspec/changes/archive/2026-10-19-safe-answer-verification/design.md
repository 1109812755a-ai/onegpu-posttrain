# Design

## Context

See `proposal.md` for motivation. The repository already has `extract_answer`, exact normalization, `answers_equal`, and a typed `AnswerAssessment`, but the latter collapses several verifier exceptions into `incorrect` or lets platform errors escape. The installed `math-verify==0.9.0` uses multiprocessing for its default timeout on Windows; calling its default `parse` and `verify` for every nonliteral comparison creates repeated child processes. SFT calibration currently reports exact-string agreement only and uses that rule to select candidate records.

The repository is local-first and optional dependencies must remain optional. Calibration artifacts are exploratory, and official Mini quality approval remains a separate human-reviewed gate.

## Goals / Non-Goals

**Goals:**
- Return explicit per-answer outcomes, including verifier unavailable, unsupported reference, parse failure, timeout, and worker/runtime failure.
- Bound verifier work on Windows and other supported platforms without relying on math-verify's per-expression process timeout.
- Preserve the current exact-match behavior and dataset selection as the default; enable a versioned opt-in semantic mode for calibration.
- Bind verification reports and candidate manifests to input content, method/version, coverage, denominators, and outcome counts.
- Ensure verification evidence never claims reasoning quality or independently approves Mini data.

**Non-Goals:**
- Changing existing smoke length defaults or official Mini targets.
- Automatically accepting semantically verified examples into official Mini data.
- Replacing math-verify or adding a required service/dependency.
- Verifying proof validity, reasoning completeness, or model capability.
- Running training, benchmark evaluations, or network access as part of verification.

## Decisions

### Use a typed result as the authoritative interface

Introduce a result model for answer assessment with stable statuses such as `correct`, `incorrect`, `unparseable`, `unsupported`, `unverified`, `timeout`, and `verifier_error`. Exact normalized agreement is a fast positive result. A nonliteral result becomes `incorrect` only when the verifier completed successfully and explicitly returned false. Missing optional dependencies, unsupported references, parsing ambiguity, worker exceptions, and timeouts remain non-decision outcomes. Keep the existing boolean helper as a compatibility wrapper that returns true only for `correct`; quality reports and calibration use the typed result.

Alternative considered: continue using a boolean and infer error state from exceptions. Rejected because `False` cannot distinguish a verified mismatch from unavailable evidence and exception paths already differ across operating systems.

### Run verification in bounded project-owned worker batches

Send a bounded list of text/reference pairs to a top-level, spawn-safe worker entrypoint. Inside that worker, call math-verify with its own parser/comparison timeout disabled so it does not spawn a child for each expression. The parent enforces a configured wall-clock deadline for each bounded batch, terminates and joins a timed-out worker, and marks the affected batch `timeout`. Worker imports and payloads remain module-level/pickleable for Windows spawn. Process cleanup runs on success, error, interruption, and timeout. Batch size and deadline are explicit protocol/config values and are recorded in the report.

Alternative considered: invoke math-verify's default timeout per answer. Rejected because its Windows implementation spawns nested per-call processes and produced repeated invalid-handle failures. Disabling all timeouts without a parent deadline was also rejected because symbolic parsing can hang on pathological inputs. A thread timeout is not sufficient because it cannot safely stop a stuck Python worker.

### Keep semantic screening opt-in and protocol-bound

Add a calibration option with exact matching as the default and semantic verification as an explicit mode. Semantic mode may select a candidate trace only when the typed verifier result is `correct`; all other outcomes are excluded from candidate training data but retained in the report. The candidate manifest includes the answer-verification protocol/version and verifier identity, and the report includes counts and denominators for all outcomes. This makes output identities differ when selection semantics differ.

Alternative considered: silently expand candidates using math-verify in the existing default. Rejected because it changes eligible data and is not reproducible for existing calibration IDs or environments without the optional dependency.

### Keep evidence additive and non-authoritative

Expose typed counts in SFT quality assessment and calibration evidence while preserving existing quality-review and Mini readiness dimensions. A verifier failure does not count as `incorrect`; a positive answer result does not certify reasoning quality or grant human approval. Existing artifacts without the new evidence continue to report it as unavailable/not measured instead of being rewritten.

Alternative considered: make answer-verification success a new automatic Mini gate. Rejected because final-answer equivalence does not measure reasoning correctness and the project requires human review for official data acceptance.

## Risks / Trade-offs

- [A bounded batch timeout can mark otherwise valid answers as unverified] → Keep batches small and expose timeout counts, batch size, and deadline; never convert timeouts into incorrect labels.
- [A worker crash can affect several records] → Limit batch size, preserve completed prior batches, and report the exact affected batch/record IDs for retry.
- [Disabling math-verify's internal timeout allows a worker to hang] → The parent supervisor owns the deadline and forcibly terminates and joins the worker.
- [Semantic equality can accept mathematically equivalent final answers whose reasoning is flawed] → State the claim boundary in every report and keep reasoning-quality review separate.
- [Optional dependency/version differences affect results] → Bind method, package version, configuration, and input digest into candidate/report provenance.
- [Worker startup adds latency on Windows] → Process multiple records per bounded batch rather than spawning once per answer.

## Migration Plan

1. Add typed verification outcomes and safe bounded worker execution while retaining the existing boolean helper behavior for callers that need it.
2. Update quality assessment to count each outcome separately and add versioned verifier evidence.
3. Add an opt-in semantic mode to SFT calibration; keep exact matching as the default and leave existing reports/manifests untouched.
4. Run deterministic unit tests and a small local Windows batch using cached inputs only; verify worker cleanup, timeout/error states, and stable report digests.
5. Re-run the 2,000-row DeepMath sample in a new calibration directory under semantic mode. Compare eligible examples and outcome counts with the existing exact-mode artifacts; do not overwrite them.
6. Rollback is additive: disable semantic mode and use the exact default. Existing dataset artifacts, training recipes, and model checkpoints are not migrated.

## Open Questions

None. Initial batch-size and deadline defaults can be conservative implementation constants recorded in the protocol and adjusted later through a separate versioned change if evidence warrants it.
