# Tasks

## 1. Typed answer verification and bounded worker execution

- [x] 1.1 Define typed verification outcomes and preserve `answers_equal` as a compatibility wrapper; add tests proving only a completed positive verifier decision yields `correct` and infrastructure failures never yield `incorrect`.
- [x] 1.2 Implement a bounded, spawn-safe batch worker that disables math-verify's per-expression multiprocessing timeout, enforces a parent deadline, cleans up workers, and add Windows-oriented success/timeout/crash tests.
- [x] 1.3 Add Chinese documentation for verification states, platform behavior, claim limits, optional dependency, and offline operation; verify the documented protocol fields match the implementation.

## 2. Quality evidence integration

- [x] 2.1 Update stage-quality assessment to consume typed verification outcomes, report every outcome with denominator and coverage, and test unavailable/timeout/runtime-error cases remain distinct from verified incorrect answers.
- [x] 2.2 Bind verifier method/version and configuration into quality evidence identity; verify digest-stable reports for identical input and protocol, and changed digests when protocol metadata changes.

## 3. Opt-in semantic SFT calibration

- [x] 3.1 Add an explicit semantic-verification option to SFT calibration while keeping exact normalized matching as the default; test candidate selection only admits `correct` results in semantic mode.
- [x] 3.2 Record outcome counts, denominator, coverage, verifier identity/version, protocol configuration, and input/candidate digests in calibration reports/manifests; test semantic and exact modes produce distinct identities without overwriting existing artifacts.
- [x] 3.3 Update the Chinese SFT calibration guide with offline semantic-mode usage and its non-authoritative status; run its documented help/command examples in a temporary output location.

## 4. Integration verification

- [x] 4.1 Run the project test suite for answer verification, quality review, calibration, and CLI paths; run Ruff on all touched Python files and verify strict OpenSpec validation.
- [x] 4.2 When the pinned DeepMath snapshot is locally available, run a bounded Windows offline calibration comparison for 1,536 and 2,048 tokens; verify worker cleanup, explicit result denominators, and no writes to official Mini data or smoke recipes.
