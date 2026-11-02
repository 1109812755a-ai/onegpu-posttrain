# Proposal

## Why

The SFT length calibration currently uses exact normalized answer-string agreement as a conservative filter, so mathematically equivalent answers with different notation remain unresolved. An attempt to assess these cases exposed that `math-verify` 0.9.0 uses per-call multiprocessing timeouts on Windows, causing worker/handle failures; verifier infrastructure failures must not be confused with incorrect answers or silently block reproducible quality evidence.

## What Changes

- Add a bounded, cross-platform answer-verification capability with explicit outcomes for correct, incorrect, unparseable, unsupported, unavailable, timed-out, and verifier-error cases.
- Ensure verifier timeout/runtime failures remain unverified evidence, never negative correctness labels; record verifier identity/version and outcome denominators.
- Add opt-in semantic answer verification to SFT length calibration while retaining exact normalized matching as the default and preserving source, sample, tokenizer, and length comparison controls.
- Keep calibration output exploratory and separate from official Mini readiness, model-quality claims, and the existing smoke recipe.

## Capabilities

### New Capabilities
- `answer-verification`: Bounded, auditable final-answer equivalence checks with explicit outcomes and platform-safe execution.

### Modified Capabilities
- `sft-length-calibration`: Support optional semantic answer verification and report its coverage without changing the default eligibility rule.
- `mini-dataset-readiness`: Preserve verifier errors/timeouts/unavailable results as unverified evidence rather than incorrect answers or approval.

## Impact

- Affected code: `evaluation/answers.py`, `synthetic/verification.py`, `data/quality_review.py`, and `data/sft_calibration.py` plus the calibration CLI and tests.
- Uses the existing optional `math-verify[antlr4_13_2]==0.9.0` dependency; no mandatory new service or network access.
- Adds versioned report/manifest evidence for verifier identity, counts, denominator, coverage, and limitations. No model training, benchmark execution, official dataset promotion, or default recipe change is included.
