# Proposal

## Why

The current objective-choice evaluator relies on exact answer parsing, which is reproducible but can misread verbose or semantically equivalent model responses. Users need an optional semantic review path while keeping existing scores stable and clearly distinguishing model-judge results from deterministic accuracy.

## What Changes

- Let users select deterministic exact matching or an Alibaba Bailian LLM judge for a new objective-choice evaluation run; exact matching remains the default.
- Reuse the configured DashScope OpenAI-compatible provider and local credential mechanism, with the configured judge model shown during preflight.
- Require explicit launch confirmation that sends evaluation questions, answer keys, and evaluated model outputs to Alibaba Cloud; show dataset size and expected request count before launch.
- Store per-question judge outcomes and reasons, distinguish undecided/provider-error records from incorrect answers, and report decision coverage alongside judge accuracy.
- Give LLM-judge runs a separate versioned protocol and comparability identity. Do not combine or rank their metrics with exact-match scores by default.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `unified-model-evaluation`: Add an explicitly selected, separately versioned Alibaba Bailian semantic-judging strategy for registered objective-choice suites while preserving exact matching as the default protocol.

## Impact

The Runs evaluation selector and preflight/start APIs, objective-choice evaluation runner and per-question report, evaluation lineage/configuration, and user documentation. Reuse the existing DashScope-compatible provider profile and credential handling; no provider secret is sent to the browser or persisted in evaluation artifacts.
