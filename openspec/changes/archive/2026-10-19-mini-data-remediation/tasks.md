# Tasks

## 1. Versioned review contracts

- [x] 1.1 Define stage-quality review profile/report schemas, metric dispositions, reviewer identity, and profile hashing; add fixtures that reject unversioned rules and composite scores, and document the profile format in Chinese.
- [x] 1.2 Add explicit dataset usage classification (`baseline`, `exploratory`, `intentionally_noisy`, `legacy_unknown`) to review/readiness interpretation without mutating historical manifests; test legacy defaults and all supported classifications.

## 2. Evidence-backed stage review

- [x] 2.1 Implement offline review generation bound to dataset ID, stage, canonical digest, evidence report ID, review profile, and checker identity; verify same inputs/protocol produce reproducible evidence content while every review gets a unique assessment ID.
- [x] 2.2 Add optional independent mathematical answer verification for supported SFT/Preference records with trusted references; record correct/incorrect/unparseable/unsupported/unverified counts and coverage, and test absent verifier, partial support, and label-vs-verification separation.
- [x] 2.3 Add explicit per-check review decisions with rationale, timestamp, reviewer/method identity, and supersession references; test that unsupported evidence cannot satisfy a required check and revised decisions preserve prior artifacts.

## 3. Readiness and remediation integration

- [x] 3.1 Add an additive review-bundle index and CLI command; validate report hashes, path containment, profile identity, dataset key/stage, dataset ID, and canonical digest, with negative tests for stale, tampered, duplicate, and escaping references.
- [x] 3.2 Extend `validate-mini-data` to consume the optional review bundle and require an approved matching review plus explicit official usage for official readiness; test exploratory/noisy/legacy artifacts remain non-ready even when budgets pass.
- [x] 3.3 Verify existing policy-built and targeted-generated candidates can be traced into review inputs using their current policy and generation lineage; test locally with owned fixtures only and never trigger a provider.
- [x] 3.4 Document review profiles, exploratory-vs-official interpretation, verifier limitations, and the offline CLI workflow in Chinese; execute all documented commands against project fixtures without network access.

## 4. Retained-data and integration verification

- [x] 4.1 Apply the review workflow to retained build-02 into new output directories; report reviewed checks, verifier coverage, legacy usage state, and remaining SFT/Preference budget shortfalls without changing the frozen Mini profile.
- [x] 4.2 Snapshot and compare hashes for all pre-existing build-02 datasets, manifests, reports, and readiness outputs; assert all new review records match exact input digests.
- [x] 4.3 Run focused review/readiness/policy/synthetic tests, Ruff, compile checks, and `openspec validate mini-data-remediation --strict`; record exact results and verify no GPU, remote provider, network, or external quality service is required.
