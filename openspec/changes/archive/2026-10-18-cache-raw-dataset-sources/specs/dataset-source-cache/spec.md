# Spec Delta

## Purpose

This capability gives data builders a verifiable local copy of a pinned upstream dataset split so repeated cleaning and eligibility experiments can reuse identical source rows without repeatedly contacting the remote dataset host.

## ADDED Requirements

### Requirement: Materialize a pinned source split
The system SHALL provide an explicit operation that writes the selected source repository, configuration, revision, and split to a local raw JSONL cache with a manifest identifying that source.

#### Scenario: Cache a pinned Hugging Face split
- **WHEN** a user requests caching for a dataset with repository, revision, configuration, and split
- **THEN** the system writes every source row to a local cache and records those exact source coordinates in its manifest

#### Scenario: Source cannot be fully fetched
- **WHEN** fetching fails before the selected split is fully consumed
- **THEN** the system reports the cache as incomplete and does not publish it as a reusable complete cache

### Requirement: Verify source cache before reuse
The system SHALL verify that a local source cache is complete and matches the requested source identity and content digest before allowing it to satisfy a build request.

#### Scenario: Reuse a matching cache without network access
- **WHEN** a build requests a source identity with a complete, valid matching local cache
- **THEN** the builder reads the cached raw rows and does not contact the remote source

#### Scenario: Reject a mismatched or modified cache
- **WHEN** a requested source identity differs from the cache manifest or the cached content digest is invalid
- **THEN** the system rejects the cache and identifies the mismatch instead of silently using it

### Requirement: Preserve cache provenance in build evidence
The system SHALL record whether source rows came from a verified local cache or a remote stream and link cache identity, source revision, row count, and content digest in build evidence.

#### Scenario: Build from a local source cache
- **WHEN** a dataset build consumes a verified cache
- **THEN** its processing report and dataset manifests identify the cache and retain the pinned upstream source identity

#### Scenario: Build directly from the remote source
- **WHEN** a build uses the existing streaming path without a source cache
- **THEN** the processing evidence records remote streaming as the acquisition mode and retains existing source revision fields

### Requirement: Keep source caches separate from processed datasets
The system SHALL store raw source caches separately from derived canonical datasets and SHALL NOT modify an existing complete cache when creating or refreshing another cache entry.

#### Scenario: Refresh a source cache
- **WHEN** a user explicitly refreshes a cache for an existing source identity
- **THEN** the system publishes a newly verified cache atomically and preserves the prior complete cache if the refresh fails

#### Scenario: Build derived datasets from a cache
- **WHEN** data cleaning or eligibility rules produce canonical datasets
- **THEN** processed outputs are written to the requested build root and the raw source cache remains unchanged
