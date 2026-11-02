# Spec Delta

## Purpose

The Runs workspace lets users find and inspect local training and evaluation executions, understand their outcomes, and follow available dataset and model provenance without starting or modifying work.

## ADDED Requirements

### Requirement: Discover recorded executions without changing them
The Runs workspace SHALL list locally recorded runs with their available identity, type, stage, status, time, recipe, and experiment label, while preserving unknown or legacy values as unavailable rather than guessing.

#### Scenario: List includes completed and failed runs
- **WHEN** the workspace contains completed, failed, partial, or interrupted run records
- **THEN** the list shows each valid execution with its recorded status and does not hide failures or infer success from existing output folders

#### Scenario: Legacy run lacks optional metadata
- **WHEN** a readable older run lacks lineage, recipe, or timing fields
- **THEN** the run remains discoverable and missing fields are labeled unavailable or legacy

### Requirement: Filter and open a stable run detail
Users SHALL be able to filter recorded runs and open a detail view addressed by the unique `run_id`, with navigation state preserved across refresh and return to the list.

#### Scenario: Filter runs by stage and status
- **WHEN** a user selects stage or status filters
- **THEN** only matching runs are shown and the selected filters remain represented in the page URL

#### Scenario: Open a run from a filtered list
- **WHEN** a user opens a run and returns to the list
- **THEN** the prior filters and list context are restored

### Requirement: Explain run evidence and lineage
Run Detail SHALL present available configuration, metrics with units and denominators, logs, output artifacts, evaluation state, and dataset/model/evaluation lineage; unavailable, partial, failed, and unresolved evidence SHALL remain distinguishable.

#### Scenario: Training run has linked evaluation
- **WHEN** a training run has one or more linked evaluations
- **THEN** the detail view identifies each evaluation and exposes its actual status, metrics, denominator, and lineage when available

#### Scenario: Evaluation failed before producing scores
- **WHEN** an evaluation run failed without metric output
- **THEN** the detail view shows the failure and available log/lineage links without displaying a fabricated score

#### Scenario: Provenance references are outside the browsed root
- **WHEN** a run references a parent artifact that is unavailable in the workspace
- **THEN** the UI marks that reference unresolved or external and still displays independently valid run information

### Requirement: Keep artifact discovery read-only and bounded
Run discovery and detail endpoints SHALL read only supported workspace artifacts, enforce workspace containment, reject symlinks and traversal, and bound scan counts, file sizes, and log output.

#### Scenario: Request attempts path traversal
- **WHEN** a client requests a run detail or artifact using a path outside configured workspace roots
- **THEN** the request is rejected and no outside file contents are returned

#### Scenario: Workspace contains oversized or malformed records
- **WHEN** discovery encounters an oversized or malformed run artifact
- **THEN** it skips or reports that record safely without failing the entire Runs page
