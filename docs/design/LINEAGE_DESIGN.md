# Data and Model Lineage Design

**Version:** Lineage v1 (design)  
**Status:** Proposed contract for review; implementation is not yet complete  
**Scope:** file-based provenance for datasets, runs, models/checkpoints and evaluations

The connected design for per-step processing evidence and downstream data-effect comparisons is [Data Processing, Quality Change, and Model Effect Design](DATA_PROCESSING_EFFECT_DESIGN.md).

## Goal

Make every experiment answerable from local, inspectable artifacts:

- Which source records and transformations produced the data consumed by a run?
- Which parent model, recipe/configuration, code and environment produced a checkpoint?
- Which model, evaluation dataset and protocol produced a reported metric?
- Can sibling experiments be shown to share the same parent and differ only in the declared OVAT variable?

This is a reproducibility contract for the open-source single-GPU experiment lab, not a metadata platform. v1 uses JSON manifests, stable IDs, content hashes and relative artifact references. No database, registry service or generic event store is required.

## Core objects

| Object | Identity and purpose |
|---|---|
| `DatasetArtifact` | Immutable source, transformed dataset, split, or framework-adapter output; has an ID, provenance, record-count semantics and content digest. |
| `Recipe` | Versioned recipe/template and the resolved configuration used for a particular run. |
| `CodeRevision` | Repository commit when available, plus dirty/source snapshot information; unknown values remain explicit. |
| `Run` | One execution attempt consuming one or more model/data inputs and producing logs, metrics and zero or more outputs. A failed/OOM attempt is still a run. |
| `ModelArtifact` | Base model, adapter checkpoint, or merged stage model; records its kind, immutable identity, parent(s) and content/path digest status. |
| `EvaluationArtifact` | Evaluation of a specific model against a dataset/protocol, with metrics, denominators and report/prediction references. |

The graph is a DAG. `Run` connects input data/model artifacts to outputs. Evaluation is a separate operation that consumes a `ModelArtifact`, evaluation `DatasetArtifact` and protocol, and produces an `EvaluationArtifact`.

```text
Raw Source → Normalized → Filtered / Split Dataset ─┐
                                                    ├→ Run → Adapter Checkpoint → Merge → Model
Base Model / Parent Model ──────────────────────────┘                         │
                                                                              └→ Evaluation
```

For the v0.1 model chain, DPO, SimPO and both GRPO profiles must be represented as sibling runs consuming the same M2 model ID/hash—not as an implicit sequence through one another.

## Data lineage

Each materialized dataset/split needs a `dataset_manifest.json` (or the existing equivalent manifest upgraded compatibly) with at least the following provenance fields, and should link to its stage-specific `quality_report.json` when assessed. The report contract is defined in [Data Quality Design](DATA_QUALITY_DESIGN.md); it is deliberately a separate artifact so a dataset can be reassessed under a new protocol without rewriting its identity:

```json
{
  "schema_version": "1.0",
  "dataset_id": "ds-openr1-pref-smoke-v1",
  "parents": ["ds-openr1-normalized-v1"],
  "source": {
    "repo": "open-r1/OpenR1-Math-220k",
    "revision": "<immutable-revision-or-null>",
    "license": "<upstream-license-or-unknown>",
    "terms_ref": "<documentation-reference-or-null>"
  },
  "transform": {
    "name": "build_preference_pairs",
    "version": "1",
    "config_sha256": "<sha256-or-null>"
  },
  "filters": [
    "math_verify_correct_chosen",
    "math_verify_wrong_rejected",
    "complete_reasoning",
    "length_ratio_0.5_2.0"
  ],
  "split": "train",
  "record_count": {"value": 200, "unit": "preference_pairs"},
  "content_digest": {"algorithm": "sha256", "value": "<digest-or-null>", "status": "computed"},
  "processing_report": {"path": "reports/openr1_data_report.json", "status": "available"},
  "quality_report": {"path": "quality_report.json", "quality_report_id": "dq-openr1-pref-smoke-v1", "status": "available"},
  "created_at": "<RFC3339 timestamp>",
  "code_revision": {"commit": null, "dirty": null}
}
```

Requirements:

- Dataset IDs are immutable artifact identifiers, not claims that two builds have identical contents. The content digest verifies contents; a rebuild that changes content gets a new ID/version.
- Record-count units are mandatory (`raw_rows`, `examples`, `preference_pairs`, `prompts`, or `tokens`); never compare unlike units.
- Record a source revision or explicitly mark it unknown. Include upstream license/terms provenance; an open repository does not grant permission to redistribute third-party data or weights.
- Record transformations, versions, filter names/versions, relevant config digest, split and parent dataset IDs.
- Record contamination/overlap checks, requested-vs-accepted counts and rejection counters in the manifest or a linked report.
- Link a versioned data-build report with ordered processing steps, input/output counts and comparable before/after quality evidence. Each materialized/reused intermediate dataset is a `DatasetArtifact` with parent IDs; an in-memory processing step may instead use logical pool references without inventing a dataset identity. See [Data Quality Design](DATA_QUALITY_DESIGN.md#processing-trace-and-beforeafter-quality).
- Link quality evidence by report ID/path; do not embed an unversioned single quality score in the dataset identity.
- Link framework-specific adapter files back to the canonical dataset artifact they were generated from.

### Record-level lineage

Where a transformation can preserve ancestry, each emitted record should carry stable `record_id`, task-level `problem_id` where applicable, source-row identity and parent record IDs. Preference pairs should retain both generation/response record IDs, so a pair can be traced to its problem and chosen/rejected candidates.

```json
{
  "record_id": "pref_<stable-id>",
  "problem_id": "problem_123",
  "parent_record_ids": ["generation_2", "generation_5"]
}
```

Use deterministic IDs derived from source identity and the versioned transformation, not row position alone. If record-level mapping is unavailable or too costly for a source, declare the lineage level as `dataset_only`; do not fabricate row ancestry. Keep large mappings in JSONL sidecars when inline records would be unwieldy.

## Run lineage

Each new run should have exactly one canonical `lineage.json` alongside its existing operational `run.json`. They serve different purposes:

- `run.json` remains the execution record: status, command, runtime, environment and produced logs, preserving compatibility with current artifacts.
- `lineage.json` is the normalized relationship record: stable IDs and references to all input/output artifacts, recipe/config and code revision. It must not duplicate mutable runtime fields from `run.json`.
- Existing artifacts without `lineage.json` remain valid legacy runs; a reader may derive a best-effort view from `run.json`, clearly labeling inferred/missing fields.

Example:

```json
{
  "schema_version": "1.0",
  "run_id": "run-dpo-mini-beta01-s42",
  "stage": "dpo",
  "status_ref": "run.json#/train/status",
  "inputs": {
    "models": [{"model_id": "model-m2-sft-v1", "sha256": "<parent-digest-or-null>"}],
    "datasets": [{"dataset_id": "ds-openr1-pref-mini-v1", "manifest_sha256": "<digest-or-null>", "role": "train"}]
  },
  "recipe": {
    "path": "recipes/mini/dpo.yaml",
    "resolved_config_path": "resolved_config.json",
    "resolved_config_sha256": "<digest-or-null>"
  },
  "code_revision": {
    "repository": "<repository-url-or-null>",
    "commit": "<commit-or-null>",
    "dirty": null,
    "source_snapshot_sha256": null
  },
  "outputs": {
    "models": [
      {"model_id": "model-m3-dpo-mini-beta01-s42-adapter-v1", "kind": "adapter", "path": "...", "sha256": null},
      {"model_id": "model-m3-dpo-mini-beta01-s42-merged-v1", "kind": "merged", "path": "...", "sha256": null}
    ],
    "evaluations": []
  }
}
```

Runs may consume multiple datasets/models and produce zero outputs (e.g. failed/OOM) or multiple outputs (adapter and merge). Paths are relative to a declared artifact root where possible. A path is a locator, not an identity; use IDs and digests for identity/integrity.

## Model and evaluation lineage

Distinguish three model artifact kinds:

1. `base`: external/pretrained model plus repository and immutable revision;
2. `adapter`: stage-specific adapter/checkpoint produced by a training run;
3. `merged`: standalone stage model produced by a merge/export operation.

Each model manifest/reference records `model_id`, `kind`, parent model IDs, producing run/operation, stage, path/URI, digest status, and base model/tokenizer revision where available. A merged artifact must link both to the adapter and its parent model. Do not assume a directory path or human label (such as `M2-SFT`) uniquely identifies bytes.

Each evaluation manifest records an `eval_id`, evaluated `model_id` and digest, evaluation dataset ID/revision, protocol/config digest, evaluator/parser/verifier versions, metric names/values/units/denominators, and report/prediction references. If an evaluation isn't attached to a training run, it still links to the model and protocol; `run_id` is optional.

For data-selection experiments, each training run consumes the exact baseline or cleaned dataset ID/digest. A comparison record references both run/model/evaluation chains and the common source pool, parent model, budget and evaluation protocol. This makes processing → dataset → run → model → evaluation navigable without assigning model outcomes to individual filter steps in the absence of a controlled experiment.

## Stable ID rules

- IDs are lowercase ASCII slugs with a type prefix: `ds-`, `run-`, `model-`, `eval-`.
- IDs are immutable and unique within their object type. Human-readable names/aliases are separate display fields.
- Include semantically useful dimensions only when stable and useful (stage, recipe variant, seed, dataset purpose/version); do not encode every tunable value into IDs.
- A changed dataset content, checkpoint content, or evaluation protocol creates a new artifact identity. A rerun is a new `run_id`, even if its recipe and seed match.
- Store hashes separately from IDs. Use SHA-256 for file/config digests; directory digests must use a documented deterministic algorithm over sorted relative paths and per-file hashes. Record `not_computed`/`unavailable` rather than a fake or empty hash.
- `git_commit` is nullable. For a source archive or dirty checkout, record that state and a source snapshot/diff digest when feasible; never invent a commit. The current workspace may not expose Git metadata, so existing smoke lineage can legitimately mark this unknown.

Suggested examples:

```text
ds-openr1-pref-mini-v1
run-dpo-mini-beta01-s42
model-m3-dpo-mini-beta01-s42-merged-v1
eval-m3-dpo-math500-protocol-v1
```

## Storage and compatibility

Keep artifacts file-based and colocated with their owners; don't move or duplicate large datasets/checkpoints just to satisfy a new directory layout. A future optional index can discover manifests from configured roots.

```text
data/<dataset-build>/dataset_manifest.json
runs/<run_id>/run.json                 # operational record (existing)
runs/<run_id>/lineage.json             # canonical run relationships (v1)
runs/<run_id>/evaluation/<eval_id>.json
```

All new lineage documents have `schema_version`. Readers must tolerate absent optional fields and unknown additive fields, reject unsupported major versions clearly, and never reinterpret missing values as zero. No historical artifact is rewritten in place; migrations create derived indexes/views or explicitly versioned new artifacts.

## Out of scope for v1

- Database, SQLite/PostgreSQL service, ML Metadata/MLflow replacement, or a generic event store;
- a universal relation-event record such as `subject_type/relation/object_type` (the graph can be derived from typed manifests; add an event log only if append-only audit or multi-writer needs emerge);
- automatic hashing of every huge file on every scan;
- mandatory record-level ancestry when upstream data lacks stable identities;
- registry UI, scheduler, remote artifact store or arbitrary filesystem crawling.

## Adoption gates

Lineage v1 should be implemented before Mini/OVAT results are treated as official:

1. Add schema definitions and validators for dataset manifest, run lineage, model reference and evaluation provenance.
2. Emit stable IDs and dataset manifests for new data builds; record sample/problem IDs and pair ancestry where available.
   Add the ordered processing trace to each official build report and link measured before/after quality snapshots when available; follow the [detailed processing/effect contract](DATA_PROCESSING_EFFECT_DESIGN.md).
3. Emit one `lineage.json` for every new run, including model/dataset inputs and adapter/merged outputs; preserve the existing `run.json` operational format.
4. Add lineage validation for missing parents, duplicate IDs, dangling references, cycles, hash-format/status errors and preference train/eval overlap.
5. Backfill only what can be proven for current smoke artifacts; label inferred or unknown fields instead of guessing.
6. Verify DPO, SimPO and GRPO variants all point to the same M2 parent hash and report lineage completeness in run summaries.

The v0.1.1 UI may render this graph read-only after these contracts stabilize. Later scheduling/database work is not required to adopt the v1 contract.
