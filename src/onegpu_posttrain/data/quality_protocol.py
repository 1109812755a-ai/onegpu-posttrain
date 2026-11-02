"""Versioned, stage-aware catalog for dataset quality evidence."""

from __future__ import annotations

from dataclasses import dataclass

QUALITY_PROTOCOL_ID = "stage-data-quality-v1"


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    metric_class: str
    unit: str
    method: str
    description: str


_COMMON = (
    MetricDefinition("schema_valid_rate", "dataset_quality", "ratio", "canonical_schema:v1",
                     "Records conforming to the stage's canonical schema."),
    MetricDefinition("duplicate_record_ids", "dataset_quality", "records", "exact_record_id:v1",
                     "Repeated record identifiers in the assessed split."),
    MetricDefinition("train_eval_overlap", "dataset_quality", "unique_problems", "build_global_exact_id_intersection:v1",
                     "Known exact overlap with held-out data in the source build."),
    MetricDefinition("near_duplicate_rate", "dataset_quality", "ratio", "not_run",
                     "Near-duplicate estimate; unavailable unless a named checker is run."),
)

_DEFINITIONS = {
    "semantic_correctness": MetricDefinition(
        "semantic_correctness", "dataset_quality", "ratio", "independent_verifier",
        "Correctness evidence from a separately identified verifier, within its supported coverage."),
    "upstream_label_eligibility_rate": MetricDefinition(
        "upstream_label_eligibility_rate", "dataset_quality", "ratio", "source_correctness_metadata:v1",
        "Fraction carrying the source builder's correctness ordering labels; not fresh verification."),
    "answer_parseability_rate": MetricDefinition(
        "answer_parseability_rate", "dataset_quality", "ratio", "answer_parser",
        "Fraction whose final answer is parseable under a named answer parser."),
    "verifier_support_coverage": MetricDefinition(
        "verifier_support_coverage", "dataset_quality", "ratio", "verifier_support_registry",
        "Fraction of prompts/references supported by the named reward verifier."),
    "scoring_coverage": MetricDefinition(
        "scoring_coverage", "dataset_quality", "ratio", "evaluation_scoring_protocol",
        "Fraction of evaluation examples scored under the declared protocol."),
}

_CPT_COMMON = tuple(
    MetricDefinition(item.metric_id, item.metric_class, "documents", item.method, item.description)
    if item.metric_id == "train_eval_overlap" else item
    for item in _COMMON
)

STAGE_METRIC_CATALOG: dict[str, tuple[MetricDefinition, ...]] = {
    "cpt": (*_CPT_COMMON, _DEFINITIONS["semantic_correctness"]),
    "cpt_holdout": (*_CPT_COMMON, _DEFINITIONS["semantic_correctness"]),
    "sft": (*_COMMON, _DEFINITIONS["semantic_correctness"],
            _DEFINITIONS["upstream_label_eligibility_rate"], _DEFINITIONS["answer_parseability_rate"]),
    "preference": (*_COMMON, _DEFINITIONS["semantic_correctness"],
                   _DEFINITIONS["upstream_label_eligibility_rate"]),
    "preference_eval": (*_COMMON, _DEFINITIONS["semantic_correctness"],
                        _DEFINITIONS["upstream_label_eligibility_rate"]),
    "grpo": (*_COMMON, _DEFINITIONS["semantic_correctness"], _DEFINITIONS["verifier_support_coverage"]),
    "eval": (*_COMMON, _DEFINITIONS["scoring_coverage"]),
}


def metric_catalog_for(stage: str) -> tuple[MetricDefinition, ...]:
    """Return the required evidence inventory for a supported dataset stage."""
    try:
        return STAGE_METRIC_CATALOG[stage]
    except KeyError as exc:
        raise ValueError(f"Unsupported quality assessment stage: {stage}") from exc
