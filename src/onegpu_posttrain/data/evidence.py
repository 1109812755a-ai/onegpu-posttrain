"""Versioned, local processing and quality evidence; no semantic scores are invented."""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..reporting.data_report import length_summary
from ..schemas import STAGE_MODELS
from .quality_protocol import QUALITY_PROTOCOL_ID, metric_catalog_for


class EvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, allow_inf_nan=False)


class Count(EvidenceModel):
    value: int = Field(ge=0)
    unit: str = Field(min_length=1)


class QualityMetric(EvidenceModel):
    value: int | float | None = None
    unit: str
    status: Literal["measured", "estimated", "not_measured", "not_applicable"]
    metric_class: Literal["dataset_quality", "build_diagnostic", "model_recipe_suitability"] = Field(alias="class")
    method: str
    numerator: int | None = Field(default=None, ge=0)
    denominator: int | None = Field(default=None, ge=0)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence(self):
        available = self.status in {"measured", "estimated"}
        if available != (self.value is not None):
            raise ValueError("Available metrics require a value; unavailable metrics require null")
        if (self.numerator is None) != (self.denominator is None):
            raise ValueError("Ratio counts must include both numerator and denominator")
        if self.unit == "ratio" and available:
            if self.denominator is None or self.denominator == 0:
                raise ValueError("Measured ratios require a positive denominator")
            if not 0 <= self.numerator <= self.denominator:
                raise ValueError("Ratio numerator must be within the denominator")
            if not math.isclose(self.value, self.numerator / self.denominator):
                raise ValueError("Ratio value does not match numerator/denominator")
        return self


def ratio(numerator: int, denominator: int, method: str, *, metric_class: str = "dataset_quality",
          limitations: list[str] | None = None) -> QualityMetric:
    notes = list(limitations or [])
    if not denominator:
        notes.append("Empty assessed population; the rate is not measured")
    return QualityMetric(
        value=numerator / denominator if denominator else None, unit="ratio",
        status="measured" if denominator else "not_measured", metric_class=metric_class,
        numerator=numerator, denominator=denominator, method=method, limitations=notes,
    )


def unavailable(method: str, reason: str, *, unit: str = "ratio", metric_class: str = "dataset_quality") -> QualityMetric:
    return QualityMetric(unit=unit, status="not_measured", metric_class=metric_class,
                         method=method, limitations=[reason])


class ProcessingStep(EvidenceModel):
    step_id: str
    order: int = Field(ge=1)
    operation: str
    operation_version: str = "1"
    kind: Literal["transform", "filter", "split", "verify", "sample", "tokenize", "export"]
    status: Literal["completed", "skipped"] = "completed"
    input_ref: str
    output_ref: str
    input_count: Count
    output_count: Count
    output_is_subset: bool = False
    rejected_count: int = Field(default=0, ge=0)
    first_rejection_reasons: dict[str, int] = Field(default_factory=dict)
    unselected_count: int = Field(default=0, ge=0)
    unselected_reasons: dict[str, int] = Field(default_factory=dict)
    metrics_before: dict[str, QualityMetric] = Field(default_factory=dict)
    metrics_after: dict[str, QualityMetric] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_accounting(self):
        if any(n < 0 for n in [*self.first_rejection_reasons.values(), *self.unselected_reasons.values()]):
            raise ValueError("Decision counters cannot be negative")
        if sum(self.first_rejection_reasons.values()) != self.rejected_count:
            raise ValueError("First-rejection reasons must reconcile with rejected_count")
        if sum(self.unselected_reasons.values()) != self.unselected_count:
            raise ValueError("Non-quality selection reasons must reconcile with unselected_count")
        if self.output_is_subset:
            if self.input_count.unit != self.output_count.unit:
                raise ValueError("Subset accounting requires equal count units")
            if self.input_count.value != self.output_count.value + self.rejected_count + self.unselected_count:
                raise ValueError("Subset step counts must reconcile")
        if self.status == "skipped" and (self.rejected_count or self.unselected_count or not self.limitations):
            raise ValueError("A skipped step must explain why and cannot reject/select records")
        return self


def step(order: int, name: str, kind: str, input_ref: str, output_ref: str,
         n_in: int, n_out: int, unit: str, *, output_unit: str | None = None,
         rejected: dict[str, int] | None = None, unselected: dict[str, int] | None = None,
         skipped: bool = False, before: dict[str, QualityMetric] | None = None,
         after: dict[str, QualityMetric] | None = None, details: dict[str, Any] | None = None,
         limitations: list[str] | None = None) -> dict[str, Any]:
    rejected = rejected or {}
    unselected = unselected or {}
    return ProcessingStep(
        step_id=f"s{order:02d}-{name}-v1", order=order, operation=name, kind=kind,
        status="skipped" if skipped else "completed", input_ref=input_ref, output_ref=output_ref,
        input_count=Count(value=n_in, unit=unit), output_count=Count(value=n_out, unit=output_unit or unit),
        output_is_subset=output_unit is None, rejected_count=sum(rejected.values()),
        first_rejection_reasons=rejected, unselected_count=sum(unselected.values()),
        unselected_reasons=unselected, metrics_before=before or {}, metrics_after=after or {},
        details=details or {}, limitations=limitations or [],
    ).model_dump(mode="json", by_alias=True)


class ReportReference(EvidenceModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    report_id: str
    # References in manifests are relative to the manifest's containing directory.
    relative_to: Literal["manifest"] = "manifest"


class ProcessingReport(EvidenceModel):
    schema_version: Literal["1.0"] = "1.0"
    build_id: str
    created_at: str
    processing_protocol: str
    source: dict[str, Any]
    tokenizer: dict[str, Any]
    benchmark_screening: dict[str, Any]
    config: dict[str, Any]
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_acquisition: dict[str, Any] = Field(default_factory=dict)
    code_revision: dict[str, Any] = Field(default_factory=lambda: {"commit": None, "status": "not_captured"})
    stop_reason: Literal["targets_met", "max_raw_rows", "source_exhausted", "request_budget_exhausted",
                         "uncertain_request_requires_explicit_review"]
    processing_steps: list[ProcessingStep]
    outputs: dict[str, dict[str, Any]]
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_steps(self):
        if len({s.step_id for s in self.processing_steps}) != len(self.processing_steps):
            raise ValueError("Duplicate step IDs")
        if [s.order for s in self.processing_steps] != list(range(1, len(self.processing_steps) + 1)):
            raise ValueError("Steps must have consecutive ordered positions")
        return self


class QualityCheck(EvidenceModel):
    check_id: str
    result: Literal["PASS", "FAIL", "NOT_EVALUATED"]
    severity: Literal["error", "warning"]


class QualityGate(EvidenceModel):
    summary_status: Literal["FAIL", "NOT_EVALUATED"]
    policy_id: str = "structural-evidence-v1"
    checks: list[QualityCheck]
    reason: str = "Structural checks are enforced; a complete semantic quality gate is not calibrated"


class EvidenceCoverage(EvidenceModel):
    status: Literal["complete", "incomplete", "legacy_unknown"]
    expected_metrics: int = Field(ge=0)
    reported_metrics: int = Field(ge=0)
    measured_metrics: int = Field(ge=0)
    estimated_metrics: int = Field(ge=0)
    not_measured_metrics: int = Field(ge=0)
    not_applicable_metrics: int = Field(ge=0)
    missing_metric_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self):
        if self.reported_metrics != (
            self.measured_metrics + self.estimated_metrics + self.not_measured_metrics + self.not_applicable_metrics
        ):
            raise ValueError("Reported evidence counts must reconcile with metric statuses")
        if self.status == "complete" and (
            self.missing_metric_ids or self.reported_metrics != self.expected_metrics
        ):
            raise ValueError("Complete evidence coverage requires every expected metric")
        if self.status == "incomplete" and not self.missing_metric_ids:
            raise ValueError("Incomplete evidence coverage must identify missing metrics")
        if self.status == "legacy_unknown" and self.expected_metrics != 0:
            raise ValueError("Legacy unknown coverage cannot claim an expected metric inventory")
        return self


class CheckerIdentity(EvidenceModel):
    name: str
    version: str


class QualityReport(EvidenceModel):
    schema_version: Literal["1.0", "1.1"] = "1.1"
    quality_report_id: str
    dataset_id: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    build_id: str
    run_id: str | None = None
    stage: str
    created_at: str
    quality_protocol: str = QUALITY_PROTOCOL_ID
    checker: CheckerIdentity | None = None
    assessment_scope: Literal["pre_training_data"] = "pre_training_data"
    metrics: dict[str, QualityMetric]
    summaries: dict[str, Any]
    gate: QualityGate
    limitations: list[str]
    evidence_coverage: EvidenceCoverage | None = None

    @model_validator(mode="after")
    def validate_schema_coverage(self):
        if self.schema_version == "1.1" and self.evidence_coverage is None:
            raise ValueError("Quality report schema 1.1 requires evidence_coverage")
        if self.schema_version == "1.1" and self.checker is None:
            raise ValueError("Quality report schema 1.1 requires checker identity")
        return self


def assess_dataset(rows: list[dict[str, Any]], *, stage: str, dataset_id: str,
                   content_sha256: str, build_id: str, tokenizer_name: str,
                   overlap: int | None) -> QualityReport:
    """Validate final records and report builder metadata without claiming fresh verification."""
    model = STAGE_MODELS["preference" if stage == "preference_eval" else "cpt" if stage == "cpt_holdout" else stage]
    valid = 0
    for row in rows:
        try:
            model.model_validate(row)
        except ValidationError:
            continue
        valid += 1
    ids = [r["id"] for r in rows if isinstance(r.get("id"), str)]
    definitions = {definition.metric_id: definition for definition in metric_catalog_for(stage)}
    metrics = {
        "schema_valid_rate": ratio(valid, len(rows), f"{model.__name__}:canonical-v1"),
        "duplicate_record_ids": QualityMetric(
            value=len(ids) - len(set(ids)), unit="records", status="measured",
            metric_class="dataset_quality", method="exact_record_id:v1"),
    }
    if stage == "sft":
        known_labels = [r.get("metadata", {}).get("correctness_math_verify") for r in rows
                        if isinstance(r.get("metadata", {}).get("correctness_math_verify"), bool)]
        metadata_ok = sum(label is True for label in known_labels)
    elif stage in {"preference", "preference_eval"}:
        known_labels = [r.get("metadata", {}) for r in rows
                        if isinstance(r.get("metadata", {}).get("chosen_correct"), bool)
                        and isinstance(r.get("metadata", {}).get("rejected_correct"), bool)]
        metadata_ok = sum(md.get("chosen_correct") is True and md.get("rejected_correct") is False
                          for md in known_labels)
    else:
        known_labels = None
        metadata_ok = 0
    if "upstream_label_eligibility_rate" in definitions:
        if known_labels is None:
            metrics["upstream_label_eligibility_rate"] = unavailable(
                "source_correctness_metadata:v1", "This stage has no correctness-ordering label fields in its canonical records.")
        else:
            metrics["upstream_label_eligibility_rate"] = ratio(
                metadata_ok, len(known_labels), "source_correctness_metadata:v1",
                limitations=["Denominator includes only records with explicit boolean source labels; labels are selection-conditioned and are not fresh verifier execution"])

    for metric_id, definition in definitions.items():
        if metric_id in metrics or metric_id == "upstream_label_eligibility_rate":
            continue
        if metric_id == "train_eval_overlap":
            if overlap is None:
                metrics[metric_id] = unavailable(
                    definition.method, "No digest-verified build overlap evidence is available.",
                    unit=definition.unit, metric_class=definition.metric_class)
            else:
                metrics[metric_id] = QualityMetric(
                    value=overlap, unit=definition.unit, status="measured", metric_class=definition.metric_class,
                    method=definition.method,
                    limitations=["Counts exact IDs known to the builder; does not cover semantic or near-duplicate overlap"])
        else:
            reason = {
                "near_duplicate_rate": "No near-duplicate checker was run on this reassessment.",
                "semantic_correctness": "No independent semantic correctness verifier was run on this reassessment.",
                "answer_parseability_rate": "The retained canonical records do not contain per-record parser outcomes.",
                "verifier_support_coverage": "Canonical records do not identify a versioned verifier support registry.",
                "scoring_coverage": "Canonical evaluation records do not contain per-record scoring outcomes.",
            }.get(metric_id, "This metric is not measured by the current protocol.")
            metrics[metric_id] = unavailable(definition.method, reason, unit=definition.unit,
                                             metric_class=definition.metric_class)
    length_fields = ("tokens",) if stage.startswith("cpt") else ("rendered_tokens", "prompt_tokens", "chosen_tokens", "rejected_tokens")
    summaries = {
        "record_count": len(rows), "tokenizer": tokenizer_name,
        "lengths": {key: length_summary(r["metadata"][key] for r in rows if key in r.get("metadata", {})) for key in length_fields},
        "source_distribution": dict(Counter(str(r.get("source") or r.get("metadata", {}).get("source") or "<missing>") for r in rows)),
    }
    expected_ids = set(definitions)
    reported_ids = set(metrics)
    counts = Counter(metric.status for metric in metrics.values())
    evidence_coverage = EvidenceCoverage(
        status="complete" if expected_ids <= reported_ids else "incomplete",
        expected_metrics=len(expected_ids),
        reported_metrics=len(metrics),
        measured_metrics=counts["measured"],
        estimated_metrics=counts["estimated"],
        not_measured_metrics=counts["not_measured"],
        not_applicable_metrics=counts["not_applicable"],
        missing_metric_ids=sorted(expected_ids - reported_ids),
    )
    failed = valid != len(rows) or len(ids) != len(set(ids)) or overlap not in {0, None}
    quality_checks = [
        QualityCheck(check_id="canonical_schema", result="PASS" if valid == len(rows) else "FAIL", severity="error"),
        QualityCheck(check_id="unique_record_ids", result="PASS" if len(ids) == len(set(ids)) else "FAIL", severity="error"),
        QualityCheck(check_id="train_eval_separation",
                     result="NOT_EVALUATED" if overlap is None else "PASS" if overlap == 0 else "FAIL",
                     severity="error"),
    ]
    return QualityReport(
        quality_report_id=f"dq-{uuid4().hex}", dataset_id=dataset_id, content_sha256=content_sha256,
        build_id=build_id, stage=stage, created_at=datetime.now(timezone.utc).isoformat(), metrics=metrics,
        checker=CheckerIdentity(name="onegpu-posttrain-stage-quality-assessor", version="1.0.0"),
        summaries=summaries, gate=QualityGate(
            summary_status="FAIL" if failed else "NOT_EVALUATED",
            checks=quality_checks),
        limitations=["Assessment covers only the emitted dataset and the processed source prefix",
                     "Source labels and metadata do not establish independent semantic correctness",
                     "Benchmark coverage and GRPO rollout calibration remain in their source/build reports",
                     "No overall quality score is computed"],
        schema_version="1.1",
        evidence_coverage=evidence_coverage,
    )
