"""Versioned Mini-data readiness profile contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class MiniDatasetProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_key: str = Field(min_length=1)
    stage: Literal["cpt", "cpt_holdout", "sft", "preference", "preference_eval", "grpo", "eval"]
    required: bool = True
    budget_required: bool = True
    target: int | None = Field(default=None, ge=1)
    unit: Literal["tokens", "examples", "preference_pairs", "prompts"]
    count_source: str = Field(min_length=1)
    manifest_path: str | None = None
    quality_report_required: bool = True
    processing_report_required: bool = True
    complete_lineage_required: bool = True

    @model_validator(mode="after")
    def validate_budget(self) -> MiniDatasetProfile:
        if self.budget_required and self.target is None:
            raise ValueError("A required budget must declare a positive target")
        if not self.budget_required and self.target is not None:
            raise ValueError("A supporting split without a budget must use target: null")
        if self.budget_required and not self.required:
            raise ValueError("A budgeted dataset must be required")
        return self


class MiniDataProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    profile_id: str = Field(min_length=1)
    profile_version: str = Field(min_length=1)
    description: str
    datasets: list[MiniDatasetProfile] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_keys(self) -> MiniDataProfile:
        keys = [dataset.dataset_key for dataset in self.datasets]
        if len(keys) != len(set(keys)):
            raise ValueError("Mini profile dataset_key values must be unique")
        return self


DimensionStatus = Literal["pass", "warning", "fail", "missing", "invalid", "not_evaluated", "not_required"]


class ReadinessIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    path: str | None = None


class ReadinessDimension(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: DimensionStatus
    summary: str
    observed_count: int | None = Field(default=None, ge=0)
    target: int | None = Field(default=None, ge=1)
    unit: str | None = None
    evidence: list[str] = Field(default_factory=list)
    issues: list[ReadinessIssue] = Field(default_factory=list)


class MiniDatasetReadiness(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_key: str
    dataset_id: str | None = None
    stage: str
    usage: Literal["baseline", "exploratory", "intentionally_noisy", "legacy_unknown"] = "legacy_unknown"
    required: bool = True
    budget: ReadinessDimension
    quality_coverage: ReadinessDimension | None = None
    quality_evidence: ReadinessDimension
    quality_review: ReadinessDimension
    processing_evidence: ReadinessDimension
    lineage: ReadinessDimension
    overall_status: Literal["ready", "not_ready", "not_evaluated"]
    blockers: list[ReadinessIssue] = Field(default_factory=list)
    warnings: list[ReadinessIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ready_state(self) -> MiniDatasetReadiness:
        dimensions = (self.budget, self.quality_coverage, self.quality_evidence, self.quality_review,
                      self.processing_evidence, self.lineage)
        if self.overall_status == "ready" and (
            self.blockers or any(d is not None and d.status not in {"pass", "warning", "not_required"} for d in dimensions)
        ):
            raise ValueError("ready requires all dimensions to pass, warn, or be not_required and no blockers")
        if self.overall_status == "not_evaluated" and not any(d.status == "not_evaluated" for d in dimensions):
            raise ValueError("not_evaluated requires at least one unevaluated dimension")
        return self


class MiniReadinessReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0", "1.1"] = "1.1"
    profile_id: str
    profile_version: str
    profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assessment_id: str = Field(min_length=1)
    artifact_root: str
    created_at: str
    datasets: list[MiniDatasetReadiness]
    overall_status: Literal["ready", "not_ready", "not_evaluated"]
    model_effect_status: Literal["not_evaluated"] = "not_evaluated"
    blockers: list[ReadinessIssue] = Field(default_factory=list)
    warnings: list[ReadinessIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_report_ready_state(self) -> MiniReadinessReport:
        required = [item for item in self.datasets if item.required]
        if self.overall_status == "ready" and (
            self.blockers or any(item.overall_status != "ready" for item in required)
        ):
            raise ValueError("ready report requires all required datasets to be ready and no blockers")
        if self.overall_status == "not_evaluated" and not any(
            item.overall_status == "not_evaluated" for item in required
        ):
            raise ValueError("not_evaluated report requires a required dataset with unevaluated evidence")
        return self


def load_mini_profile(path: str | Path) -> tuple[MiniDataProfile, str]:
    """Load a profile and return its byte-level SHA256 for provenance."""
    profile_path = Path(path)
    raw = profile_path.read_bytes()
    payload = yaml.safe_load(raw.decode("utf-8-sig"))
    return MiniDataProfile.model_validate(payload), hashlib.sha256(raw).hexdigest()
