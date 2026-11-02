"""Typed domain objects for the source × training-stage dataset workbench."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..lineage import Digest

SourceKind = Literal["open_source", "local_upload", "model_generated", "dataset_mixture", "dataset_derived"]
TrainingStage = Literal["cpt", "sft", "preference", "grpo", "eval"]


class RegistryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataSource(RegistryModel):
    """A reusable origin; it is not itself a training dataset."""

    source_id: str = Field(min_length=1)
    kind: SourceKind
    name: str = Field(min_length=1)
    uri: str | None = None
    revision: str | None = None
    revision_status: Literal["known", "unknown", "not_provided"] = "not_provided"
    license_id: str | None = None
    license_status: Literal["verified", "user_declared_unverified", "unknown", "not_provided"] = "not_provided"
    attribution: str | None = None
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    declared_by: Literal["adapter", "user", "provider"]


class SourceStageCapability(RegistryModel):
    source_id: str = Field(min_length=1)
    source_kind: SourceKind
    supported_stages: list[TrainingStage]
    input_format: str = Field(min_length=1)
    requires_network: bool = False
    requires_credentials: bool = False
    status: Literal["available", "planned", "unsupported"] = "available"
    reason: str | None = None


class DatasetBuild(RegistryModel):
    """Evidence for one real source acquisition/transform into one training stage."""

    run_id: str = Field(pattern=r"^run-[a-z0-9][a-z0-9-]*$")
    source_id: str = Field(min_length=1)
    stage: TrainingStage
    protocol: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_digest: Digest
    output_dataset_id: str | None = None
    output_digest: Digest
    status: Literal["running", "completed", "failed", "partial"]


class DatasetArtifact(RegistryModel):
    """A persisted, versioned stage dataset that a training Run can consume."""

    dataset_id: str = Field(min_length=1)
    stage: TrainingStage
    source_id: str = Field(min_length=1)
    produced_by_run_id: str = Field(pattern=r"^run-[a-z0-9][a-z0-9-]*$")
    record_count: int = Field(gt=0)
    canonical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    native_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    quality_report_id: str | None = None
