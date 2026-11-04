"""Typed, file-based lineage primitives shared by data, training and evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

LINEAGE_SCHEMA_VERSION = "1.0"
RunType = Literal["data_build", "quality_assess", "train", "merge", "evaluate"]
ArtifactKind = Literal["dataset", "quality_report", "model", "evaluation", "processing_report", "file"]
Availability = Literal["available", "unavailable", "not_computed"]
RunStatus = Literal["running", "completed", "failed", "oom", "partial"]


class LineageModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class Digest(LineageModel):
    algorithm: Literal["sha256"] = "sha256"
    value: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    status: Availability
    reason: str | None = None

    @model_validator(mode="after")
    def validate_status(self):
        if (self.status == "available") != (self.value is not None):
            raise ValueError("available digest requires a value; unavailable digest must not have one")
        if self.status != "available" and not self.reason:
            raise ValueError("unavailable/not_computed digest requires a reason")
        return self


class ArtifactRef(LineageModel):
    id: str
    kind: ArtifactKind
    path: str | None = None
    revision: str | None = None
    digest: Digest | None = None
    role: str | None = None
    availability: Availability = "available"


class RunRecord(LineageModel):
    lineage_schema_version: Literal["1.0"] = LINEAGE_SCHEMA_VERSION
    run_id: str = Field(pattern=r"^run-[a-z0-9][a-z0-9-]*$")
    run_type: RunType
    status: RunStatus
    lineage_status: Literal["complete", "incomplete"] = "complete"
    run_label: str | None = None
    experiment_id: str | None = None
    targeted_plan_id: str | None = Field(default=None, pattern=r"^plan-targeted-[0-9a-f]{20}$")
    targeted_plan_version: int | None = Field(default=None, ge=1)
    started_at: str
    completed_at: str | None = None
    inputs: list[ArtifactRef] = Field(default_factory=list)
    outputs: list[ArtifactRef] = Field(default_factory=list)
    recipe_path: str | None = None
    resolved_config_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    training_memory_mode: dict[str, Any] | None = None
    code_revision: dict[str, Any] = Field(default_factory=lambda: {"status": "unavailable", "reason": "not captured"})
    lineage_warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_targeted_plan_identity(self):
        if (self.targeted_plan_id is None) != (self.targeted_plan_version is None):
            raise ValueError("targeted plan ID and version must be supplied together")
        return self


class DatasetLineage(LineageModel):
    lineage_schema_version: Literal["1.0"] = LINEAGE_SCHEMA_VERSION
    parents: list[str] = Field(default_factory=list)
    produced_by_run_id: str
    transform: dict[str, Any]
    source_evidence: dict[str, Any]
    record_provenance: Literal["record", "dataset_only"]
    count_unit: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    processing_report_id: str | None = None
    quality_report_ids: list[str] = Field(default_factory=list)


class QualityReportLineage(LineageModel):
    lineage_schema_version: Literal["1.0"] = LINEAGE_SCHEMA_VERSION
    quality_report_id: str
    dataset_id: str
    produced_by_run_id: str
    protocol_version: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ModelLineage(LineageModel):
    lineage_schema_version: Literal["1.0"] = LINEAGE_SCHEMA_VERSION
    model_id: str
    kind: Literal["base", "adapter", "merged", "weight_space_candidate"]
    path: str | None = None
    digest: Digest
    parent_model_ids: list[str] = Field(default_factory=list)
    adapter_id: str | None = None
    produced_by_run_id: str | None = None
    stage: str | None = None
    edit_artifact_path: str | None = None


class EvaluationLineage(LineageModel):
    lineage_schema_version: Literal["1.0"] = LINEAGE_SCHEMA_VERSION
    eval_id: str
    produced_by_run_id: str
    model_id: str
    model_produced_by_run_id: str | None = None
    targeted_plan_id: str | None = Field(default=None, pattern=r"^plan-targeted-[0-9a-f]{20}$")
    targeted_plan_version: int | None = Field(default=None, ge=1)
    dataset_ids: list[str] = Field(default_factory=list)
    dataset_artifacts: list[ArtifactRef] = Field(default_factory=list)
    protocol_version: str
    config_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    status: Literal["completed", "failed", "partial"]
    metrics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    experiment_id: str | None = None
    iteration: int | None = Field(default=None, ge=0)
    dataset_role: Literal["practice", "final_holdout"] | None = None
    candidate_id: str | None = None
    candidate_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    parent_candidate_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    target_model_revision: str | None = None
    processor_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    runtime_image: str | None = None
    runtime_versions: dict[str, str] = Field(default_factory=dict)
    inference_config: dict[str, Any] = Field(default_factory=dict)
    comparison_identity: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_targeted_plan_identity(self):
        if (self.targeted_plan_id is None) != (self.targeted_plan_version is None):
            raise ValueError("targeted plan ID and version must be supplied together")
        return self


def new_run_id(run_type: RunType, *, label: str | None = None) -> str:
    """Create a unique identifier for one real operation."""
    slug = re.sub(r"[^a-z0-9]+", "-", (label or run_type).lower()).strip("-")[:36]
    return f"run-{slug or run_type}-{uuid4().hex[:12]}"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible configuration independent of mapping key order."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_directory(path: str | Path, *, excluded_names: set[str] | None = None) -> str:
    """Hash a directory deterministically from sorted relative paths and file hashes."""
    root = Path(path).resolve(strict=True)
    excluded_names = excluded_names or {"lineage.json", "run.json", "environment.json"}
    digest = hashlib.sha256()
    for file_path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in excluded_names):
        relative = file_path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(file_path)))
        digest.update(b"\n")
    return digest.hexdigest()


def resolve_local_reference(root: str | Path, reference: str | Path) -> Path:
    """Resolve a local artifact path and reject references outside the configured root."""
    base = Path(root).resolve(strict=True)
    candidate = (base / Path(reference)).resolve(strict=True)
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"lineage path escapes artifact root: {reference}") from exc
    return candidate


def assert_unique_ids(records: list[dict[str, Any]], *, key: str = "run_id") -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record in records:
        value = record.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"missing required unique identifier: {key}")
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"duplicate {key}: {', '.join(sorted(duplicates))}")


def load_versioned_payload(path: str | Path, *, legacy_keys: set[str]) -> tuple[dict[str, Any], str]:
    """Load a JSON object and identify legacy lineage without reinterpreting old fields."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("lineage document must be a JSON object")
    version = payload.get("lineage_schema_version")
    if version is None:
        if legacy_keys.intersection(payload):
            return payload, "legacy"
        return payload, "incomplete"
    major = str(version).split(".", 1)[0]
    if major != "1":
        raise ValueError(f"unsupported lineage schema major version: {major}")
    return payload, "v1"
