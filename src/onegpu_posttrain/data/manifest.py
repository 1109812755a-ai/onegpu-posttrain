from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..lineage import DatasetLineage
from .evidence import ReportReference
from .registry import DataSource, TrainingStage


class SourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Legacy source identity retained for existing source-specific builders.
    # User uploads may have no repository or immutable revision.
    repo: str | None = None
    config: str | None = None
    revision: str | None = None


class TokenizerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo: str
    revision: str


class DatasetParent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_id: str = Field(min_length=1, max_length=200)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality_report_id: str | None = None


class DatasetComposition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    protocol: Literal["dataset-mixture-v1"] = "dataset-mixture-v1"
    stage: TrainingStage
    strategy: Literal["concat", "weighted_sample"]
    parents: list[DatasetParent] = Field(min_length=2)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_record_count: int = Field(gt=0)
    seed: int | None = Field(default=None, ge=0)
    weights: dict[str, float] = Field(default_factory=dict)
    budget: int | None = Field(default=None, ge=1)
    provenance_path: str = "provenance/records.jsonl"
    provenance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("provenance_path")
    @classmethod
    def safe_provenance_path(cls, value: str) -> str:
        relative = _safe_artifact_relative_path(value)
        if not value.startswith("provenance/"):
            raise ValueError("composition provenance must be stored under provenance/")
        return relative.as_posix()

    @model_validator(mode="after")
    def consistent_strategy(self):
        ids = [parent.dataset_id for parent in self.parents]
        if len(ids) != len(set(ids)):
            raise ValueError("composition parent dataset IDs must be unique")
        if self.strategy == "concat":
            if self.seed is not None or self.budget is not None or self.weights:
                raise ValueError("concat composition cannot declare sampling parameters")
        elif self.strategy == "weighted_sample":
            if self.seed is None or self.budget != self.output_record_count:
                raise ValueError("weighted sample composition requires seed and matching output budget")
            if set(self.weights) != set(ids) or any(value <= 0 for value in self.weights.values()):
                raise ValueError("weighted sample composition requires positive weights for every parent")
        else:
            raise ValueError("unsupported dataset composition strategy")
        return self


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str
    schema_version: str = "1.0"
    stage: str
    source: SourceSpec | None = None
    origin: DataSource | None = None
    tokenizer: TokenizerSpec | None = None
    prompt_template_id: str | None = None
    protocol_versions: dict[str, str] = Field(default_factory=dict)
    split_seed: int = 42
    sample_count: int
    hashes: dict[str, str] = Field(default_factory=dict)
    processing_report: ReportReference | None = None
    quality_report: ReportReference | None = None
    lineage: DatasetLineage | None = None
    composition: DatasetComposition | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: str | Path) -> DatasetManifest:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return DatasetManifest.model_validate(raw)


def _safe_artifact_relative_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("manifest metadata.canonical_path must be a non-empty POSIX artifact-relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts) or ":" in relative.parts[0]:
        raise ValueError("manifest metadata.canonical_path must remain inside the dataset artifact root")
    return relative


def resolve_canonical_path(path: str | Path, manifest: DatasetManifest | None = None) -> Path:
    """Resolve and verify canonical records using the manifest's artifact-root-relative path."""
    manifest_path = Path(path).resolve(strict=False)
    manifest = manifest or load_manifest(manifest_path)
    artifact_root = manifest_path.parent.parent.resolve(strict=True)
    relative = _safe_artifact_relative_path(manifest.metadata.get("canonical_path"))
    candidate = artifact_root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("canonical records cannot be reached through a symbolic link")
    canonical = candidate.resolve(strict=True)
    try:
        canonical.relative_to(artifact_root)
    except ValueError as exc:
        raise ValueError("canonical records path escapes the dataset artifact root") from exc
    if not canonical.is_file():
        raise ValueError("canonical records path is not a regular file")
    expected = manifest.hashes.get("canonical_records_sha256")
    if not expected or sha256_file(canonical) != expected:
        raise ValueError("canonical data hash（canonical 数据摘要 / records SHA-256）does not match manifest")
    if manifest.lineage is not None and manifest.lineage.content_sha256 != expected:
        raise ValueError("canonical data hash（canonical 数据摘要 / records SHA-256）does not match dataset lineage")
    return canonical


def save_manifest(path: str | Path, manifest: DatasetManifest, *, allow_legacy: bool = False) -> None:
    """Atomically save a manifest; new dataset artifacts must reference verified canonical records.

    ``allow_legacy`` is reserved for compatibility fixtures and explicit preservation of old
    manifests which cannot yet satisfy the current artifact contract.
    """
    path = Path(path)
    if path.is_symlink():
        raise ValueError("manifest path cannot be a symbolic link")
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical_rel = manifest.metadata.get("canonical_path")
    if canonical_rel is None and not allow_legacy:
        raise ValueError("new dataset manifest must declare metadata.canonical_path")
    if canonical_rel is not None:
        resolve_canonical_path(path, manifest)
    payload = yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False, allow_unicode=True).encode("utf-8")
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def repair_manifest_canonical_path(path: str | Path) -> str:
    """Repair a legacy path only from its hash-verified processing-report output record."""
    supplied_path = Path(path)
    if supplied_path.is_symlink():
        raise ValueError("manifest path cannot be a symbolic link")
    manifest_path = supplied_path.resolve(strict=True)
    manifest = load_manifest(manifest_path)
    if manifest.processing_report is None:
        raise ValueError("cannot safely repair manifest without a processing report reference")
    artifact_root = manifest_path.parent.parent.resolve(strict=True)
    report_candidate = manifest_path.parent / manifest.processing_report.path
    if report_candidate.is_symlink():
        raise ValueError("processing report cannot be a symbolic link")
    report_path = report_candidate.resolve(strict=True)
    try:
        report_path.relative_to(artifact_root)
    except ValueError as exc:
        raise ValueError("processing report path escapes the dataset artifact root") from exc
    if sha256_file(report_path) != manifest.processing_report.sha256:
        raise ValueError("processing report digest does not match manifest")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    outputs = report.get("outputs")
    if isinstance(outputs, dict):
        output_records = [item for item in outputs.values() if isinstance(item, dict)]
    elif isinstance(outputs, list):
        output_records = [item for item in outputs if isinstance(item, dict)]
    else:
        raise TypeError("processing report has no dataset outputs")
    matches = [item for item in output_records if item.get("dataset_id") == manifest.dataset_id]
    if len(matches) != 1:
        raise ValueError("processing report does not uniquely identify this dataset output")
    output = matches[0]
    relative = _safe_artifact_relative_path(output.get("path"))
    expected_digest = manifest.hashes.get("canonical_records_sha256")
    if (not expected_digest or output.get("content_sha256") != expected_digest
            or manifest.lineage is None or manifest.lineage.content_sha256 != expected_digest):
        raise ValueError("processing report, manifest, and lineage do not share the same canonical digest")
    manifest.metadata["canonical_path"] = relative.as_posix()
    resolve_canonical_path(manifest_path, manifest)
    save_manifest(manifest_path, manifest)
    return relative.as_posix()


def validate_manifest_evidence(path: str | Path) -> list[str]:
    """Verify report digests and cross-links for a dataset manifest; legacy fields remain readable."""
    manifest_path = Path(path).resolve(strict=True)
    manifest = load_manifest(manifest_path)
    errors: list[str] = []
    artifact_root = manifest_path.parent.parent
    if manifest.composition is not None:
        composition = manifest.composition
        if composition.stage != manifest.stage or composition.output_record_count != manifest.sample_count:
            errors.append("composition stage/count does not match dataset manifest")
        parent_ids = [parent.dataset_id for parent in composition.parents]
        if manifest.lineage is None or manifest.lineage.parents != parent_ids:
            errors.append("composition parent IDs do not match dataset lineage")
        provenance_relative = _safe_artifact_relative_path(composition.provenance_path)
        provenance = artifact_root
        symlinked = False
        for part in provenance_relative.parts:
            provenance = provenance / part
            if provenance.is_symlink():
                symlinked = True
                break
        if symlinked:
            errors.append("composition provenance path contains a symbolic link")
        elif not provenance.is_file():
            errors.append(f"composition provenance is missing: {composition.provenance_path}")
        elif sha256_file(provenance) != composition.provenance_sha256:
            errors.append("composition provenance sha256 mismatch")
        elif manifest.lineage is None or manifest.lineage.record_provenance != "record":
            errors.append("composition record provenance is not declared in dataset lineage")
        else:
            parent_digests = {parent.dataset_id: parent.content_sha256 for parent in composition.parents}
            try:
                provenance_count = 0
                with provenance.open("r", encoding="utf-8") as stream:
                    for provenance_count, line in enumerate(stream, start=1):
                        index = provenance_count - 1
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            errors.append("composition provenance record must be an object")
                            break
                        parent_id = record.get("dataset_id")
                        if record.get("output_index") != index or parent_id not in parent_digests:
                            errors.append("composition provenance has an invalid output index or parent ID")
                            break
                        if record.get("parent_content_sha256") != parent_digests[parent_id]:
                            errors.append("composition provenance parent digest does not match manifest")
                            break
                if provenance_count != composition.output_record_count:
                    errors.append("composition provenance record count does not match manifest")
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"composition provenance is invalid: {exc}")
    for label, reference in (("processing_report", manifest.processing_report), ("quality_report", manifest.quality_report)):
        if reference is None:
            continue
        report_path = (manifest_path.parent / reference.path).resolve(strict=False)
        try:
            report_path.relative_to(artifact_root.resolve(strict=True))
        except ValueError:
            errors.append(f"{label} path escapes dataset artifact root")
            continue
        if not report_path.is_file():
            errors.append(f"{label} is missing: {reference.path}")
            continue
        if sha256_file(report_path) != reference.sha256:
            errors.append(f"{label} sha256 mismatch: {reference.path}")
            continue
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            errors.append(f"{label} is not valid JSON: {exc}")
            continue
        if label == "quality_report":
            if payload.get("quality_report_id") != reference.report_id:
                errors.append("quality_report ID does not match manifest reference")
            if payload.get("dataset_id") != manifest.dataset_id:
                errors.append("quality_report dataset_id does not match manifest")
            expected_content = manifest.hashes.get("canonical_records_sha256")
            if expected_content and payload.get("content_sha256") != expected_content:
                errors.append("quality_report content digest does not match dataset manifest")
    if manifest.lineage:
        quality_id = manifest.quality_report.report_id if manifest.quality_report else None
        if quality_id and quality_id not in manifest.lineage.quality_report_ids:
            errors.append("manifest quality report ID is missing from dataset lineage")
        if manifest.quality_report:
            quality_path = (manifest_path.parent / manifest.quality_report.path).resolve(strict=False)
            if quality_path.is_file():
                quality_payload = json.loads(quality_path.read_text(encoding="utf-8"))
                quality_run_id = quality_payload.get("run_id")
                if quality_run_id:
                    run_file = artifact_root / "runs" / quality_run_id / "lineage.json"
                    if not run_file.is_file():
                        errors.append(f"quality assessment run is missing: {quality_run_id}")
                    else:
                        run_payload = json.loads(run_file.read_text(encoding="utf-8"))
                        if run_payload.get("run_type") != "quality_assess" or run_payload.get("run_id") != quality_run_id:
                            errors.append("quality assessment run identity or type mismatch")
                        if not any(item.get("id") == manifest.dataset_id for item in run_payload.get("inputs", [])):
                            errors.append("quality assessment run does not reference the dataset")
                if quality_run_id is None:
                    errors.append("quality report has no producing run_id")
        build_file = artifact_root / "runs" / manifest.lineage.produced_by_run_id / "lineage.json"
        if not build_file.is_file():
            errors.append(f"data build run is missing: {manifest.lineage.produced_by_run_id}")
        else:
            build_payload = json.loads(build_file.read_text(encoding="utf-8"))
            if build_payload.get("run_type") != "data_build" or build_payload.get("run_id") != manifest.lineage.produced_by_run_id:
                errors.append("data build run identity or type mismatch")
            if not any(item.get("id") == manifest.dataset_id for item in build_payload.get("outputs", [])):
                errors.append("data build run does not reference the dataset output")
    return errors
