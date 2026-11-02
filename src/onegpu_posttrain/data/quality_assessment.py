"""Offline, additive reassessment bundles for retained Mini datasets."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from .evidence import EvidenceModel, QualityReport, assess_dataset
from .manifest import DatasetManifest, load_manifest, sha256_file
from .mini import load_mini_profile

MAX_REASSESS_DATA_BYTES = 512 * 1024 * 1024


class QualityAssessmentEntry(EvidenceModel):
    dataset_key: str
    dataset_id: str
    stage: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    report_id: str
    report_path: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class QualityAssessmentIndex(EvidenceModel):
    schema_version: Literal["1.0"] = "1.0"
    assessment_id: str
    created_at: str
    quality_protocol: str
    profile_id: str
    profile_version: str
    profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    datasets: list[QualityAssessmentEntry]


def _load_processing_output(root: Path, manifest_path: Path, manifest: DatasetManifest) -> tuple[Path, dict[str, Any]]:
    if manifest.processing_report is None:
        raise ValueError(f"{manifest.dataset_id}: manifest has no processing report")
    unresolved_report_path = manifest_path.parent / manifest.processing_report.path
    if unresolved_report_path.is_symlink():
        raise ValueError(f"{manifest.dataset_id}: processing report path is a symlink")
    report_path = unresolved_report_path.resolve(strict=True)
    report_root = manifest_path.parent.parent.resolve(strict=True)
    report_path.relative_to(report_root)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if payload.get("build_id") != manifest.processing_report.report_id:
        raise ValueError(f"{manifest.dataset_id}: processing report identity mismatch")
    output = payload.get("outputs", {}).get(manifest.stage)
    if not isinstance(output, dict) or output.get("dataset_id") != manifest.dataset_id:
        raise ValueError(f"{manifest.dataset_id}: processing report output identity mismatch")
    unresolved_data_path = report_root / output.get("path", "")
    if unresolved_data_path.is_symlink():
        raise ValueError(f"{manifest.dataset_id}: canonical data path is a symlink")
    data_path = unresolved_data_path.resolve(strict=True)
    data_path.relative_to(report_root)
    if not data_path.is_file():
        raise ValueError(f"{manifest.dataset_id}: canonical data path is not a regular local file")
    if data_path.stat().st_size > MAX_REASSESS_DATA_BYTES:
        raise ValueError(f"{manifest.dataset_id}: canonical dataset exceeds reassessment limit")
    expected_digest = manifest.hashes.get("canonical_records_sha256")
    if not expected_digest or sha256_file(data_path) != expected_digest:
        raise ValueError(f"{manifest.dataset_id}: canonical dataset digest mismatch")
    if output.get("content_sha256") != expected_digest:
        raise ValueError(f"{manifest.dataset_id}: processing output digest mismatch")
    return data_path, output


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path.name}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _existing_overlap(manifest_path: Path, manifest: DatasetManifest) -> int | None:
    if manifest.quality_report is None:
        return None
    report_path = (manifest_path.parent / manifest.quality_report.path).resolve(strict=False)
    try:
        report_path.relative_to(manifest_path.parent.parent.resolve(strict=True))
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if payload.get("quality_report_id") != manifest.quality_report.report_id:
        return None
    if payload.get("dataset_id") != manifest.dataset_id:
        return None
    if payload.get("content_sha256") != manifest.hashes.get("canonical_records_sha256"):
        return None
    metric = payload.get("metrics", {}).get("train_eval_overlap", {})
    value = metric.get("value") if isinstance(metric, dict) else None
    return value if isinstance(value, int) and value >= 0 else None


def create_quality_assessment_bundle(
    artifact_root: str | Path,
    profile_path: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[QualityAssessmentIndex, Path]:
    """Read retained datasets and write new reports without changing source artifacts."""
    root = Path(artifact_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("artifact root must be a directory")
    profile, profile_sha256 = load_mini_profile(profile_path)
    assessment_id = f"mini-quality-{uuid4().hex}"
    default_target = root / "reports" / "mini-quality-assessments" / assessment_id
    target = Path(output_dir).resolve(strict=False) if output_dir else default_target
    target.relative_to(root)
    if target.exists():
        raise FileExistsError(f"Quality assessment output already exists: {target}")
    target.mkdir(parents=True)
    entries: list[QualityAssessmentEntry] = []
    reports_dir = target / "quality"
    reports_dir.mkdir()

    for dataset_profile in profile.datasets:
        matches: list[tuple[Path, DatasetManifest]] = []
        if dataset_profile.manifest_path:
            manifest_path = (root / dataset_profile.manifest_path).resolve(strict=False)
            try:
                manifest_path.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"{dataset_profile.dataset_key}: manifest_path escapes artifact root") from exc
            if manifest_path.is_file():
                manifest = load_manifest(manifest_path)
                if manifest.stage == dataset_profile.stage:
                    matches.append((manifest_path, manifest))
        else:
            for manifest_path in root.rglob("manifests/*"):
                if manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.suffix.lower() not in {".yaml", ".yml", ".json"}:
                    continue
                manifest = load_manifest(manifest_path)
                if manifest.stage == dataset_profile.stage:
                    matches.append((manifest_path, manifest))
        if len(matches) != 1:
            raise ValueError(f"{dataset_profile.dataset_key}: expected one manifest for stage {dataset_profile.stage}, found {len(matches)}")
        manifest_path, manifest = matches[0]
        data_path, _ = _load_processing_output(root, manifest_path, manifest)
        rows = _read_rows(data_path)
        tokenizer = (
            f"{manifest.tokenizer.repo}@{manifest.tokenizer.revision}"
            if manifest.tokenizer else "unavailable"
        )
        overlap = _existing_overlap(manifest_path, manifest)
        report = assess_dataset(
            rows,
            stage=manifest.stage,
            dataset_id=manifest.dataset_id,
            content_sha256=manifest.hashes["canonical_records_sha256"],
            build_id=manifest.processing_report.report_id if manifest.processing_report else "unavailable",
            tokenizer_name=tokenizer,
            overlap=overlap,
        )
        report_path = reports_dir / f"{dataset_profile.dataset_key.replace('.', '-')}.json"
        report_path.write_text(
            json.dumps(report.model_dump(mode="json", by_alias=True), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        entries.append(QualityAssessmentEntry(
            dataset_key=dataset_profile.dataset_key,
            dataset_id=manifest.dataset_id,
            stage=manifest.stage,
            content_sha256=manifest.hashes["canonical_records_sha256"],
            report_id=report.quality_report_id,
            report_path=report_path.relative_to(target).as_posix(),
            report_sha256=sha256_file(report_path),
        ))

    first_report = QualityReport.model_validate(
        json.loads((target / entries[0].report_path).read_text(encoding="utf-8"))
    ) if entries else None
    index = QualityAssessmentIndex(
        assessment_id=assessment_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        quality_protocol=first_report.quality_protocol if first_report else "stage-data-quality-v1",
        profile_id=profile.profile_id,
        profile_version=profile.profile_version,
        profile_sha256=profile_sha256,
        datasets=entries,
    )
    index_path = target / "index.json"
    index_path.write_text(json.dumps(index.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index, index_path


def load_quality_assessment_bundle(path: str | Path) -> tuple[Path, QualityAssessmentIndex]:
    supplied = Path(path)
    if supplied.is_symlink():
        raise ValueError("quality assessment path cannot be a symlink")
    supplied = supplied.resolve(strict=True)
    index_path = supplied / "index.json" if supplied.is_dir() else supplied
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError("quality assessment index must be a regular file")
    bundle_root = index_path.parent.resolve(strict=True)
    if index_path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("quality assessment index exceeds the read limit")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    index = QualityAssessmentIndex.model_validate(payload)
    dataset_keys = [entry.dataset_key for entry in index.datasets]
    if len(dataset_keys) != len(set(dataset_keys)):
        raise ValueError("quality assessment bundle contains duplicate dataset keys")
    for entry in index.datasets:
        unresolved_report_path = bundle_root / entry.report_path
        if unresolved_report_path.is_symlink():
            raise ValueError(f"quality report path is a symlink: {entry.report_path}")
        report_path = unresolved_report_path.resolve(strict=True)
        report_path.relative_to(bundle_root)
        if not report_path.is_file():
            raise ValueError(f"quality report is not a regular file: {entry.report_path}")
        if sha256_file(report_path) != entry.report_sha256:
            raise ValueError(f"quality report digest mismatch: {entry.report_path}")
        report = QualityReport.model_validate(json.loads(report_path.read_text(encoding="utf-8")))
        if report.quality_report_id != entry.report_id or report.dataset_id != entry.dataset_id:
            raise ValueError(f"quality report identity mismatch: {entry.report_path}")
        if report.content_sha256 != entry.content_sha256 or report.stage != entry.stage:
            raise ValueError(f"quality report dataset binding mismatch: {entry.report_path}")
    return bundle_root, index
