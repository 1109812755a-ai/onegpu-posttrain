"""Safe, bounded previews for registered canonical datasets."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from pydantic import ValidationError

from ..data.evidence import ProcessingReport, QualityReport
from ..data.manifest import DatasetManifest, load_manifest, validate_manifest_evidence
from ..schemas import STAGE_MODELS

PREVIEW_STAGES = {"cpt", "sft", "preference", "grpo"}
PREVIEW_ROOTS = ("data/processed", "artifacts/ui-demo", "artifacts/synthetic")
MAX_MANIFESTS = 5000
MAX_LINE_BYTES = 8 * 1024 * 1024
MAX_FIELD_CHARS = 2000

FIELDS_BY_STAGE = {
    "cpt": ("text",),
    "sft": ("raw_problem", "messages", "prompt_template_id"),
    "preference": ("raw_problem", "prompt", "chosen", "rejected", "prompt_template_id"),
    "grpo": ("raw_problem", "prompt", "answer", "prompt_template_id"),
}


class PreviewUnavailable(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CanonicalLocation:
    manifest_path: Path
    artifact_root: Path
    canonical_path: Path
    manifest: DatasetManifest
    record_count: int


def _contained_path(root: Path, relative: str, *, relative_to_manifest: Path | None = None) -> Path:
    if not isinstance(relative, str) or not relative.strip() or "\x00" in relative:
        raise PreviewUnavailable("canonical_path_missing", "数据集没有声明 canonical 文件路径")
    parsed = PurePosixPath(relative.replace("\\", "/"))
    if parsed.is_absolute() or not parsed.parts:
        raise PreviewUnavailable("canonical_path_invalid", "canonical 文件路径不是安全的工作区相对路径")
    base = relative_to_manifest or root
    try:
        resolved_root = root.resolve(strict=True)
        base.resolve(strict=True).relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise PreviewUnavailable("canonical_path_invalid", "manifest 路径越出数据集目录") from exc
    # Reject symlink components even when they resolve back inside the root. This
    # keeps the actual bytes being previewed tied to the manifest's artifact tree.
    current = base
    for part in parsed.parts:
        if part == ".":
            continue
        if part == "..":
            current = current.parent
            try:
                current.resolve(strict=False).relative_to(resolved_root)
            except (OSError, ValueError) as exc:
                raise PreviewUnavailable("canonical_path_invalid", "canonical 文件路径越出数据集目录") from exc
            continue
        current = current / part
        if current.is_symlink():
            raise PreviewUnavailable("canonical_path_invalid", "canonical 文件路径包含符号链接")
    candidate = current
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except FileNotFoundError as exc:
        raise PreviewUnavailable("canonical_missing", "登记的 canonical 文件不存在") from exc
    except (OSError, ValueError) as exc:
        raise PreviewUnavailable("canonical_path_invalid", "canonical 文件路径越出数据集目录") from exc
    if not resolved.is_file():
        raise PreviewUnavailable("canonical_missing", "登记的 canonical 路径不是文件")
    return resolved


def _canonical_for_manifest(manifest_path: Path, manifest: DatasetManifest) -> CanonicalLocation:
    artifact_root = manifest_path.parent.parent.resolve(strict=True)
    if manifest.stage not in PREVIEW_STAGES:
        raise PreviewUnavailable("unsupported_stage", "该阶段没有可预览的训练 canonical schema")

    if manifest.origin is not None and manifest.origin.kind == "local_upload":
        canonical_rel = manifest.metadata.get("canonical_path")
        canonical_path = _contained_path(artifact_root, canonical_rel)
    else:
        reference = manifest.processing_report
        if reference is None:
            raise PreviewUnavailable("evidence_missing", "旧数据缺少 processing report，无法确认 canonical 文件")
        report_path = _contained_path(artifact_root, reference.path, relative_to_manifest=manifest_path.parent)
        if _file_sha256(report_path) != reference.sha256:
            raise PreviewUnavailable("evidence_mismatch", "processing report 摘要与 manifest 不一致")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise PreviewUnavailable("evidence_invalid", "processing report 无法解析") from exc
        if not isinstance(report, dict) or report.get("build_id") != reference.report_id:
            raise PreviewUnavailable("evidence_mismatch", "processing report 身份与 manifest 不一致")
        output = report.get("outputs", {}).get(manifest.stage)
        if not isinstance(output, dict) or output.get("dataset_id") != manifest.dataset_id:
            raise PreviewUnavailable("evidence_missing", "processing report 没有对应 dataset ID 的阶段输出")
        canonical_path = _contained_path(artifact_root, output.get("path"))
        if output.get("content_sha256") != manifest.hashes.get("canonical_records_sha256"):
            raise PreviewUnavailable("evidence_mismatch", "processing report 中的 canonical 摘要与 manifest 不一致")

    expected_digest = manifest.hashes.get("canonical_records_sha256")
    if not expected_digest:
        raise PreviewUnavailable("evidence_missing", "manifest 没有 canonical SHA256")
    digest, record_count = _file_sha256_and_count(canonical_path)
    if digest != expected_digest:
        raise PreviewUnavailable("digest_mismatch", "canonical 内容摘要与 manifest 不一致")
    if record_count != manifest.sample_count:
        raise PreviewUnavailable("record_count_mismatch", "canonical 样本数与 manifest 不一致")
    return CanonicalLocation(manifest_path, artifact_root, canonical_path, manifest, record_count)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_sha256_and_count(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    record_count = 0
    last_byte = b""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            record_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if last_byte and last_byte != b"\n":
        record_count += 1
    return digest.hexdigest(), record_count


def _find_locations(workspace: Path, dataset_id: str) -> list[CanonicalLocation]:
    matches: list[tuple[Path, DatasetManifest]] = []
    visited = 0
    for relative_root in PREVIEW_ROOTS:
        root = workspace / relative_root
        if not root.is_dir() or root.is_symlink():
            continue
        for current, dirs, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            dirs[:] = [name for name in dirs if not (current_path / name).is_symlink() and not name.startswith(".")]
            if current_path.name != "manifests":
                continue
            for name in files:
                if Path(name).suffix.lower() not in {".yaml", ".yml", ".json"}:
                    continue
                path = current_path / name
                if path.is_symlink():
                    continue
                visited += 1
                if visited > MAX_MANIFESTS:
                    raise PreviewUnavailable("catalog_limit", "数据集 manifest 数量超过预览索引上限")
                try:
                    manifest = load_manifest(path)
                except (OSError, ValueError, ValidationError, yaml.YAMLError):
                    continue
                if manifest.dataset_id == dataset_id:
                    matches.append((path, manifest))
    if not matches:
        raise KeyError(dataset_id)
    identities = {(manifest.stage, manifest.hashes.get("canonical_records_sha256")) for _, manifest in matches}
    if len(identities) != 1:
        raise PreviewUnavailable("identity_conflict", "同一 dataset ID 对应的 manifest 证据不一致")
    locations: list[CanonicalLocation] = []
    errors: list[PreviewUnavailable] = []
    for path, manifest in matches:
        try:
            locations.append(_canonical_for_manifest(path, manifest))
        except PreviewUnavailable as exc:
            errors.append(exc)
    if not locations:
        if errors:
            raise errors[0]
        raise PreviewUnavailable("evidence_missing", "没有可验证的 canonical 数据")
    return locations


def _clip(value: Any, path: str, truncated: list[str]) -> Any:
    if isinstance(value, str):
        if len(value) > MAX_FIELD_CHARS:
            truncated.append(path)
            return value[:MAX_FIELD_CHARS] + "…"
        return value
    if isinstance(value, list):
        return [_clip(item, f"{path}[{index}]", truncated) for index, item in enumerate(value)]
    if isinstance(value, dict):
        return {key: _clip(item, f"{path}.{key}", truncated) for key, item in value.items()}
    return value


def preview_dataset(workspace: str | Path, dataset_id: str, *, limit: int = 3) -> dict[str, Any]:
    """Return a small schema-validated preview after verifying the full file digest."""
    if not isinstance(dataset_id, str) or not dataset_id or len(dataset_id) > 200:
        raise KeyError(dataset_id)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10:
        raise ValueError("预览数量必须是 1 到 10 之间的整数")
    locations = _find_locations(Path(workspace).resolve(strict=True), dataset_id)
    location = locations[0]
    manifest = location.manifest
    samples: list[dict[str, Any]] = []
    count = 0
    model = STAGE_MODELS[manifest.stage]
    try:
        with location.canonical_path.open("rb") as stream:
            while len(samples) < limit:
                raw_line = stream.readline(MAX_LINE_BYTES + 1)
                if not raw_line:
                    break
                count += 1
                if len(raw_line) > MAX_LINE_BYTES:
                    raise PreviewUnavailable("record_too_large", "样本行超过预览读取上限")
                try:
                    record = model.model_validate_json(raw_line)
                except (ValueError, UnicodeError, ValidationError) as exc:
                    raise PreviewUnavailable("record_invalid", "canonical 样本不符合该阶段 schema") from exc
                fields = record.model_dump(mode="json", include=set(FIELDS_BY_STAGE[manifest.stage]))
                truncated: list[str] = []
                samples.append({"id": record.id, "fields": _clip(fields, "fields", truncated),
                                "truncated_fields": truncated})
    except PreviewUnavailable:
        raise
    except OSError as exc:
        raise PreviewUnavailable("canonical_unavailable", "canonical 文件无法读取") from exc

    if count == 0 or manifest.sample_count <= 0:
        raise PreviewUnavailable("empty_dataset", "数据集没有可预览的样本")
    return {
        "available": True,
        "dataset_id": manifest.dataset_id,
        "stage": manifest.stage,
        "canonical_sha256": manifest.hashes["canonical_records_sha256"],
        "integrity": "verified",
        "total_records": location.record_count,
        "returned_records": len(samples),
        "truncated": location.record_count > len(samples),
        "samples": samples,
    }


def dataset_status(workspace: str | Path, dataset_id: str) -> dict[str, Any]:
    """Return a stable, read-only dataset/quality/lineage summary without sample rows."""
    if not isinstance(dataset_id, str) or not dataset_id or len(dataset_id) > 200:
        raise KeyError(dataset_id)
    locations = _find_locations(Path(workspace).resolve(strict=True), dataset_id)
    signatures = {
        (item.manifest.stage, item.manifest.hashes.get("canonical_records_sha256"),
         item.manifest.lineage.model_dump_json() if item.manifest.lineage else None,
         item.manifest.origin.model_dump_json() if item.manifest.origin else None,
         item.manifest.processing_report.model_dump_json() if item.manifest.processing_report else None,
         item.manifest.quality_report.model_dump_json() if item.manifest.quality_report else None)
        for item in locations
    }
    if len(signatures) != 1:
        raise PreviewUnavailable("identity_conflict", "同一 dataset ID 的 lineage/source/report 身份不一致")
    location = locations[0]
    manifest = location.manifest
    evidence_errors = validate_manifest_evidence(location.manifest_path)
    missing: list[str] = []
    processing: dict[str, Any] | None = None
    quality: dict[str, Any] | None = None

    def read_report(reference, kind: str):
        if reference is None:
            missing.append(f"{kind}_report")
            return None
        report_path = _contained_path(location.artifact_root, reference.path,
                                      relative_to_manifest=location.manifest_path.parent)
        digest = _file_sha256(report_path)
        if digest != reference.sha256:
            missing.append(f"{kind}_report_hash")
            return None
        try:
            report_raw = json.loads(report_path.read_text(encoding="utf-8"))
            if not isinstance(report_raw, dict):
                raise ValueError("report root is not an object")
            return report_raw, digest
        except (OSError, UnicodeError, ValueError):
            missing.append(f"{kind}_report_invalid")
            return None

    raw_processing = read_report(manifest.processing_report, "processing")
    if raw_processing:
        try:
            parsed = ProcessingReport.model_validate(raw_processing[0])
            output = next((item for item in parsed.outputs.values()
                           if item.get("dataset_id") == manifest.dataset_id), None)
            if (parsed.build_id != manifest.processing_report.report_id or output is None
                    or output.get("content_sha256") != manifest.hashes.get("canonical_records_sha256")):
                raise ValueError("processing output identity mismatch")
            processing = {"status": "verified", "report_id": parsed.build_id,
                          "sha256": raw_processing[1], "protocol": parsed.processing_protocol}
        except (TypeError, ValueError):
            missing.append("processing_report_identity")

    raw_quality = read_report(manifest.quality_report, "quality")
    if raw_quality:
        try:
            report = QualityReport.model_validate(raw_quality[0])
            if (report.quality_report_id != manifest.quality_report.report_id
                    or report.dataset_id != manifest.dataset_id
                    or report.content_sha256 != manifest.hashes.get("canonical_records_sha256")):
                raise ValueError("quality report identity mismatch")
            quality = {
                "status": report.gate.summary_status, "report_id": report.quality_report_id,
                "sha256": raw_quality[1], "protocol": report.quality_protocol,
                "metrics": {key: value.model_dump(mode="json", by_alias=True)
                            for key, value in report.metrics.items()},
                "evidence_coverage": (report.evidence_coverage.model_dump(mode="json", by_alias=True)
                                      if report.evidence_coverage else None),
                "limitations": report.limitations,
            }
        except (TypeError, ValueError):
            missing.append("quality_report_identity")
    else:
        missing.append("quality_assessment")

    if manifest.lineage is None:
        missing.append("dataset_lineage")
    if evidence_errors:
        missing.extend(f"manifest_evidence:{item}" for item in evidence_errors[:10])
    origin = manifest.origin
    source = {
        "kind": origin.kind if origin else ("declared_source" if manifest.source else "unknown"),
        "name": origin.name if origin else manifest.source.repo if manifest.source else None,
        "revision": origin.revision if origin else manifest.source.revision if manifest.source else None,
        "revision_status": origin.revision_status if origin else "known" if manifest.source and manifest.source.revision else "not_provided",
        "license_status": origin.license_status if origin else "not_provided",
    }
    lineage = None
    if manifest.lineage:
        lineage = {"parents": manifest.lineage.parents,
                   "produced_by_run_id": manifest.lineage.produced_by_run_id,
                   "record_provenance": manifest.lineage.record_provenance,
                   "processing_report_id": manifest.lineage.processing_report_id,
                   "quality_report_ids": manifest.lineage.quality_report_ids}
    return {
        "schema_version": "dataset-status-v1", "read_only": True,
        "dataset_id": manifest.dataset_id, "stage": manifest.stage,
        "record_count": location.record_count,
        "manifest_sha256": _file_sha256(location.manifest_path),
        "canonical_sha256": manifest.hashes.get("canonical_records_sha256"),
        "integrity": "verified", "source": source,
        "processing": processing, "quality": quality, "lineage": lineage,
        "missing_evidence": list(dict.fromkeys(missing)),
        "notice": "只读证据摘要；不返回样本正文，不启动 Provider、登记、训练或评测。",
    }
