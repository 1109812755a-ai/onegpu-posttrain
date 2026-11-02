"""Safe, immutable role repair for verifiable legacy FineMath CPT artifacts."""

from __future__ import annotations

import io
import json
from pathlib import Path, PurePosixPath

from .dataset_import import RegistrationResult, register_jsonl
from .importer import validate_canonical_jsonl
from .manifest import load_manifest, resolve_canonical_path, sha256_file, validate_manifest_evidence
from .registry import DataSource

_LEGACY_PROTOCOL = "finemath-build-v2"
_REPAIR_PROTOCOL = "cpt-role-repair-v1"


def inspect_legacy_cpt_role_repair(manifest_path: str | Path) -> dict:
    """Validate all local evidence and return a read-only repair preview."""
    supplied = Path(manifest_path).expanduser()
    if supplied.is_symlink():
        raise ValueError("CPT manifest path cannot be a symbolic link")
    path = supplied.resolve(strict=True)
    manifest = load_manifest(path)
    if manifest.stage != "cpt":
        raise ValueError("自动角色修复仅支持有可信 FineMath 构建证据的 CPT 训练 split")
    if manifest.metadata.get("build_protocol") != _LEGACY_PROTOCOL:
        raise ValueError("manifest 不属于允许修复的 finemath-build-v2；上传或来源不明的数据不会自动补角色")
    if manifest.protocol_versions.get("data_role") is not None:
        raise ValueError("该数据集已声明 data_role 协议；请修复构建源或重新构建，不执行历史补标")
    evidence_errors = validate_manifest_evidence(path)
    if evidence_errors:
        raise ValueError("CPT manifest evidence invalid: " + "; ".join(evidence_errors))
    canonical = resolve_canonical_path(path, manifest)
    processing_path = _resolve_processing_report(path, manifest.processing_report.path)
    if sha256_file(processing_path) != manifest.processing_report.sha256:
        raise ValueError("processing report digest does not match manifest")
    processing = json.loads(processing_path.read_text(encoding="utf-8"))
    if processing.get("processing_protocol") != _LEGACY_PROTOCOL:
        raise ValueError("processing report is not from the allowlisted legacy FineMath protocol")
    outputs = processing.get("outputs")
    output = outputs.get("cpt") if isinstance(outputs, dict) else None
    if not isinstance(output, dict):
        raise ValueError("processing report has no unique CPT output record")
    if (output.get("dataset_id") != manifest.dataset_id
            or output.get("path") != manifest.metadata.get("canonical_path")
            or output.get("content_sha256") != manifest.hashes.get("canonical_records_sha256")
            or output.get("record_count") != manifest.sample_count):
        raise ValueError("processing report CPT output identity/path/digest/count does not match manifest")
    artifact_root = path.parent.parent.resolve(strict=True)
    build_run_path = artifact_root / "runs" / manifest.lineage.produced_by_run_id / "lineage.json"
    if build_run_path.is_symlink() or not build_run_path.is_file():
        raise ValueError("trusted FineMath data-build lineage run is missing or linked")
    build_run = json.loads(build_run_path.read_text(encoding="utf-8"))
    matching_outputs = [item for item in build_run.get("outputs", [])
                        if isinstance(item, dict) and item.get("id") == manifest.dataset_id]
    if len(matching_outputs) != 1:
        raise ValueError("data-build run does not uniquely identify this dataset")
    run_output = matching_outputs[0]
    if (run_output.get("digest", {}).get("value") != manifest.hashes.get("canonical_records_sha256")
            or run_output.get("path") != manifest.metadata.get("canonical_path")):
        raise ValueError("data-build run output path/digest does not match the CPT manifest")

    rows: list[dict] = []
    with canonical.open("rb") as stream:
        imported = validate_canonical_jsonl(stream, "cpt", max_records=250_000)
    if not imported.valid or imported.record_count != manifest.sample_count:
        raise ValueError("legacy CPT canonical schema/count is invalid; no repair preview can be trusted")
    with canonical.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"canonical JSONL 第 {line_number} 行无效") from exc
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            existing_role = metadata.get("data_role")
            if existing_role is not None:
                raise ValueError("旧协议 canonical 中已有角色字段；不覆盖或归一化任何既有角色")
            rows.append(row)
    if len(rows) != manifest.sample_count or not rows:
        raise ValueError("canonical 行数为空或与登记数量不一致")
    return {
        "status": "eligible", "read_only": True, "apply_required": True,
        "parent_dataset_id": manifest.dataset_id,
        "parent_content_sha256": manifest.hashes["canonical_records_sha256"],
        "parent_manifest": str(path), "parent_stage": manifest.stage,
        "source_protocol": _LEGACY_PROTOCOL, "repair_protocol": _REPAIR_PROTOCOL,
        "role_assignment": {"source_split": "cpt", "data_role": "train_seed", "record_count": len(rows)},
        "output_preview": {"stage": "cpt", "record_count": len(rows),
                           "new_dataset_id": "generated on explicit apply", "old_artifacts_modified": False},
        "downloads_performed": False, "execution_started": False,
    }


def repair_legacy_cpt_roles(
    manifest_path: str | Path,
    *,
    workspace: str | Path,
    output_root: str | Path,
    apply: bool = False,
) -> dict:
    """Preview by default; with explicit apply, register a new immutable child dataset."""
    preview = inspect_legacy_cpt_role_repair(manifest_path)
    if not apply:
        return preview

    parent_id = preview["parent_dataset_id"]
    parent_digest = preview["parent_content_sha256"]
    canonical = resolve_canonical_path(manifest_path)
    rows = []
    with canonical.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            row.setdefault("metadata", {})["data_role"] = "train_seed"
            rows.append(row)
    payload = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
                       for row in rows)
    source = DataSource(
        source_id=f"source-cpt-role-repair-{parent_id}", kind="dataset_derived",
        name=f"CPT role repair from {parent_id}", revision="cpt-role-repair-v1",
        revision_status="known", license_status="not_provided", declared_by="adapter",
    )
    workspace_path = Path(workspace).expanduser().resolve(strict=True)
    output_path = Path(output_root).expanduser()
    if not output_path.is_absolute():
        output_path = workspace_path / output_path
    registration: RegistrationResult = register_jsonl(
        io.BytesIO(payload), workspace=workspace_path, output_root=output_path, stage="cpt", origin=source,
        registration_metadata={
            "parent_dataset_ids": [parent_id],
            "parent_dataset_refs": [{"dataset_id": parent_id, "content_sha256": parent_digest}],
            "derivation_protocol": _REPAIR_PROTOCOL,
            "derivation_config_sha256": _sha256_text(f"{_REPAIR_PROTOCOL}:{_LEGACY_PROTOCOL}:cpt=train_seed"),
            "role_source_protocol": _LEGACY_PROTOCOL,
            "role_assignment": "manifest-stage-and-digest-verified-builder-output",
            "data_role_protocol": "cpt-data-role-v1",
            "data_role_counts": {"train_seed": len(rows)},
        },
    )
    return {
        **preview, "status": "registered", "read_only": False,
        "dataset_id": registration.dataset_id, "canonical_sha256": registration.canonical_sha256,
        "record_count": registration.record_count, "reused": registration.reused,
        "manifest_path": str(registration.path / "manifests/cpt-dataset-derived-v1.yaml"),
        "old_artifacts_modified": False,
    }


def _resolve_processing_report(manifest_path: Path, reference: str) -> Path:
    relative = PurePosixPath(reference)
    if relative.is_absolute() or "\\" in reference or any(part in {"", "."} for part in relative.parts):
        raise ValueError("processing report path must be a safe relative path")
    artifact_root = manifest_path.parent.parent.resolve(strict=True)
    candidate = manifest_path.parent
    for part in relative.parts:
        candidate = candidate.parent if part == ".." else candidate / part
        if candidate.is_symlink():
            raise ValueError("processing report cannot be reached through a symbolic link")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(artifact_root) or not resolved.is_file():
        raise ValueError("processing report path escapes dataset artifact root")
    return resolved


def _sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()
