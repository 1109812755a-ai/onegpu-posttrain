"""Create a trainer-native view only from a verified registered dataset manifest."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..data.io import read_jsonl, write_jsonl
from ..data.manifest import load_manifest, resolve_canonical_path, sha256_file
from ..lineage import ArtifactRef, Digest, RunRecord, new_run_id, sha256_json
from .ms_swift import ADAPTERS, validate_native_record

_PROVENANCE_KEYS = (
    "source", "source_row_id", "source_record_id", "source_uuid", "problem_id",
    "parent_record_ids", "record_provenance", "synthetic_candidate_ids",
)


def _canonical_path(manifest_path: Path, manifest) -> Path:
    canonical_path = resolve_canonical_path(manifest_path, manifest)
    if manifest.processing_report is None:
        raise ValueError("registered dataset is missing its processing report")
    artifact_root = manifest_path.parent.parent.resolve(strict=True)
    unresolved_report = manifest_path.parent / manifest.processing_report.path
    if unresolved_report.is_symlink():
        raise ValueError("processing report path cannot be a symbolic link")
    report_path = unresolved_report.resolve(strict=True)
    report_path.relative_to(artifact_root)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if payload.get("build_id") != manifest.processing_report.report_id:
        raise ValueError("processing report identity does not match manifest")
    output = payload.get("outputs", {}).get(manifest.stage)
    if not isinstance(output, dict) or output.get("dataset_id") != manifest.dataset_id:
        raise ValueError("processing report does not reference this dataset ID")
    unresolved_path = artifact_root / str(output.get("path", ""))
    if unresolved_path.is_symlink():
        raise ValueError("canonical dataset path cannot be a symbolic link")
    path = unresolved_path.resolve(strict=True)
    path.relative_to(artifact_root)
    expected = manifest.hashes.get("canonical_records_sha256")
    if not expected or sha256_file(path) != expected or output.get("content_sha256") != expected:
        raise ValueError("canonical data hash does not match registered dataset manifest")
    if path != canonical_path:
        raise ValueError("processing report canonical path does not match manifest metadata.canonical_path")
    return canonical_path


def _safe_provenance(row: dict[str, Any], *, dataset_id: str, row_index: int) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return {
        "dataset_id": dataset_id,
        "canonical_record_id": str(row.get("id", "")),
        "row_index": row_index,
        "source": row.get("source") or metadata.get("source"),
        "metadata": {key: metadata[key] for key in _PROVENANCE_KEYS if key in metadata},
    }


def adapt_registered_dataset(manifest: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Adapt a registered canonical dataset; never mutates the source artifact."""
    supplied_manifest = Path(manifest).expanduser()
    if supplied_manifest.is_symlink():
        raise ValueError("manifest path cannot be a symbolic link")
    manifest_path = supplied_manifest.resolve(strict=True)
    dataset = load_manifest(manifest_path)
    stage = dataset.stage
    if stage not in {"cpt", "sft", "preference", "grpo"} or stage not in ADAPTERS:
        raise ValueError(f"registered dataset stage is not supported by ms-swift: {stage}")
    if dataset.lineage is None:
        raise ValueError("registered dataset is missing dataset lineage")
    canonical_path = _canonical_path(manifest_path, dataset)
    target = Path(output_dir).expanduser().resolve(strict=False)
    if target.exists():
        raise FileExistsError("native dataset output already exists; historical adapter files are never overwritten")
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".dataset-adapter-", dir=target.parent) as temporary:
        staging = Path(temporary)
        native_path = staging / "native" / f"{stage}.jsonl"
        provenance_path = staging / "provenance" / "record_lineage.jsonl"
        native_rows: list[dict[str, Any]] = []
        provenance_rows: list[dict[str, Any]] = []
        record_count = 0
        for record_count, row in enumerate(read_jsonl(canonical_path), start=1):
            native = ADAPTERS[stage](row)
            validate_native_record(stage, native)
            native_rows.append(native)
            provenance_rows.append(_safe_provenance(row, dataset_id=dataset.dataset_id, row_index=record_count - 1))
        if record_count != dataset.sample_count:
            raise ValueError("canonical record count does not match registered manifest")
        if not native_rows:
            raise ValueError("registered canonical dataset is empty")

        write_jsonl(native_path, native_rows)
        native_sidecar = {
            "adapter": "ms-swift", "stage": stage, "input": str(canonical_path),
            "output": str(native_path), "records": record_count,
            "input_sha256": dataset.hashes["canonical_records_sha256"],
            "output_sha256": sha256_file(native_path),
        }
        native_path.with_suffix(native_path.suffix + ".adapter.json").write_text(
            json.dumps(native_sidecar, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8",
        )
        sidecar_sha = sha256_file(native_path.with_suffix(native_path.suffix + ".adapter.json"))
        write_jsonl(provenance_path, provenance_rows)
        native_sha = sha256_file(native_path)
        provenance_sha = sha256_file(provenance_path)
        run_id = new_run_id("data_build", label=f"adapt-{stage}")
        created_at = datetime.now(timezone.utc).isoformat()
        config = {"adapter": "ms-swift", "adapter_version": "canonical-v1", "stage": stage}
        report = {
            "schema_version": "1.0",
            "dataset_id": dataset.dataset_id,
            "stage": stage,
            "canonical_sha256": dataset.hashes["canonical_records_sha256"],
            "manifest_path": str(manifest_path),
            "record_count": record_count,
            "adapter": config,
            "native_path": f"native/{stage}.jsonl",
            "native_sha256": native_sha,
            "native_sidecar_path": f"native/{stage}.jsonl.adapter.json",
            "native_sidecar_sha256": sidecar_sha,
            "provenance_path": "provenance/record_lineage.jsonl",
            "provenance_sha256": provenance_sha,
            "created_at": created_at,
            "run_id": run_id,
        }
        report_path = staging / "adapter_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        report_sha = sha256_file(report_path)
        run = RunRecord(
            run_id=run_id, run_type="data_build", status="completed", run_label=f"adapt-{stage}",
            started_at=created_at, completed_at=datetime.now(timezone.utc).isoformat(),
            inputs=[ArtifactRef(id=dataset.dataset_id, kind="dataset", role="canonical_input",
                                digest=Digest(value=dataset.hashes["canonical_records_sha256"], status="available"))],
            outputs=[
                ArtifactRef(id=f"native-{dataset.dataset_id}", kind="file", path=f"native/{stage}.jsonl",
                            role="trainer_native", digest=Digest(value=native_sha, status="available")),
                ArtifactRef(id=f"native-sidecar-{dataset.dataset_id}", kind="file",
                            path=f"native/{stage}.jsonl.adapter.json", role="dataset_lineage_sidecar",
                            digest=Digest(value=sidecar_sha, status="available")),
                ArtifactRef(id=f"provenance-{dataset.dataset_id}", kind="file", path="provenance/record_lineage.jsonl",
                            role="record_provenance", digest=Digest(value=provenance_sha, status="available")),
                ArtifactRef(id=f"adapter-report-{dataset.dataset_id}", kind="file", path="adapter_report.json",
                            role="adapter_report", digest=Digest(value=report_sha, status="available")),
            ],
            recipe_path="ms-swift registered dataset adapter",
            resolved_config_sha256=sha256_json(config),
        )
        run_dir = staging / "runs" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "lineage.json").write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")
        os.replace(staging, target)
    return {**report, "output_dir": str(target)}
