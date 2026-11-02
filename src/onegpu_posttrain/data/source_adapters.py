"""Thin adapters from existing open-source builder manifests to registry objects."""

from __future__ import annotations

import json
from pathlib import Path

from .manifest import DatasetManifest, load_manifest, sha256_file
from .registry import DatasetArtifact

_SOURCE_REPOSITORIES = {
    "finemath": ("finemath", "huggingfacetb/finemath"),
    "deepmath": ("deepmath", "zwhe99/deepmath-103k"),
    "openr1": ("openr1", "open-r1/openr1-math-220k"),
}


def source_id_for_manifest(manifest: DatasetManifest) -> str | None:
    if manifest.origin is not None:
        return manifest.origin.source_id if manifest.origin.kind == "open_source" else None
    repository = (manifest.source.repo if manifest.source else "") or ""
    normalized = repository.lower().strip()
    for source_id, (_, expected) in _SOURCE_REPOSITORIES.items():
        if normalized == expected or normalized.startswith(expected + "/"):
            return source_id
    return None


def adapt_open_source_manifest(path: str | Path) -> DatasetArtifact:
    """Validate one already-built open-source artifact without rewriting its cache."""
    manifest_path = Path(path).resolve(strict=True)
    manifest = load_manifest(manifest_path)
    source_id = source_id_for_manifest(manifest)
    if source_id is None:
        raise ValueError("manifest does not identify a supported open-source source adapter")
    if manifest.lineage is None or manifest.processing_report is None:
        raise ValueError("open-source artifact is missing build lineage or processing evidence")
    if manifest.stage not in {"cpt", "sft", "preference", "grpo"}:
        raise ValueError(f"stage {manifest.stage!r} is not a trainable registry stage")

    artifact_root = manifest_path.parent.parent
    processing_path = (manifest_path.parent / manifest.processing_report.path).resolve(strict=True)
    processing_path.relative_to(artifact_root.resolve(strict=True))
    processing = json.loads(processing_path.read_text(encoding="utf-8"))
    if processing.get("build_id") != manifest.processing_report.report_id:
        raise ValueError("processing report identity does not match manifest")
    output = processing.get("outputs", {}).get(manifest.stage)
    if not isinstance(output, dict) or output.get("dataset_id") != manifest.dataset_id:
        raise ValueError("processing report does not contain this manifest's dataset")
    canonical_path = (artifact_root / str(output.get("path", ""))).resolve(strict=True)
    canonical_path.relative_to(artifact_root.resolve(strict=True))
    digest = sha256_file(canonical_path)
    if digest != manifest.hashes.get("canonical_records_sha256") or output.get("content_sha256") != digest:
        raise ValueError("canonical dataset digest does not match manifest and processing report")
    return DatasetArtifact(
        dataset_id=manifest.dataset_id,
        stage=manifest.stage,
        source_id=source_id,
        produced_by_run_id=manifest.lineage.produced_by_run_id,
        record_count=manifest.sample_count,
        canonical_sha256=digest,
        quality_report_id=manifest.quality_report.report_id if manifest.quality_report else None,
    )
