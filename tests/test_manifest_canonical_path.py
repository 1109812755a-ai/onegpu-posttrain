from __future__ import annotations

import json

import pytest

from onegpu_posttrain.data.evidence import ReportReference
from onegpu_posttrain.data.manifest import (
    DatasetManifest, load_manifest, repair_manifest_canonical_path, save_manifest,
    sha256_file,
)
from onegpu_posttrain.lineage import DatasetLineage


def _legacy_artifact(root):
    canonical = root / "canonical/sft.jsonl"
    canonical.parent.mkdir(parents=True)
    canonical.write_text('{"messages": []}\n', encoding="utf-8")
    digest = sha256_file(canonical)
    report = root / "reports/processing_report.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"build_id": "build-fixture", "outputs": {"sft": {
        "dataset_id": "ds-fixture", "path": "canonical/sft.jsonl", "content_sha256": digest,
    }}}), encoding="utf-8")
    manifest_path = root / "manifests/sft.yaml"
    manifest = DatasetManifest(
        dataset_id="ds-fixture", stage="sft", sample_count=1,
        hashes={"canonical_records_sha256": digest},
        processing_report=ReportReference(path="../reports/processing_report.json",
                                          sha256=sha256_file(report), report_id="build-fixture"),
        lineage=DatasetLineage(produced_by_run_id="run-fixture", transform={"name": "fixture", "version": "1"},
                               source_evidence={}, record_provenance="dataset_only", count_unit="examples",
                               content_sha256=digest),
    )
    save_manifest(manifest_path, manifest, allow_legacy=True)
    return manifest_path


def test_new_manifest_requires_verified_artifact_relative_canonical_path(tmp_path):
    root = tmp_path / "artifact"
    canonical = root / "canonical/sft.jsonl"
    canonical.parent.mkdir(parents=True)
    canonical.write_text('{"messages": []}\n', encoding="utf-8")
    manifest_path = root / "manifests/sft.yaml"
    manifest = DatasetManifest(dataset_id="ds", stage="sft", sample_count=1,
                               hashes={"canonical_records_sha256": sha256_file(canonical)})
    with pytest.raises(ValueError, match="canonical_path"):
        save_manifest(manifest_path, manifest)
    manifest.metadata["canonical_path"] = "../canonical/sft.jsonl"
    with pytest.raises(ValueError, match="inside"):
        save_manifest(manifest_path, manifest)
    manifest.metadata["canonical_path"] = "canonical/sft.jsonl"
    save_manifest(manifest_path, manifest)


def test_legacy_canonical_path_repair_uses_unique_hash_verified_processing_output(tmp_path):
    manifest_path = _legacy_artifact(tmp_path / "artifact")

    repaired = repair_manifest_canonical_path(manifest_path)

    assert repaired == "canonical/sft.jsonl"
    assert load_manifest(manifest_path).metadata["canonical_path"] == repaired


def test_legacy_canonical_path_repair_fails_closed_on_digest_mismatch(tmp_path):
    manifest_path = _legacy_artifact(tmp_path / "artifact")
    report_path = manifest_path.parent.parent / "reports/processing_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["outputs"]["sft"]["content_sha256"] = "0" * 64
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        repair_manifest_canonical_path(manifest_path)
    assert "canonical_path" not in load_manifest(manifest_path).metadata
