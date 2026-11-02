import json

import pytest

from onegpu_posttrain.data.manifest import DatasetManifest, ReportReference, SourceSpec, save_manifest, sha256_file
from onegpu_posttrain.data.source_adapters import adapt_open_source_manifest, source_id_for_manifest
from onegpu_posttrain.lineage import DatasetLineage


@pytest.mark.parametrize(("source_id", "repository", "stage"), [
    ("finemath", "HuggingFaceTB/finemath", "cpt"),
    ("deepmath", "zwhe99/DeepMath-103K", "sft"),
    ("openr1", "open-r1/OpenR1-Math-220k", "preference"),
    ("openr1", "open-r1/OpenR1-Math-220k", "grpo"),
])
def test_existing_open_source_artifact_adapts_without_rewriting_cache(tmp_path, source_id, repository, stage):
    root = tmp_path / source_id
    canonical = root / "canonical" / f"{stage}.jsonl"
    canonical.parent.mkdir(parents=True)
    canonical.write_text('{"id":"fixture"}\n', encoding="utf-8")
    processing = root / "reports" / "processing_report.json"
    processing.parent.mkdir(parents=True)
    processing.write_text(json.dumps({
        "build_id": "build-fixture", "outputs": {stage: {
            "dataset_id": f"ds-{source_id}-{stage}", "path": f"canonical/{stage}.jsonl",
            "content_sha256": sha256_file(canonical),
        }},
    }), encoding="utf-8")
    processing_hash = sha256_file(processing)
    run_id = f"run-{source_id}-{stage}-fixture"
    manifest_path = root / "manifests" / f"{stage}.yaml"
    manifest = DatasetManifest(
        dataset_id=f"ds-{source_id}-{stage}", stage=stage,
        source=SourceSpec(repo=repository, revision="pinned-revision-1"), sample_count=1,
        hashes={"canonical_records_sha256": sha256_file(canonical)},
        processing_report=ReportReference(path="../reports/processing_report.json", sha256=processing_hash,
                                           report_id="build-fixture"),
        metadata={"canonical_path": f"canonical/{stage}.jsonl"},
        lineage=DatasetLineage(
            produced_by_run_id=run_id, transform={"name": "fixture", "version": "1"},
            source_evidence={"repo": repository, "revision": "pinned-revision-1"},
            record_provenance="dataset_only", count_unit="records",
            content_sha256=sha256_file(canonical),
        ),
    )
    save_manifest(manifest_path, manifest)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}

    adapted = adapt_open_source_manifest(manifest_path)

    assert source_id_for_manifest(manifest) == source_id
    assert adapted.dataset_id == manifest.dataset_id
    assert adapted.source_id == source_id
    assert adapted.produced_by_run_id == run_id
    assert adapted.canonical_sha256 == sha256_file(canonical)
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
