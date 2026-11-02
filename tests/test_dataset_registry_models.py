from pydantic import ValidationError
import pytest

from onegpu_posttrain.data.manifest import DatasetManifest
from onegpu_posttrain.data.registry import (
    DataSource,
    DatasetArtifact,
    DatasetBuild,
    SourceStageCapability,
)
from onegpu_posttrain.lineage import Digest


def test_legacy_manifest_without_new_origin_remains_readable():
    manifest = DatasetManifest.model_validate({
        "dataset_id": "legacy-sft",
        "stage": "sft",
        "sample_count": 1,
        "source": {"repo": "fixture", "revision": "r1"},
    })
    assert manifest.source.repo == "fixture"
    assert manifest.origin is None


def test_local_upload_origin_can_express_missing_provenance_without_fabrication():
    source = DataSource(
        source_id="source-upload-abc123",
        kind="local_upload",
        name="local file upload",
        revision_status="unknown",
        license_status="not_provided",
        declared_by="user",
    )
    manifest = DatasetManifest(
        dataset_id="ds-sft-abc123",
        stage="sft",
        sample_count=1,
        source=None,
        origin=source,
    )
    assert manifest.source is None
    assert manifest.origin.revision is None
    assert manifest.origin.license_id is None
    assert manifest.origin.license_status == "not_provided"


def test_source_stage_capability_models_independent_axes():
    capability = SourceStageCapability(
        source_id="model-generated",
        source_kind="model_generated",
        supported_stages=["sft", "preference"],
        input_format="canonical-jsonl",
    )
    assert capability.source_kind == "model_generated"
    assert capability.supported_stages == ["sft", "preference"]
    with pytest.raises(ValidationError):
        SourceStageCapability(
            source_id="local-upload",
            source_kind="local_upload",
            supported_stages=["cpt", "unknown"],
            input_format="canonical-jsonl",
        )


def test_dataset_build_and_artifact_share_source_stage_and_hash_contract():
    digest = "a" * 64
    build = DatasetBuild(
        run_id="run-import-sft-123456789abc",
        source_id="source-upload-abc123",
        stage="sft",
        protocol="canonical-import-v1",
        config_sha256=digest,
        input_digest=Digest(value=digest, status="available"),
        output_dataset_id="ds-sft-abc123",
        output_digest=Digest(value=digest, status="available"),
        status="completed",
    )
    artifact = DatasetArtifact(
        dataset_id=build.output_dataset_id,
        stage=build.stage,
        source_id=build.source_id,
        produced_by_run_id=build.run_id,
        record_count=1,
        canonical_sha256=digest,
    )
    assert artifact.source_id == build.source_id
    assert artifact.stage == build.stage
    assert artifact.produced_by_run_id == build.run_id
