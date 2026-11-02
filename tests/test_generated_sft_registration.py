from __future__ import annotations

import hashlib

import yaml
from test_sft_generation_workflow import _responses
from test_sft_workflow_bundle import _inputs

from onegpu_posttrain.data.generated_sft_registration import (
    register_generated_sft_dataset,
)
from onegpu_posttrain.data.manifest import load_manifest, validate_manifest_evidence
from onegpu_posttrain.synthetic.providers import FakeProvider
from onegpu_posttrain.synthetic.sft_workflow import execute_sft_generation
from onegpu_posttrain.synthetic.workflow_bundle import register_sft_workflow_bundle


def _generated(tmp_path):
    workspace, inputs = _inputs(tmp_path)
    bundle = register_sft_workflow_bundle(workspace, inputs=inputs)
    result = execute_sft_generation(
        workspace, bundle.bundle_id, execution_id="sftgen-register-0001",
        usable_target=2, max_requests=2, target_candidates=2,
        provider=FakeProvider(_responses(inputs)),
    )
    state = {"status": result["status"], "result": result, "bundle_id": bundle.bundle_id,
             "execution_id": result["execution_id"]}
    return workspace, state


def test_reviewed_generation_registers_dataset_with_quality_and_lineage(tmp_path):
    workspace, state = _generated(tmp_path)

    registration = register_generated_sft_dataset(workspace, state)
    manifest_path = workspace / registration["manifest_path"]
    manifest = load_manifest(manifest_path)

    assert registration["count"] > 0
    assert manifest.dataset_id == registration["dataset_id"]
    assert manifest.stage == "sft"
    assert manifest.sample_count == registration["count"]
    assert manifest.origin.kind == "model_generated"
    assert manifest.origin.license_status == "not_provided"
    assert manifest.metadata["quality_status"] == "report_only"
    assert manifest.metadata["generation_execution_id"] == state["execution_id"]
    evidence = manifest.lineage.source_evidence
    assert evidence["provider_identity_status"] == "profile_declared_not_independently_verified"
    assert evidence["registered_model_association_status"] == "not_proven"
    assert evidence["registered_model_id"] is None
    assert evidence["registered_model_sha256"] is None
    assert evidence["provider_model"]
    assert evidence["provider_revision_status"] in {"available", "unavailable"}
    assert manifest.hashes["canonical_records_sha256"] == registration["canonical_sha256"]
    assert validate_manifest_evidence(manifest_path) == []

    # Repeating registration returns the same immutable bytes and does not rewrite the manifest.
    before = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    again = register_generated_sft_dataset(workspace, state)
    after = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert again["dataset_id"] == registration["dataset_id"]
    assert again["reused"] is True
    assert before == after


def test_empty_generated_sft_output_cannot_be_registered(tmp_path):
    workspace, state = _generated(tmp_path)
    result = state["result"]
    artifacts = result["generation_artifacts"]
    canonical = workspace / artifacts["training_path"]
    canonical.write_bytes(b"")
    manifest_path = workspace / artifacts["training_manifest"]
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["hashes"]["canonical_records_sha256"] = hashlib.sha256(b"").hexdigest()
    manifest_path.write_text(yaml.safe_dump(manifest, allow_unicode=True), encoding="utf-8")
    result["usable_records"] = 0

    try:
        register_generated_sft_dataset(workspace, state)
    except ValueError as exc:
        assert "empty or schema-invalid" in str(exc)
    else:
        raise AssertionError("empty generated data must not be registered")
