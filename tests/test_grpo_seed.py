import json
import hashlib
import shutil
from io import BytesIO
from pathlib import Path
from fastapi.testclient import TestClient

import pytest

from onegpu_posttrain.data.dataset_import import register_jsonl
from onegpu_posttrain.data.grpo_seed import (
    GRPOSeedRequest,
    build_grpo_seed_dataset,
    list_grpo_seed_candidates,
    preflight_grpo_seed,
)
from onegpu_posttrain.data.manifest import SourceSpec, load_manifest, save_manifest
from onegpu_posttrain.data.registry import DataSource
from onegpu_posttrain.lineage_validation import validate_lineage_root
from onegpu_posttrain.ui.server import create_app
from onegpu_posttrain.ui.training_preflight import _inspect_grpo_reward_contract


def _register_sft_seed(root, *, trusted=True, record_answer="7", correctness=True):
    row = {
        "id": "sft-row-1", "raw_problem": "What is 3+4?",
        "messages": [{"role": "user", "content": "What is 3+4?"},
                     {"role": "assistant", "content": "7"}],
        "prompt_template_id": "qwen3-math-v1",
        "metadata": {"problem_id": "problem-1", "source_record_id": "openr1:uuid-1",
                     "correctness_math_verify": correctness, "parsed_answer": record_answer},
    }
    payload = (json.dumps(row) + "\n").encode()
    result = register_jsonl(
        BytesIO(payload), workspace=root, output_root=root / "data/processed/seeds",
        stage="sft", origin=DataSource(source_id="seed-source", kind="local_upload", name="fixture",
                                       declared_by="user", license_status="unknown"),
    )
    manifest_path = result.path / "manifests/sft-user-upload-v1.yaml"
    manifest = load_manifest(manifest_path)
    if trusted:
        manifest.source = SourceSpec(repo="open-r1/OpenR1-Math-220k", revision="a" * 40)
        manifest.metadata["build_protocol"] = "openr1-build-v1"
        manifest.metadata["dataset_role"] = "train_seed"
        manifest.lineage.transform = {"name": "build_openr1_artifacts", "version": "openr1-build-v1",
                                      "config_sha256": "b" * 64}
        manifest.lineage.source_evidence = {"repo": "open-r1/OpenR1-Math-220k", "revision": "a" * 40}
        report_path = manifest_path.parent / manifest.processing_report.path
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["processing_protocol"] = "openr1-build-v1"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        manifest.processing_report.sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
        run_path = manifest_path.parent.parent / "runs" / manifest.lineage.produced_by_run_id / "lineage.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run["inputs"] = [{"id": "open-r1/OpenR1-Math-220k", "kind": "file", "role": "source"}]
        for output in run.get("outputs", []):
            if output.get("kind") == "processing_report":
                output["digest"] = {"algorithm": "sha256", "status": "available",
                                    "value": manifest.processing_report.sha256}
        run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
        save_manifest(manifest_path, manifest)
    return result.dataset_id


def test_ordinary_grpo_seed_build_registers_quality_and_parent_lineage(tmp_path):
    dataset_id = _register_sft_seed(tmp_path)
    request = GRPOSeedRequest(dataset_ids=[dataset_id], target_count=1, seed=7)
    candidates = list_grpo_seed_candidates(tmp_path)
    assert candidates[0]["eligible_count"] == 1
    preflight = preflight_grpo_seed(tmp_path, request)
    assert preflight["available_count"] == 1
    assert "OpenR1" in preflight["reference_evidence"]

    result = build_grpo_seed_dataset(tmp_path, request)
    assert result["record_count"] == 1
    manifest_path = tmp_path / "data/processed/grpo-seed-derived" / result["dataset_id"] / "manifests/grpo-dataset-derived-v1.yaml"
    manifest = load_manifest(manifest_path)
    assert manifest.lineage.parents == [dataset_id]
    assert manifest.metadata["derivation_protocol"] == "grpo-openr1-seed-v1"
    assert manifest.metadata["reward_contract_verified"] is True
    assert manifest.metadata["derivation_preflight"]["available_count"] == 1
    assert manifest.metadata["derivation_preflight_sha256"] == result["derivation_preflight_sha256"]
    row = json.loads((manifest_path.parent.parent / manifest.metadata["canonical_path"]).read_text(encoding="utf-8").splitlines()[0])
    assert row["metadata"]["source_dataset_id"] == dataset_id
    assert row["metadata"]["source_dataset_sha256"] == manifest.lineage.source_evidence["registration_context"]["parent_dataset_refs"][0]["content_sha256"]
    assert "targeted_plan_id" not in row["metadata"]
    assert row["metadata"]["derivation_preflight_sha256"] == result["derivation_preflight_sha256"]
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[1] / "plugins/grpo_rewards.py", plugin_dir / "grpo_rewards.py")
    contract, blockers, checks = _inspect_grpo_reward_contract(tmp_path, manifest_path, manifest)
    assert blockers == []
    assert contract["status"] == "verified"
    assert next(item for item in checks if item["check_id"] == "grpo:reference_rows")["status"] == "pass"
    assert validate_lineage_root(tmp_path)["errors"] == []
    repeated = build_grpo_seed_dataset(tmp_path, request)
    assert repeated["dataset_id"] == result["dataset_id"]
    assert repeated["reused"] is True


@pytest.mark.parametrize("trusted,answer,correct", [(False, "7", True), (True, "", True), (True, "7", False)])
def test_grpo_seed_builder_fails_closed_on_untrusted_or_incomplete_evidence(tmp_path, trusted, answer, correct):
    dataset_id = _register_sft_seed(tmp_path, trusted=trusted, record_answer=answer, correctness=correct)
    request = GRPOSeedRequest(dataset_ids=[dataset_id], target_count=1)
    if not trusted:
        with pytest.raises(ValueError, match="build protocol"):
            preflight_grpo_seed(tmp_path, request)
    else:
        report = preflight_grpo_seed(tmp_path, request)
        assert report["eligible_count"] == 0
        with pytest.raises(ValueError, match="少于目标"):
            build_grpo_seed_dataset(tmp_path, request)


def test_dataset_import_does_not_accept_user_forged_derived_metadata(tmp_path):
    source = DataSource(source_id="fake-derived", kind="dataset_derived", name="fake", declared_by="user")
    with pytest.raises(ValueError, match="requires local_upload"):
        register_jsonl(BytesIO(b"{}\n"), workspace=tmp_path, output_root=tmp_path / "out",
                       stage="grpo", origin=source, registration_metadata={
                           "derivation_protocol": "grpo-openr1-seed-v1", "reward_contract_verified": True,
                           "reward_contract_id": "reward-contract-" + "0" * 20,
                           "reward_contract_protocol_sha256": "0" * 64,
                           "parent_dataset_ids": [], "parent_dataset_refs": [],
                       })


def test_grpo_seed_api_resolves_source_and_contract_on_server(tmp_path):
    dataset_id = _register_sft_seed(tmp_path)
    client = TestClient(create_app(tmp_path))
    candidates = client.get("/api/data/grpo-seeds")
    assert candidates.status_code == 200
    assert candidates.json()["candidates"][0]["dataset_id"] == dataset_id
    payload = {"dataset_ids": [dataset_id], "target_count": 1, "seed": 42}
    preflight = client.post("/api/data/grpo-seeds/preflight", json=payload)
    assert preflight.status_code == 200
    assert preflight.json()["available_count"] == 1
    assert "许可状态沿用父数据登记" in "".join(preflight.json()["limitations"])
    built = client.post("/api/data/grpo-seeds/build", json=payload)
    assert built.status_code == 201
    result = built.json()
    assert result["quality_status"] == "report_only"
    assert result["reward_contract"]["contract_id"] == preflight.json()["reward_contract"]["contract_id"]
    manifest_path = tmp_path / result["registration_path"] / "manifests/grpo-dataset-derived-v1.yaml"
    manifest = load_manifest(manifest_path)
    assert manifest.metadata["reward_contract_verified"] is True
    assert manifest.metadata["parent_dataset_ids"] == [dataset_id]
    assert manifest.origin.license_status == "not_provided"
    assert validate_lineage_root(tmp_path)["errors"] == []
    assert client.post("/api/data/grpo-seeds/build", json=payload).json()["dataset_id"] == result["dataset_id"]

    untrusted_id = _register_sft_seed(tmp_path, trusted=False, record_answer="8")
    rejected = client.post("/api/data/grpo-seeds/preflight", json={"dataset_ids": [untrusted_id], "target_count": 1})
    assert rejected.status_code == 422
    assert "build protocol" in rejected.json()["detail"]
    # The browser cannot inject its own reward IDs, correctness evidence, or lineage assertions.
    assert client.post("/api/data/grpo-seeds/preflight", json=payload | {"reward_contract_id": "fake"}).status_code == 422
    assert client.post("/api/data/grpo-seeds/build", json=payload | {"source": "fake OpenR1"}).status_code == 422
