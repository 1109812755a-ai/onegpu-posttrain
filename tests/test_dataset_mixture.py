import json
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from onegpu_posttrain.data.dataset_import import register_jsonl
from onegpu_posttrain.data.dataset_mixture import (
    MixtureInput,
    build_mixture,
    preflight_mixture,
)
from onegpu_posttrain.data.manifest import (
    DatasetParent,
    load_manifest,
    save_manifest,
    validate_manifest_evidence,
)
from onegpu_posttrain.data.registry import DataSource
from onegpu_posttrain.lineage import ArtifactRef, Digest, RunRecord
from onegpu_posttrain.lineage_validation import validate_lineage_root
from onegpu_posttrain.ui.server import create_app
from onegpu_posttrain.ui.stage_profiles import stage_profile_catalog
from onegpu_posttrain.ui.training_preflight import (
    _inspect_dataset,
    stage_workbench_options,
)


def _origin(**changes):
    data = {"source_id": "mixture-test-source", "kind": "local_upload", "name": "test source",
            "declared_by": "user", "license_status": "unknown"}
    data.update(changes)
    return DataSource(**data)


def _records(stage: str, ids: list[str]):
    rows = []
    for index, record_id in enumerate(ids):
        if stage == "cpt":
            row = {"id": record_id, "text": f"Document {index} with useful content.", "source": "unit-test"}
        elif stage == "sft":
            row = {"id": record_id, "raw_problem": f"What is {index}+1?", "messages": [
                {"role": "user", "content": f"What is {index}+1?"},
                {"role": "assistant", "content": str(index + 1)},
            ], "prompt_template_id": "chat-v1"}
        elif stage == "preference":
            row = {"id": record_id, "raw_problem": f"Choose {index}", "prompt": [
                {"role": "user", "content": f"Choose {index}"}], "chosen": f"Good {index}",
                "rejected": f"Bad {index}", "prompt_template_id": "chat-v1"}
        else:
            contract = _grpo_contract()
            row = {"id": record_id, "raw_problem": f"Solve {index}", "prompt": [
                {"role": "user", "content": f"Solve {index}"}], "answer": str(index),
                "prompt_template_id": "chat-v1", "metadata": {
                    "reward_contract_id": contract.contract_id,
                    "reward_contract_protocol_sha256": contract.protocol_sha256,
                    "reward_contract_implementation_sha256": contract.implementation_sha256,
                    "problem_id": record_id, "data_role": "train_seed",
                    "reference_source": "registered_train_seed",
                }}
        rows.append(row)
    return b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in rows)


def _register_parent(root, stage, ids, *, role="train", metadata=None):
    registration_metadata = {"dataset_role": role, **(metadata or {})}
    if stage == "grpo":
        contract = _grpo_contract()
        registration_metadata.setdefault("reward_contract_verified", True)
        registration_metadata.setdefault("reward_contract_id", contract.contract_id)
        registration_metadata.setdefault("reward_contract_protocol_sha256", contract.protocol_sha256)
        registration_metadata.setdefault("reward_contract_implementation_sha256", contract.implementation_sha256)
        plugin = Path(__file__).resolve().parents[1] / "plugins" / "grpo_rewards.py"
        target = root / "plugins" / "grpo_rewards.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(plugin.read_bytes())
    return register_jsonl(
        BytesIO(_records(stage, ids)), workspace=root,
        output_root=root / "data" / "processed" / "user-upload", stage=stage,
        origin=_origin(source_id=f"source-{ids[0]}"),
        registration_metadata=registration_metadata,
    )


def _grpo_contract():
    from onegpu_posttrain.data.targeted_grpo import ExecutableRewardContract
    return ExecutableRewardContract.create()


def _build(root, stage, datasets, **overrides):
    request = MixtureInput(stage=stage, dataset_ids=[dataset.dataset_id for dataset in datasets], **overrides)
    return build_mixture(root, request)


@pytest.mark.parametrize("stage", ["cpt", "sft", "preference", "grpo"])
def test_concat_mixture_registers_stage_canonical_dataset_with_multi_parent_lineage(tmp_path, stage):
    parents = [_register_parent(tmp_path, stage, [f"{stage}-a", f"{stage}-b"]),
               _register_parent(tmp_path, stage, [f"{stage}-c"])]

    result, summary = _build(tmp_path, stage, parents, display_name="阶段组合测试集")
    manifest_path = result.path / "manifests" / f"{stage}-dataset-mixture-v1.yaml"
    manifest = load_manifest(manifest_path)

    assert result.record_count == 3
    assert summary["dataset_id"] == result.dataset_id
    assert summary["display_name"] == "阶段组合测试集"
    assert manifest.metadata["display_name"] == "阶段组合测试集"
    assert manifest.composition is not None
    assert manifest.lineage.parents == [parent.dataset_id for parent in parents]
    assert manifest.lineage.record_provenance == "record"
    assert (result.path / "provenance" / "records.jsonl").is_file()
    assert not validate_manifest_evidence(manifest_path)
    quality_path = manifest_path.parent / manifest.quality_report.path
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    assert quality["content_sha256"] == manifest.hashes["canonical_records_sha256"]
    assert quality["dataset_id"] == result.dataset_id
    assert quality["summaries"]["mixture"]["strategy"] == "concat"
    assert quality["summaries"]["mixture"]["source_record_counts"] == summary["source_record_counts"]
    if stage == "preference":
        composed = [json.loads(line) for line in (result.path / "canonical" / "preference.jsonl").read_text().splitlines()]
        assert [(row["chosen"], row["rejected"]) for row in composed] == [("Good 0", "Bad 0"), ("Good 1", "Bad 1"), ("Good 0", "Bad 0")]


def test_mixture_name_is_required_to_be_nonblank_and_bounded():
    with pytest.raises(ValueError, match="名称不能为空"):
        MixtureInput(stage="sft", dataset_ids=["one", "two"], display_name="  ")
    with pytest.raises(ValueError):
        MixtureInput(stage="sft", dataset_ids=["one", "two"], display_name="x" * 121)


def test_weighted_sample_is_deterministic_and_preserves_record_source_provenance(tmp_path):
    parents = [_register_parent(tmp_path, "sft", [f"a-{i}" for i in range(4)]),
               _register_parent(tmp_path, "sft", [f"b-{i}" for i in range(4)])]
    request = MixtureInput(stage="sft", dataset_ids=[p.dataset_id for p in parents],
                           strategy="weighted_sample", budget=5, seed=19,
                           weights={parents[0].dataset_id: 3, parents[1].dataset_id: 1})

    first, detail = build_mixture(tmp_path, request)
    second, _ = build_mixture(tmp_path, request)

    assert first.dataset_id == second.dataset_id
    assert first.canonical_sha256 == second.canonical_sha256
    assert first.reused is False and second.reused is True
    assert first.record_count == 5
    provenance = [json.loads(line) for line in (first.path / "provenance" / "records.jsonl").read_text().splitlines()]
    assert len(provenance) == 5
    assert {row["dataset_id"] for row in provenance} == {p.dataset_id for p in parents}
    assert len({row["record_id"] for row in provenance}) == 5
    assert sum(detail["source_record_counts"].values()) == 5


def test_preflight_rejects_mixed_stage_holdout_and_budget_overflow(tmp_path):
    sft = _register_parent(tmp_path, "sft", ["s1", "s2"])
    preference = _register_parent(tmp_path, "preference", ["p1"])
    with pytest.raises(ValueError):
        preflight_mixture(tmp_path, MixtureInput(stage="sft", dataset_ids=[sft.dataset_id, preference.dataset_id]))

    holdout = _register_parent(tmp_path, "sft", ["h1"], role="final_holdout")
    with pytest.raises(ValueError, match="final_holdout"):
        preflight_mixture(tmp_path, MixtureInput(stage="sft", dataset_ids=[sft.dataset_id, holdout.dataset_id]))

    another_train = _register_parent(tmp_path, "sft", ["t1"])
    with pytest.raises(ValueError, match="超过可用记录总数"):
        preflight_mixture(tmp_path, MixtureInput(stage="sft", dataset_ids=[sft.dataset_id, another_train.dataset_id],
                                                  strategy="weighted_sample", budget=20))


@pytest.mark.parametrize("weights", [{"x": 0, "y": 1}, {"x": -1, "y": 1}, {"x": float("inf"), "y": 1}])
def test_weighted_sampling_rejects_non_positive_or_non_finite_weights(weights):
    with pytest.raises(ValueError):
        MixtureInput(stage="sft", dataset_ids=["x", "y"], strategy="weighted_sample", budget=1, weights=weights)


def test_mixture_requires_two_unique_parents():
    with pytest.raises(ValueError):
        MixtureInput(stage="sft", dataset_ids=["only-one"])
    with pytest.raises(ValueError):
        MixtureInput(stage="sft", dataset_ids=["same", "same"])


def test_preflight_blocks_holdout_in_composition_ancestor(tmp_path):
    first = _register_parent(tmp_path, "sft", ["ancestor-a"])
    second = _register_parent(tmp_path, "sft", ["ancestor-b"])
    combined, _ = _build(tmp_path, "sft", [first, second])
    first_manifest_path = first.path / "manifests" / "sft-user-upload-v1.yaml"
    first_manifest = load_manifest(first_manifest_path)
    first_manifest.metadata["dataset_role"] = "diagnostic_dev"
    save_manifest(first_manifest_path, first_manifest)
    third = _register_parent(tmp_path, "sft", ["ancestor-c"])

    with pytest.raises(ValueError, match="diagnostic_dev"):
        preflight_mixture(tmp_path, MixtureInput(stage="sft", dataset_ids=[combined.dataset_id, third.dataset_id]))


def test_grpo_requires_matching_verified_reward_contract(tmp_path):
    left = _register_parent(tmp_path, "grpo", ["g-left"])
    right = _register_parent(tmp_path, "grpo", ["g-right"], metadata={"reward_contract_id": "reward-b"})
    with pytest.raises(ValueError, match="契约"):
        preflight_mixture(tmp_path, MixtureInput(stage="grpo", dataset_ids=[left.dataset_id, right.dataset_id]))


def test_lineage_validator_checks_composition_parent_digest_and_run_inputs(tmp_path):
    parents = [_register_parent(tmp_path, "sft", ["left"]), _register_parent(tmp_path, "sft", ["right"])]
    result, _ = _build(tmp_path, "sft", parents, display_name="阶段组合测试集")

    report = validate_lineage_root(tmp_path)
    assert report["valid"] is True
    mixture_manifest = load_manifest(result.path / "manifests" / "sft-dataset-mixture-v1.yaml")
    assert mixture_manifest.composition.parents[0].content_sha256 == parents[0].canonical_sha256
    options = stage_workbench_options(tmp_path)
    assert result.dataset_id in {item["dataset_id"] for item in options["datasets"]}
    option = next(item for item in options["datasets"] if item["dataset_id"] == result.dataset_id)
    assert option["eligible"] is True
    assert option["display_name"] == "阶段组合测试集"
    profile = next(item for item in stage_profile_catalog(tmp_path).profiles if item.profile_id == "sft-mini-1024-v1")
    _, blockers, _ = _inspect_dataset(tmp_path, result.path / "manifests" / "sft-dataset-mixture-v1.yaml",
                                      mixture_manifest, profile)
    assert blockers == []


def test_stage_selector_marks_legacy_manifest_without_registration_evidence_ineligible(tmp_path):
    registered = _register_parent(tmp_path, "preference", ["pair-a"])
    path = next((registered.path / "manifests").glob("*.yaml"))
    manifest = load_manifest(path)
    metadata = dict(manifest.metadata)
    metadata.pop("canonical_path", None)
    save_manifest(path, manifest.model_copy(update={
        "lineage": None, "processing_report": None, "metadata": metadata,
    }), allow_legacy=True)

    option = next(item for item in stage_workbench_options(tmp_path)["datasets"]
                  if item["dataset_id"] == registered.dataset_id)

    assert option["eligible"] is False
    assert "processing report" in option["block_reason"]


def test_stage_selector_blocks_grpo_without_registered_executable_contract(tmp_path):
    registered = _register_parent(
        tmp_path, "grpo", ["grpo-missing-contract"],
        metadata={"reward_contract_verified": False},
    )

    option = next(item for item in stage_workbench_options(tmp_path)["datasets"]
                  if item["dataset_id"] == registered.dataset_id)

    assert option["eligible"] is False
    assert "reward_contract_verified" in option["block_reason"]


@pytest.mark.parametrize("stage", ["cpt", "sft", "preference", "grpo"])
def test_four_stage_mixture_quality_train_run_and_recursive_lineage(tmp_path, stage):
    parents = [_register_parent(tmp_path, stage, [f"e2e-{stage}-a"]),
               _register_parent(tmp_path, stage, [f"e2e-{stage}-b"])]
    composed, summary = _build(tmp_path, stage, parents)
    manifest_path = composed.path / "manifests" / f"{stage}-dataset-mixture-v1.yaml"
    manifest = load_manifest(manifest_path)
    quality = json.loads((manifest_path.parent / manifest.quality_report.path).read_text(encoding="utf-8"))

    train_run = RunRecord(
        run_id=f"run-{stage}-mixture-e2e", run_type="train", status="completed",
        started_at="2026-10-24T00:00:00Z", recipe_path=f"recipes/{stage}-fixture.yaml",
        resolved_config_sha256="a" * 64,
        inputs=[ArtifactRef(
            id=composed.dataset_id, kind="dataset",
            path=str(composed.path / "canonical" / f"{stage}.jsonl"),
            digest=Digest(value=composed.canonical_sha256, status="available"), role="train",
        )],
    )
    run_path = tmp_path / "runs" / train_run.run_id / "lineage.json"
    run_path.parent.mkdir(parents=True)
    run_path.write_text(train_run.model_dump_json(indent=2), encoding="utf-8")

    report = validate_lineage_root(tmp_path, require_complete=True)
    assert report["valid"], report["errors"]
    assert quality["dataset_id"] == composed.dataset_id
    assert quality["content_sha256"] == composed.canonical_sha256
    assert summary["dataset_id"] == composed.dataset_id
    assert len(train_run.inputs) == 1
    assert train_run.inputs[0].id == composed.dataset_id
    assert manifest.composition is not None
    assert {parent.dataset_id for parent in manifest.composition.parents} == {p.dataset_id for p in parents}


def test_manifest_evidence_rejects_changed_provenance_sidecar(tmp_path):
    parents = [_register_parent(tmp_path, "sft", ["prov-a"]), _register_parent(tmp_path, "sft", ["prov-b"])]
    result, _ = _build(tmp_path, "sft", parents)
    manifest_path = result.path / "manifests" / "sft-dataset-mixture-v1.yaml"
    provenance = result.path / "provenance" / "records.jsonl"
    provenance.write_text(provenance.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    assert any("provenance sha256 mismatch" in error for error in validate_manifest_evidence(manifest_path))


@pytest.mark.parametrize("corruption,expected_code", [
    ("missing_parent", "dataset_composition_parent_missing"),
    ("parent_digest", "dataset_composition_parent_digest_mismatch"),
    ("cycle", "lineage_cycle"),
])
def test_lineage_validator_rejects_broken_or_cyclic_composition_parent(corruption, expected_code, tmp_path):
    parents = [_register_parent(tmp_path, "sft", [f"broken-{corruption}-a"]),
               _register_parent(tmp_path, "sft", [f"broken-{corruption}-b"])]
    result, _ = _build(tmp_path, "sft", parents)
    manifest_path = result.path / "manifests" / "sft-dataset-mixture-v1.yaml"
    manifest = load_manifest(manifest_path)
    declared = list(manifest.composition.parents)
    if corruption == "missing_parent":
        declared[0] = DatasetParent(dataset_id="ds-sft-missing-parent", content_sha256="1" * 64)
    elif corruption == "parent_digest":
        declared[0] = declared[0].model_copy(update={"content_sha256": "1" * 64})
    else:
        declared[0] = DatasetParent(dataset_id=result.dataset_id, content_sha256=result.canonical_sha256)
    manifest.composition = manifest.composition.model_copy(update={"parents": declared})
    manifest.lineage = manifest.lineage.model_copy(update={"parents": [parent.dataset_id for parent in declared]})
    save_manifest(manifest_path, manifest)

    report = validate_lineage_root(tmp_path)
    assert any(item["code"] == expected_code for item in report["errors"])


def test_parent_order_changes_composition_identity(tmp_path):
    first = _register_parent(tmp_path, "sft", ["order-a"])
    second = _register_parent(tmp_path, "sft", ["order-b"])
    forward = preflight_mixture(tmp_path, MixtureInput(stage="sft", dataset_ids=[first.dataset_id, second.dataset_id]))
    reverse = preflight_mixture(tmp_path, MixtureInput(stage="sft", dataset_ids=[second.dataset_id, first.dataset_id]))
    assert forward["config_sha256"] != reverse["config_sha256"]


def test_sampling_seed_weight_and_budget_are_part_of_identity(tmp_path):
    parents = [_register_parent(tmp_path, "sft", [f"identity-{i}"]) for i in range(2)]
    ids = [parent.dataset_id for parent in parents]
    requests = [
        MixtureInput(stage="sft", dataset_ids=ids, strategy="weighted_sample", budget=1, seed=1,
                     weights={ids[0]: 3, ids[1]: 1}),
        MixtureInput(stage="sft", dataset_ids=ids, strategy="weighted_sample", budget=1, seed=2,
                     weights={ids[0]: 3, ids[1]: 1}),
        MixtureInput(stage="sft", dataset_ids=ids, strategy="weighted_sample", budget=2, seed=1,
                     weights={ids[0]: 3, ids[1]: 1}),
        MixtureInput(stage="sft", dataset_ids=ids, strategy="weighted_sample", budget=1, seed=1,
                     weights={ids[0]: 1, ids[1]: 3}),
    ]
    hashes = {preflight_mixture(tmp_path, request)["config_sha256"] for request in requests}
    assert len(hashes) == 4


def test_failed_registration_leaves_no_mixture_artifact_or_temporary_files(tmp_path, monkeypatch):
    import onegpu_posttrain.data.dataset_mixture as mixture_module

    parents = [_register_parent(tmp_path, "sft", ["fail-a"]), _register_parent(tmp_path, "sft", ["fail-b"])]
    def fail_registration(*args, **kwargs):
        raise ValueError("simulated atomic registration failure")
    monkeypatch.setattr(mixture_module, "register_jsonl", fail_registration)

    with pytest.raises(ValueError, match="simulated atomic"):
        _build(tmp_path, "sft", parents)

    assert not (tmp_path / "data" / "processed" / "dataset-mixtures").exists()
    assert not list(tmp_path.glob(".dataset-mixture-*"))


def test_mixture_api_preflight_and_build(tmp_path):
    parents = [_register_parent(tmp_path, "sft", ["api-a"]), _register_parent(tmp_path, "sft", ["api-b"])]
    payload = {"stage": "sft", "dataset_ids": [parent.dataset_id for parent in parents], "strategy": "concat"}
    with TestClient(create_app(tmp_path)) as client:
        preflight = client.post("/api/data/mixtures/preflight", json=payload)
        assert preflight.status_code == 200
        built = client.post("/api/data/mixtures/build", json=payload)
    assert built.status_code == 201
    assert built.json()["record_count"] == 2


def test_mixture_api_returns_conflict_details_and_does_not_build_on_invalid_selection(tmp_path):
    sft = _register_parent(tmp_path, "sft", ["api-sft"])
    preference = _register_parent(tmp_path, "preference", ["api-pref"])
    payload = {"stage": "sft", "dataset_ids": [sft.dataset_id, preference.dataset_id], "strategy": "concat"}
    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/api/data/mixtures/preflight", json=payload)
    assert response.status_code == 422
    assert not (tmp_path / "data" / "processed" / "dataset-mixtures").exists()


def test_mixture_api_rejects_parent_digest_change_and_invalid_strategy_parameters(tmp_path):
    first = _register_parent(tmp_path, "sft", ["api-hash-a"])
    second = _register_parent(tmp_path, "sft", ["api-hash-b"])
    canonical = first.path / "canonical" / "sft.jsonl"
    canonical.write_text(canonical.read_text(encoding="utf-8") + " ", encoding="utf-8")
    payload = {"stage": "sft", "dataset_ids": [first.dataset_id, second.dataset_id], "strategy": "concat"}
    invalid_weights = {"stage": "sft", "dataset_ids": [first.dataset_id, second.dataset_id],
                       "strategy": "weighted_sample", "budget": 1, "weights": {
                           first.dataset_id: 0, second.dataset_id: 1}}
    with TestClient(create_app(tmp_path)) as client:
        digest_response = client.post("/api/data/mixtures/preflight", json=payload)
        weight_response = client.post("/api/data/mixtures/preflight", json=invalid_weights)
    assert digest_response.status_code == 422
    assert weight_response.status_code == 422
    assert not (tmp_path / "data" / "processed" / "dataset-mixtures").exists()
