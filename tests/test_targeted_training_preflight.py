from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from test_targeted_stage_plan import stage_plan

from onegpu_posttrain.data.manifest import load_manifest, save_manifest
from onegpu_posttrain.data.targeted_grpo import (
    ExecutableRewardContract,
    build_targeted_grpo_prompts,
    register_targeted_grpo_prompts,
)
from onegpu_posttrain.data.targeted_preference import (
    build_targeted_preference_pairs,
    register_targeted_preference_pairs,
)
from onegpu_posttrain.data.targeted_stage_plan import register_targeted_stage_plan
from onegpu_posttrain.data.targeted_training_preflight import (
    preflight_targeted_training,
)
from onegpu_posttrain.lineage import sha256_file
from onegpu_posttrain.evaluation import verification


@pytest.fixture(autouse=True)
def deterministic_optional_answer_verifier(monkeypatch):
    monkeypatch.setattr(verification, "_verification_identity", lambda: ("math-verify", "test"))
    monkeypatch.setattr(
        verification,
        "_run_bounded_batch",
        lambda pairs, _timeout: [
            "correct" if pair.prediction.strip() == pair.reference.strip() else "incorrect"
            for pair in pairs
        ],
    )


def _parent(tmp_path):
    path = tmp_path / "parent-model.bin"
    path.write_bytes(b"frozen parent fixture")
    digest = sha256_file(path)
    return path, {"model_id": f"model-{digest[:20]}", "content_sha256": digest}


def _preference(tmp_path):
    model, model_identity = _parent(tmp_path)
    plan = stage_plan("preference", starting_model=model_identity)
    register_targeted_stage_plan(tmp_path, plan)
    report = build_targeted_preference_pairs([
        {"candidate_id": "c1", "source_record_id": "seed-1", "problem_id": "p1",
         "source_role": "train_seed", "problem": "1+1?", "completion": r"\boxed{2}",
         "reference_answer": "2"},
        {"candidate_id": "c2", "source_record_id": "seed-1", "problem_id": "p1",
         "source_role": "train_seed", "problem": "1+1?", "completion": r"\boxed{3}",
         "reference_answer": "2"},
    ], plan=plan)
    registration = register_targeted_preference_pairs(
        report, plan=plan, workspace=tmp_path, output_root=tmp_path / "data/processed/generated",
        provider_profile_id="offline-fixture",
    )
    dataset = tmp_path / registration["registration_path"] / "canonical/preference.jsonl"
    return model, plan, dataset


def _grpo(tmp_path):
    model, model_identity = _parent(tmp_path)
    contract = ExecutableRewardContract.create()
    plan = stage_plan("grpo", starting_model=model_identity, target={
        "stage": "grpo", "scenario_id": "algebra",
        "reward_contract": {"evidence_id": contract.contract_id,
                            "content_sha256": contract.protocol_sha256,
                            "evidence_kind": "reward_contract"},
    })
    register_targeted_stage_plan(tmp_path, plan)
    report = build_targeted_grpo_prompts([
        {"id": "seed-1", "problem_id": "p1", "problem": "1+1?", "answer": "2"},
    ], plan=plan, contract=contract)
    registration = register_targeted_grpo_prompts(
        report, plan=plan, contract=contract, workspace=tmp_path,
        output_root=tmp_path / "data/processed/generated", provider_profile_id="offline-fixture",
    )
    dataset = tmp_path / registration["registration_path"] / "canonical/grpo.jsonl"
    return model, plan, dataset, registration


def test_preference_preflight_binds_plan_seed_parent_and_actual_dataset(tmp_path):
    model, plan, dataset = _preference(tmp_path)
    result = preflight_targeted_training(
        dataset=dataset, model=model, stage="dpo", workspace=tmp_path,
    )
    assert result["plan_id"] == plan.plan_id
    assert result["dataset_id"].startswith("ds-")
    assert result["record_count"] == 1
    assert result["starting_model_sha256"] == plan.starting_model.content_sha256
    assert result["preflight"] == "passed"


def test_preference_preflight_rejects_different_parent_and_cross_stage(tmp_path):
    model, _, dataset = _preference(tmp_path)
    other_parent = tmp_path / "other-model.bin"
    other_parent.write_bytes(b"different parent")
    with pytest.raises(ValueError, match="parent model content"):
        preflight_targeted_training(dataset=dataset, model=other_parent, stage="simpo", workspace=tmp_path)
    with pytest.raises(ValueError, match="cannot be used for grpo"):
        preflight_targeted_training(dataset=dataset, model=model, stage="grpo", workspace=tmp_path)


def test_grpo_preflight_requires_matching_executable_contract_and_plugin(tmp_path):
    model, plan, dataset, registration = _grpo(tmp_path)
    plugin = Path(__file__).resolve().parents[1] / "plugins/grpo_rewards.py"
    result = preflight_targeted_training(
        dataset=dataset, model=model, stage="grpo", workspace=tmp_path,
        external_plugin=plugin,
    )
    assert result["plan_id"] == plan.plan_id
    assert result["reward_contract"]["contract_id"].startswith("reward-contract-")

    invalid_plugin = tmp_path / "reward_plugin.py"
    invalid_plugin.write_text("# arbitrary reward implementation\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the verified project adapter"):
        preflight_targeted_training(
            dataset=dataset, model=model, stage="grpo", workspace=tmp_path,
            external_plugin=invalid_plugin,
        )

    manifest_path = tmp_path / registration["registration_path"] / "manifests/grpo-user-upload-v1.yaml"
    manifest = load_manifest(manifest_path)
    manifest.metadata["reward_contract_verified"] = False
    save_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="missing matching executable reward-contract"):
        preflight_targeted_training(
            dataset=dataset, model=model, stage="grpo", workspace=tmp_path,
            external_plugin=plugin,
        )

def test_grpo_cli_blocks_bad_targeted_reward_before_creating_run_or_launching(tmp_path, monkeypatch):
    model, _, dataset, registration = _grpo(tmp_path)
    manifest_path = tmp_path / registration["registration_path"] / "manifests/grpo-user-upload-v1.yaml"
    manifest = load_manifest(manifest_path)
    manifest.metadata.pop("reward_contract_protocol_sha256")
    save_manifest(manifest_path, manifest)
    plugin = Path(__file__).resolve().parents[1] / "plugins/grpo_rewards.py"
    config = tmp_path / "grpo.yaml"
    config.write_text(yaml.safe_dump({"profile": "standard", "training": {"max_steps": 1}}), encoding="utf-8")
    output = tmp_path / "should-not-start"

    from onegpu_posttrain import cli

    def unexpected_launch(*args, **kwargs):
        raise AssertionError("training subprocess must not start after a failed preflight")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "run_monitored_command", unexpected_launch)
    args = type("Args", (), {
        "config": str(config), "model": str(model), "dataset": str(dataset), "output": str(output),
        "external_plugin": str(plugin), "dry_run": False, "merge": False,
    })()
    with pytest.raises(ValueError, match="missing matching executable reward-contract"):
        cli.cmd_run_grpo(args)
    assert not output.exists()


def test_non_targeted_legacy_dataset_keeps_existing_cli_compatibility(tmp_path):
    dataset = tmp_path / "ordinary.jsonl"
    dataset.write_text('{"messages": []}\n', encoding="utf-8")
    model = tmp_path / "model.bin"
    model.write_bytes(b"parent")
    assert preflight_targeted_training(
        dataset=dataset, model=model, stage="dpo", workspace=tmp_path,
    ) is None
