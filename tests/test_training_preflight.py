import io
import json
import time
from argparse import Namespace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from onegpu_posttrain import cli, subprocess_utils
from onegpu_posttrain.data.dataset_import import register_jsonl
from onegpu_posttrain.data.registry import DataSource
from onegpu_posttrain.data.targeted_grpo import ExecutableRewardContract
from onegpu_posttrain.lineage import (
    ArtifactRef,
    ModelLineage,
    RunRecord,
    sha256_directory,
)
from onegpu_posttrain.ui.grpo_stage_adapter import GRPOStageAdapter, GRPOStageRequest
from onegpu_posttrain.ui import launcher as launcher_module
from onegpu_posttrain.ui.server import create_app
from onegpu_posttrain.ui.stage_training import RegisteredStageTraining, RegisteredStageTrainingRequest
from onegpu_posttrain.ui.training_preflight import (
    StagePreflightRequest,
    preflight_stage,
)
from onegpu_posttrain.ui.training_validation import inspect_registered_validation_dataset


def _dataset(workspace, stage="sft", *, role=None, license_status="not_provided", preference_evidence=False,
             variant=""):
    rows = {
        "sft": {"id": f"sft-1{variant}", "raw_problem": f"1+1?{variant}", "messages": [
            {"role": "user", "content": f"1+1?{variant}"}, {"role": "assistant", "content": "2"}],
            "prompt_template_id": "math-v1"},
        "preference": {"id": "pref-1", "raw_problem": "1+1?", "prompt": [
            {"role": "user", "content": "1+1?"}], "chosen": "2", "rejected": "3",
            "prompt_template_id": "math-v1"},
        "grpo": {"id": "grpo-1", "raw_problem": "1+1?", "prompt": [
            {"role": "user", "content": "1+1?"}], "answer": "2", "prompt_template_id": "math-v1"},
    }
    row = rows[stage]
    if stage == "preference" and preference_evidence:
        row["metadata"] = {"chosen_correct": True, "rejected_correct": False,
                            "parent_record_ids": ["chosen-1", "rejected-1"]}
    if role:
        row.setdefault("metadata", {})["dataset_role"] = role
    encoded = (json.dumps(row, ensure_ascii=False) + "\n").encode()
    result = register_jsonl(
        io.BytesIO(encoded), workspace=workspace,
        output_root=workspace / "data" / "processed" / "preflight-fixture",
        stage=stage,
        origin=DataSource(source_id="fixture", kind="local_upload", name="fixture",
                          license_status=license_status, declared_by="user"),
    )
    return result.dataset_id


def _parent_model(workspace, stage="cpt", *, nested_merge=False):
    run_id = f"run-{stage}-fixture"
    model_id = f"model-{stage}-fixture"
    model_dir = workspace / "models" / model_id
    model_dir.mkdir(parents=True)
    (model_dir / "weights.txt").write_text("local model fixture", encoding="utf-8")
    digest = sha256_directory(model_dir)
    output = workspace / "runs" / run_id
    artifacts = output / "model_artifacts"
    artifacts.mkdir(parents=True)
    model = ModelLineage(model_id=model_id, kind="merged", path=str(model_dir),
                         digest={"value": digest, "status": "available"},
                         parent_model_ids=["base-fixture"], produced_by_run_id=run_id, stage=stage)
    (artifacts / f"{model_id}.json").write_text(model.model_dump_json(), encoding="utf-8")
    run = RunRecord(
        run_id=run_id, run_type="train", status="completed", started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:01:00Z",
        outputs=[ArtifactRef(id=model_id, kind="model", path=str(model_dir), role="merged",
                             digest={"value": digest, "status": "available"})],
    )
    lineage_dir = output / "merge_runs" / run_id if nested_merge else output
    lineage_dir.mkdir(parents=True, exist_ok=True)
    (lineage_dir / "lineage.json").write_text(run.model_dump_json(), encoding="utf-8")
    return model_id


def _write_recipes(workspace):
    recipes = {
        "recipes/smoke/10-peft-cpt.yaml": "max_length: 1024\nmax_steps: 20\nlearning_rate: 0.0001\ngradient_accumulation_steps: 8\n",
        "recipes/mini/23-sft-1024-fallback.yaml": "max_length: 1024\nnum_train_epochs: 1\nlearning_rate: 0.0001\ngradient_accumulation_steps: 8\n",
        "recipes/mini/24-sft-registered-parent.yaml": "max_length: 1024\nnum_train_epochs: 1\nlearning_rate: 0.0001\ngradient_accumulation_steps: 8\n",
        "recipes/smoke/30-dpo.yaml": "max_length: 1024\nmax_steps: 20\nlearning_rate: 0.0001\ngradient_accumulation_steps: 8\n",
        "recipes/smoke/31-simpo-original.yaml": "max_length: 1024\nmax_steps: 20\nlearning_rate: 0.0001\ngradient_accumulation_steps: 8\n",
        "recipes/smoke/40-grpo-standard.yaml": "max_prompt_length_candidate: 512\nmax_completion_length_candidate: 512\nmax_steps: 20\nlearning_rate: 0.00001\ngradient_accumulation_steps: 4\n",
        "recipes/smoke/41-grpo-lowmem.yaml": "max_prompt_length_candidate: 512\nmax_completion_length_candidate: 512\nmax_steps: 20\nlearning_rate: 0.00001\ngradient_accumulation_steps: 4\n",
    }
    for relative, training in recipes.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"training:\n{''.join('  ' + line for line in training.splitlines(keepends=True))}", encoding="utf-8")
    reward_plugin = workspace / "plugins/grpo_rewards.py"
    reward_plugin.parent.mkdir(parents=True, exist_ok=True)
    reward_plugin.write_text("fixture", encoding="utf-8")


def _base_model(workspace, *, external_path=None):
    model_dir = external_path or workspace / "models" / "base-fixture"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}), encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"fixture weights")
    digest = sha256_directory(model_dir)
    model_id = f"model-{digest[:20]}"
    registry = workspace / "runs" / "base-models" / "model_artifacts" / f"{model_id}.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    record = ModelLineage(model_id=model_id, kind="base", path=str(model_dir), stage="base",
                          digest={"value": digest, "status": "available"})
    registry.write_text(record.model_dump_json(), encoding="utf-8")
    return model_id


def test_registered_sft_preflight_accepts_verified_base_model(tmp_path, monkeypatch):
    _write_recipes(tmp_path)
    dataset_id = _dataset(tmp_path)
    parent_id = _base_model(tmp_path)
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.console_command_available", lambda _: True)
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight._validate_sft_model_config", lambda _: None)

    result = preflight_stage(tmp_path, StagePreflightRequest(
        stage="sft", profile_id="sft-registered-v1", dataset_ids=[dataset_id], parent_model_id=parent_id,
    ), launcher_enabled=True)

    assert result["status"] == "ready_for_confirmation"
    assert result["parent_model"]["model_id"] == parent_id
    assert result["parent_model"]["stage"] == "base"
    assert result["parent_model"]["kind"] == "base"
    assert result["parent_model"]["digest"]
    assert result["training_parameters"]["config_sha256"]


def test_registered_sft_preflight_blocks_untrusted_external_base_path(tmp_path, monkeypatch):
    _write_recipes(tmp_path)
    dataset_id = _dataset(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-base"
    parent_id = _base_model(tmp_path, external_path=outside)
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.console_command_available", lambda _: True)
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight._validate_sft_model_config", lambda _: None)

    result = preflight_stage(tmp_path, StagePreflightRequest(
        stage="sft", profile_id="sft-registered-v1", dataset_ids=[dataset_id], parent_model_id=parent_id,
    ), launcher_enabled=True)

    assert result["status"] == "blocked"
    assert any("Base 模型权重不可用" in item for item in result["blockers"])


def test_registered_sft_preflight_accepts_verified_huggingface_cache_base(tmp_path, monkeypatch):
    _write_recipes(tmp_path)
    dataset_id = _dataset(tmp_path)
    cache = tmp_path / "trusted-hf-cache"
    monkeypatch.setenv("HF_HOME", str(cache))
    outside = cache / "hub" / "models--fixture--base" / "snapshots" / "revision"
    parent_id = _base_model(tmp_path, external_path=outside)
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.console_command_available", lambda _: True)
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight._validate_sft_model_config", lambda _: None)

    result = preflight_stage(tmp_path, StagePreflightRequest(
        stage="sft", profile_id="sft-registered-v1", dataset_ids=[dataset_id], parent_model_id=parent_id,
    ), launcher_enabled=True)

    assert result["status"] == "ready_for_confirmation"
    assert result["parent_model"]["model_id"] == parent_id


def test_registered_sft_start_uses_selected_parent_and_registered_dataset(tmp_path, monkeypatch):
    _write_recipes(tmp_path)
    dataset_id = _dataset(tmp_path, license_status="verified")
    parent_id = _base_model(tmp_path)
    parent_path = tmp_path / "models" / "base-fixture"
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.console_command_available", lambda _: True)
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight._validate_sft_model_config", lambda _: None)

    class Confirmations:
        def consume(self, _token, _request, _summary):
            return {"run_id": "run-base-sft-start", "output_location": "runs/sft/sft-registered-v1/run-base-sft-start"}

    class Launcher:
        captured = None

        def start_stage_cli(self, **kwargs):
            self.captured = kwargs
            return {"run_id": kwargs["run_id"], "status": "running"}

    launcher = Launcher()
    service = RegisteredStageTraining(tmp_path, launcher, Confirmations())
    request = RegisteredStageTrainingRequest(
        stage="sft", profile_id="sft-registered-v1", dataset_ids=[dataset_id], parent_model_id=parent_id,
    )

    result = service.start(request, "fixture-token", acknowledged_dataset_id=dataset_id)

    assert result["run_id"] == "run-base-sft-start"
    assert launcher.captured["parent_model_id"] == parent_id
    assert launcher.captured["training_parameters"]["parent_model_sha256"]
    argv = launcher.captured["argv"]
    assert "run-sft" in argv
    assert str(parent_path) in argv
    assert dataset_id not in argv
    assert (tmp_path / ".onegpu-posttrain" / "launcher" / "configs" / "run-base-sft-start.yaml").is_file()


def test_sft_preflight_verifies_registered_dataset_parent_and_reports_license_warning(tmp_path, monkeypatch):
    _write_recipes(tmp_path)
    dataset_id = _dataset(tmp_path, license_status="not_provided")
    parent_id = _parent_model(tmp_path, "cpt")
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.console_command_available", lambda _: True)

    result = preflight_stage(tmp_path, StagePreflightRequest(
        stage="sft", profile_id="sft-mini-1024-v1", dataset_ids=[dataset_id], parent_model_id=parent_id,
    ), launcher_enabled=True)

    assert result["status"] == "ready_for_confirmation"
    assert result["execution_started"] is False
    assert result["datasets"][0]["content_sha256"]
    assert result["parent_model"]["status"] == "valid"
    assert any("许可状态" in warning for warning in result["warnings"])


def test_preflight_resolves_bounded_user_parameters_and_rejects_invalid_overrides(tmp_path, monkeypatch):
    _write_recipes(tmp_path)
    dataset_id = _dataset(tmp_path)
    parent_id = _parent_model(tmp_path, "cpt")
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.console_command_available", lambda _: True)

    result = preflight_stage(tmp_path, StagePreflightRequest(
        stage="sft", profile_id="sft-mini-1024-v1", dataset_ids=[dataset_id],
        parent_model_id=parent_id, parameter_overrides={"max_length": 2048, "num_train_epochs": 2},
    ), launcher_enabled=True)
    assert result["training_parameters"]["defaults"]["max_length"] == 1024
    assert result["training_parameters"]["effective"]["max_length"] == 2048
    assert result["training_parameters"]["effective"]["num_train_epochs"] == 2
    assert len(result["training_parameters"]["config_sha256"]) == 64

    invalid = preflight_stage(tmp_path, StagePreflightRequest(
        stage="sft", profile_id="sft-mini-1024-v1", dataset_ids=[dataset_id],
        parent_model_id=parent_id, parameter_overrides={"max_length": 4048},
    ), launcher_enabled=True)
    assert invalid["status"] == "blocked"
    assert any("max_length" in blocker and "512 到 4047" in blocker for blocker in invalid["blockers"])


def test_sft_preflight_resolves_only_server_owned_memory_modes(tmp_path, monkeypatch):
    _write_recipes(tmp_path)
    dataset_id = _dataset(tmp_path)
    parent_id = _parent_model(tmp_path, "cpt")
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.console_command_available", lambda _: True)

    qlora = preflight_stage(tmp_path, StagePreflightRequest(
        stage="sft", profile_id="sft-mini-1024-v1", dataset_ids=[dataset_id], parent_model_id=parent_id,
        memory_mode_id="qlora-4bit-nf4",
    ), launcher_enabled=True)
    assert qlora["training_memory_mode"]["mode_id"] == "qlora-4bit-nf4"
    assert qlora["training_parameters"]["effective"]["quant_bits"] == 4

    invalid = preflight_stage(tmp_path, StagePreflightRequest(
        stage="sft", profile_id="sft-mini-1024-v1", dataset_ids=[dataset_id], parent_model_id=parent_id,
        memory_mode_id="qlora-custom",
    ), launcher_enabled=True)
    assert invalid["status"] == "blocked"
    assert any("显存模式" in blocker for blocker in invalid["blockers"])

    with pytest.raises(ValueError):
        StagePreflightRequest.model_validate({
            "stage": "sft", "profile_id": "sft-mini-1024-v1", "dataset_ids": [dataset_id],
            "parent_model_id": parent_id, "quant_bits": 8,
        })


def test_sft_preflight_blocks_missing_qlora_dependency_before_any_training_artifact(tmp_path, monkeypatch):
    _write_recipes(tmp_path)
    dataset_id = _dataset(tmp_path)
    parent_id = _parent_model(tmp_path, "cpt")
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.console_command_available", lambda _: True)
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.inspect_training_memory_mode",
                        lambda _: {"ok": False, "dependencies": [],
                                   "blockers": ["QLoRA needs bitsandbytes; not installed"]})

    before_run_records = set((tmp_path / "runs").rglob("run.json"))
    result = preflight_stage(tmp_path, StagePreflightRequest(
        stage="sft", profile_id="sft-mini-1024-v1", dataset_ids=[dataset_id], parent_model_id=parent_id,
        memory_mode_id="qlora-4bit-nf4",
    ), launcher_enabled=True)

    assert result["status"] == "blocked"
    assert any("bitsandbytes" in blocker for blocker in result["blockers"])
    assert result["execution_started"] is False
    assert set((tmp_path / "runs").rglob("run.json")) == before_run_records


def test_preflight_accepts_multiple_registered_stage_datasets_with_bounded_custom_parameters(tmp_path, monkeypatch):
    _write_recipes(tmp_path)
    first = _dataset(tmp_path, variant="-a")
    second = _dataset(tmp_path, variant="-b")
    parent_id = _parent_model(tmp_path, "cpt")
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.console_command_available", lambda _: True)

    result = preflight_stage(tmp_path, StagePreflightRequest(
        stage="sft", profile_id="sft-mini-1024-v1", dataset_ids=[first, second],
        parent_model_id=parent_id, parameter_overrides={"max_length": 768, "num_train_epochs": 2},
    ), launcher_enabled=True)

    assert result["status"] == "ready_for_confirmation"
    assert [item["dataset_id"] for item in result["datasets"]] == [first, second]
    assert len({item["content_sha256"] for item in result["datasets"]}) == 2
    assert result["training_parameters"]["effective"]["max_length"] == 768
    assert len(result["training_parameters"]["config_sha256"]) == 64
    assert result["execution_started"] is False


def test_sft_preflight_accepts_parent_produced_by_nested_merge_run(tmp_path, monkeypatch):
    _write_recipes(tmp_path)
    dataset_id = _dataset(tmp_path)
    parent_id = _parent_model(tmp_path, "cpt", nested_merge=True)
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.console_command_available", lambda _: True)

    result = preflight_stage(tmp_path, StagePreflightRequest(
        stage="sft", profile_id="sft-mini-1024-v1", dataset_ids=[dataset_id], parent_model_id=parent_id,
    ), launcher_enabled=True)

    assert result["status"] == "ready_for_confirmation"
    assert result["parent_model"]["status"] == "valid"


def test_swift_probe_uses_the_same_child_path_as_fixed_launcher(monkeypatch):
    observed = {}
    monkeypatch.setattr(subprocess_utils, "child_process_env", lambda: {"PATH": "python-scripts;parent-path"})

    def fake_which(command, path=None):
        observed.update(command=command, path=path)
        return "python-scripts/swift.exe"

    monkeypatch.setattr(subprocess_utils.shutil, "which", fake_which)
    assert subprocess_utils.console_command_available("swift") is True
    assert observed == {"command": "swift", "path": "python-scripts;parent-path"}


def test_preflight_blocks_stage_role_profile_and_tampered_content(tmp_path, monkeypatch):
    _write_recipes(tmp_path)
    holdout_id = _dataset(tmp_path, role="final_holdout")
    parent_id = _parent_model(tmp_path, "cpt")
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.console_command_available", lambda _: True)

    holdout = preflight_stage(tmp_path, StagePreflightRequest(
        stage="sft", profile_id="sft-mini-1024-v1", dataset_ids=[holdout_id], parent_model_id=parent_id,
    ), launcher_enabled=True)
    assert holdout["status"] == "blocked"
    assert any("角色" in blocker for blocker in holdout["blockers"])

    wrong_stage = preflight_stage(tmp_path, StagePreflightRequest(
        stage="dpo", profile_id="dpo-smoke-v1", dataset_ids=[holdout_id], parent_model_id=parent_id,
    ), launcher_enabled=True)
    assert wrong_stage["status"] == "blocked"
    assert any("阶段为 sft" in blocker for blocker in wrong_stage["blockers"])

    manifest = next((tmp_path / "data" / "processed").glob("**/manifests/*.yaml"))
    canonical = manifest.parent.parent / "canonical" / "sft.jsonl"
    canonical.write_text(canonical.read_text(encoding="utf-8") + " ", encoding="utf-8")
    tampered = preflight_stage(tmp_path, StagePreflightRequest(
        stage="sft", profile_id="sft-mini-1024-v1", dataset_ids=[holdout_id], parent_model_id=parent_id,
    ), launcher_enabled=True)
    assert tampered["status"] == "blocked"
    assert any("摘要" in blocker for blocker in tampered["blockers"])


def test_final_holdout_cannot_be_registered_as_training_time_validation(tmp_path):
    rows = [{"id": f"holdout-{index}", "raw_problem": f"question {index}", "messages": [
        {"role": "user", "content": f"question {index}"}, {"role": "assistant", "content": "answer"}],
        "prompt_template_id": "math-v1", "metadata": {"data_role": "final_holdout"}}
        for index in range(2)]
    encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    registered = register_jsonl(
        io.BytesIO(encoded), workspace=tmp_path,
        output_root=tmp_path / "data" / "processed" / "final-holdout-validation-fixture",
        stage="sft", origin=DataSource(source_id="fixture", kind="local_upload", name="holdout",
                                       license_status="not_provided", declared_by="user"),
    )
    with pytest.raises(ValueError, match="final_holdout"):
        inspect_registered_validation_dataset(tmp_path, registered.dataset_id, "sft", ["validation", "diagnostic_dev"])


def test_preflight_api_is_read_only_and_blocks_unsupported_dpo(tmp_path):
    _write_recipes(tmp_path)
    dataset_id = _dataset(tmp_path, "preference")
    parent_id = _parent_model(tmp_path, "sft")
    client = TestClient(create_app(tmp_path, enable_experiment_launcher=True))

    response = client.post("/api/training/preflight", json={
        "stage": "dpo", "profile_id": "dpo-smoke-v1", "dataset_ids": [dataset_id],
        "parent_model_id": parent_id,
    })

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "blocked"
    assert payload["execution_started"] is False
    assert any("adapter" in blocker for blocker in payload["blockers"])

    rejected = client.post("/api/training/preflight", json={
        "stage": "dpo", "profile_id": "dpo-smoke-v1", "dataset_ids": [dataset_id],
        "parent_model_id": parent_id, "recipe_path": "arbitrary.yaml",
    })
    assert rejected.status_code == 422


@pytest.mark.parametrize(("profile_id", "stage"), [
    ("dpo-smoke-v1", "dpo"), ("simpo-smoke-v1", "simpo"),
])
def test_preference_profiles_validate_pairs_and_surface_ordering_evidence(tmp_path, monkeypatch, profile_id, stage):
    _write_recipes(tmp_path)
    dataset_id = _dataset(tmp_path, "preference")
    parent_id = _parent_model(tmp_path, "sft")
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.console_command_available", lambda _: True)
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight._gpu_snapshot", lambda: {"available": False})

    result = preflight_stage(tmp_path, StagePreflightRequest(
        stage=stage, profile_id=profile_id, dataset_ids=[dataset_id], parent_model_id=parent_id,
    ), launcher_enabled=True)

    assert result["algorithm"] == stage
    assert result["datasets"][0]["ordering_evidence_status"] == "unverified"
    assert result["datasets"][0]["ordering_evidence_count"] == 0
    assert any("chosen/rejected 非空且不相同" in check["detail"]
               for check in result["checks"] if check["check_id"].endswith(":preference_pairs"))
    assert any("排序证据" in warning for warning in result["warnings"])
    # Evidence quality is shown as a warning; the absent generic web adapter remains the blocker.
    assert any("adapter" in blocker for blocker in result["blockers"])


def test_preference_ordering_evidence_is_reported_when_present(tmp_path, monkeypatch):
    _write_recipes(tmp_path)
    dataset_id = _dataset(tmp_path, "preference", preference_evidence=True)
    parent_id = _parent_model(tmp_path, "sft")
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.console_command_available", lambda _: True)

    result = preflight_stage(tmp_path, StagePreflightRequest(
        stage="dpo", profile_id="dpo-smoke-v1", dataset_ids=[dataset_id], parent_model_id=parent_id,
    ), launcher_enabled=True)

    assert result["datasets"][0]["ordering_evidence_status"] == "verified"
    assert result["datasets"][0]["ordering_evidence_count"] == 1
    assert not any("排序证据" in warning for warning in result["warnings"])


def test_preference_optimization_rejects_non_sft_parent(tmp_path, monkeypatch):
    _write_recipes(tmp_path)
    dataset_id = _dataset(tmp_path, "preference")
    cpt_parent = _parent_model(tmp_path, "cpt")
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.console_command_available", lambda _: True)

    result = preflight_stage(tmp_path, StagePreflightRequest(
        stage="simpo", profile_id="simpo-smoke-v1", dataset_ids=[dataset_id], parent_model_id=cpt_parent,
    ), launcher_enabled=True)

    assert any("父模型阶段为 cpt" in blocker for blocker in result["blockers"])


def test_confirmation_api_returns_bound_token_without_starting_run(tmp_path, monkeypatch):
    _write_recipes(tmp_path)
    dataset_id = _dataset(tmp_path)
    parent_id = _parent_model(tmp_path, "cpt")
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.console_command_available", lambda _: True)
    client = TestClient(create_app(tmp_path, enable_experiment_launcher=True))

    response = client.post("/api/training/preflight/confirm", json={
        "stage": "sft", "profile_id": "sft-mini-1024-v1", "dataset_ids": [dataset_id],
        "parent_model_id": parent_id,
    })

    assert response.status_code == 200
    payload = response.json()
    assert payload["confirmation_token"]
    assert payload["expires_in_seconds"] == 300
    assert payload["bound_config_sha256"]
    assert payload["output_location"].startswith("runs/sft/sft-mini-1024-v1/run-")
    assert payload["execution_started"] is False
    assert not (tmp_path / payload["output_location"]).exists()


def _grpo_contract_dataset(workspace, *, contract_metadata=True, corrupt_protocol=False):
    contract = ExecutableRewardContract.create()
    metadata = {
        "problem_id": "grpo-problem-1", "data_role": "train_seed",
        "reference_source": "registered_train_seed",
        "reward_contract_id": contract.contract_id,
        "reward_contract_protocol_sha256": "0" * 64 if corrupt_protocol else contract.protocol_sha256,
        "reward_contract_implementation_sha256": contract.implementation_sha256,
    }
    row = {"id": "grpo-contract-1", "raw_problem": "1+1?", "prompt": [
        {"role": "user", "content": "1+1?"}], "answer": "2",
        "prompt_template_id": "qwen3-math-v1", "metadata": metadata}
    registration_metadata = ({
        "reward_contract_verified": True,
        "reward_contract_id": contract.contract_id,
        "reward_contract_protocol_sha256": contract.protocol_sha256,
        "reward_contract_implementation_sha256": contract.implementation_sha256,
    } if contract_metadata else {})
    result = register_jsonl(
        io.BytesIO((json.dumps(row) + "\n").encode()), workspace=workspace,
        output_root=workspace / "data/processed/grpo-contract-fixture", stage="grpo",
        origin=DataSource(source_id="grpo-contract-source", kind="local_upload", name="grpo fixture",
                          license_status="not_provided", declared_by="user"),
        registration_metadata=registration_metadata,
    )
    return result.dataset_id


@pytest.mark.parametrize(("contract_metadata", "corrupt_protocol"), [(False, False), (True, True)])
def test_grpo_preflight_blocks_missing_or_mismatched_reward_contract(
    tmp_path, contract_metadata, corrupt_protocol,
):
    _write_recipes(tmp_path)
    (tmp_path / "plugins/grpo_rewards.py").write_text(
        'orms["onegpu_accuracy"] = OneGPUAccuracyReward\n'
        'orms["onegpu_boxed_format"] = OneGPUBoxedFormatReward\n', encoding="utf-8")
    dataset_id = _grpo_contract_dataset(tmp_path, contract_metadata=contract_metadata,
                                      corrupt_protocol=corrupt_protocol)
    parent_id = _parent_model(tmp_path, "sft")
    result = preflight_stage(tmp_path, StagePreflightRequest(
        stage="grpo", profile_id="grpo-standard-smoke-v1", dataset_ids=[dataset_id],
        parent_model_id=parent_id,
    ), launcher_enabled=True, allow_cli_adapter=True)
    assert result["status"] == "blocked"
    assert any("GRPO" in blocker for blocker in result["blockers"])


def test_grpo_preflight_accepts_matching_executable_contract_and_sft_parent(tmp_path):
    _write_recipes(tmp_path)
    plugin = tmp_path / "plugins/grpo_rewards.py"
    plugin.write_text(
        'orms["onegpu_accuracy"] = OneGPUAccuracyReward\n'
        'orms["onegpu_boxed_format"] = OneGPUBoxedFormatReward\n', encoding="utf-8")
    dataset_id = _grpo_contract_dataset(tmp_path)
    parent_id = _parent_model(tmp_path, "sft")
    result = preflight_stage(tmp_path, StagePreflightRequest(
        stage="grpo", profile_id="grpo-standard-smoke-v1", dataset_ids=[dataset_id],
        parent_model_id=parent_id,
    ), launcher_enabled=True, allow_cli_adapter=True)
    contract = result["datasets"][0]["reward_contract"]
    assert contract["status"] == "verified"
    assert contract["reward_names"] == ["accuracy", "boxed_format"]
    assert contract["plugin_sha256"]
    assert result["parent_model"]["stage"] == "sft"
    assert not any("reward/reference 契约" in blocker for blocker in result["blockers"])


def test_grpo_preflight_rejects_non_sft_parent_even_with_valid_contract(tmp_path):
    _write_recipes(tmp_path)
    (tmp_path / "plugins/grpo_rewards.py").write_text(
        'orms["onegpu_accuracy"] = OneGPUAccuracyReward\n'
        'orms["onegpu_boxed_format"] = OneGPUBoxedFormatReward\n', encoding="utf-8")
    dataset_id = _grpo_contract_dataset(tmp_path)
    cpt_parent = _parent_model(tmp_path, "cpt")
    result = preflight_stage(tmp_path, StagePreflightRequest(
        stage="grpo", profile_id="grpo-standard-smoke-v1", dataset_ids=[dataset_id],
        parent_model_id=cpt_parent,
    ), launcher_enabled=True, allow_cli_adapter=True)
    assert any("父模型阶段为 cpt" in blocker for blocker in result["blockers"])


def test_grpo_cli_adapter_keeps_standard_and_lowmem_profiles_separate(tmp_path, monkeypatch):
    _write_recipes(tmp_path)
    (tmp_path / "plugins/grpo_rewards.py").write_text(
        'orms["onegpu_accuracy"] = OneGPUAccuracyReward\n'
        'orms["onegpu_boxed_format"] = OneGPUBoxedFormatReward\n', encoding="utf-8")
    dataset_id = _grpo_contract_dataset(tmp_path)
    parent_id = _parent_model(tmp_path, "sft")
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.console_command_available", lambda _: True)
    adapter = GRPOStageAdapter(tmp_path)
    standard = adapter.build_command(GRPOStageRequest(
        profile_id="grpo-standard-smoke-v1", dataset_id=dataset_id, parent_model_id=parent_id,
    ), run_id="run-grpo-standard-adapter")
    lowmem = adapter.build_command(GRPOStageRequest(
        profile_id="grpo-lowmem-smoke-v1", dataset_id=dataset_id, parent_model_id=parent_id,
    ), run_id="run-grpo-lowmem-adapter")
    assert standard != lowmem
    assert standard[standard.index("--profile-id") + 1] == "grpo-standard-smoke-v1"
    assert lowmem[lowmem.index("--profile-id") + 1] == "grpo-lowmem-smoke-v1"


def test_registered_preference_stage_web_flow_confirms_launches_and_persists_run_lineage(
    tmp_path, monkeypatch,
):
    _write_recipes(tmp_path)
    dataset_id = _dataset(tmp_path, "preference", license_status="not_provided")
    parent_id = _parent_model(tmp_path, "sft")
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.console_command_available", lambda _: True)
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight._gpu_snapshot", lambda: {"available": False})

    class FinishedProcess:
        def __init__(self):
            self.stdout = io.StringIO("global_step: 1/1\n")

        def poll(self):
            return 0

        def wait(self):
            return 0

    def fake_popen(argv, *, cwd, **kwargs):
        output = Path(cwd) / argv[argv.index("--output") + 1]
        output.mkdir(parents=True)
        run_id = argv[argv.index("--run-id") + 1]
        (output / "run.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
        (output / "lineage.json").write_text(json.dumps({
            "run_id": run_id, "status": "completed", "lineage_warnings": [],
        }), encoding="utf-8")
        return FinishedProcess()

    monkeypatch.setattr(launcher_module.subprocess, "Popen", fake_popen)
    app = create_app(tmp_path, enable_experiment_launcher=True)
    payload = {"stage": "dpo", "profile_id": "dpo-smoke-v1", "dataset_ids": [dataset_id],
               "parent_model_id": parent_id, "parameter_overrides": {"max_steps": 3}}
    with TestClient(app) as client:
        token = app.state.experiment_launcher_token
        headers = {"X-Local-Launcher-Token": token}

        preflight = client.post("/api/training/stage/preflight", json=payload, headers=headers)
        assert preflight.status_code == 200, preflight.text
        assert preflight.json()["status"] == "ready_for_confirmation"
        assert preflight.json()["training_parameters"]["effective"]["max_steps"] == 3
        confirmation = client.post("/api/training/stage/confirm", json=payload, headers=headers)
        assert confirmation.status_code == 200, confirmation.text
        confirmed = confirmation.json()
        assert confirmed["run_id"].startswith("run-")
        assert confirmed["output_location"].endswith(confirmed["run_id"])
        assert not (tmp_path / confirmed["output_location"]).exists()

        started = client.post("/api/training/stage/start", json={
            **payload, "confirmation_token": confirmed["confirmation_token"],
            "acknowledged_dataset_id": dataset_id,
        }, headers=headers)
        assert started.status_code == 200, started.text
        run_id = started.json()["run_id"]
        deadline = time.monotonic() + 3
        observed = None
        while time.monotonic() < deadline:
            observed = client.get(f"/api/launcher/runs/{run_id}").json()
            if observed["status"] not in {"starting", "running"}:
                break
            time.sleep(0.01)
        assert observed["status"] == "completed"
        assert observed["progress_status"] == "measured"
        lineage = client.get(f"/api/launcher/runs/{run_id}/lineage")
        assert lineage.status_code == 200
        assert lineage.json()["run_id"] == run_id
        assert any(dataset_id in warning for warning in lineage.json()["lineage_warnings"])
        resolved_recipe = tmp_path / ".onegpu-posttrain/launcher/configs" / f"{run_id}.yaml"
        assert __import__("yaml").safe_load(resolved_recipe.read_text(encoding="utf-8"))["training"]["max_steps"] == 3


def test_registered_grpo_adapter_runs_fixed_profile_and_preserves_lineage(tmp_path, monkeypatch):
    _write_recipes(tmp_path)
    recipe = tmp_path / "recipes/smoke/40-grpo-standard.yaml"
    recipe.write_text("profile: standard\nseed: 42\ntraining:\n  max_steps: 1\n", encoding="utf-8")
    (tmp_path / "plugins/grpo_rewards.py").write_text(
        'orms["onegpu_accuracy"] = OneGPUAccuracyReward\n'
        'orms["onegpu_boxed_format"] = OneGPUBoxedFormatReward\n', encoding="utf-8")
    dataset_id = _grpo_contract_dataset(tmp_path)
    parent_id = _parent_model(tmp_path, "sft")
    monkeypatch.setattr(cli, "console_command_available", lambda _: True)
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.console_command_available", lambda _: True)

    def fake_train(command, *, log_dir, dry_run=False, allow_failure=False):
        assert allow_failure is True and dry_run is False
        logging_dir = Path(command[command.index("--logging_dir") + 1])
        assert logging_dir.is_dir()
        assert logging_dir.parent == tmp_path / args.output
        train_dir = Path(command[command.index("--output_dir") + 1])
        checkpoint = train_dir / "checkpoint-1"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter.bin").write_bytes(b"fake-grpo-adapter")
        return {"status": "PASS", "returncode": 0, "command": command}

    def fake_merge(command, *, log_dir=None):
        merged = Path(command[command.index("--output_dir") + 1])
        merged.mkdir(parents=True)
        (merged / "weights.bin").write_bytes(b"fake-grpo-merged")
        return {"status": "PASS", "returncode": 0}

    monkeypatch.setattr(cli, "run_monitored_command", fake_train)
    monkeypatch.setattr(cli, "run_command", fake_merge)
    args = Namespace(workspace=str(tmp_path), profile_id="grpo-standard-smoke-v1",
                     dataset_id=dataset_id, parent_model_id=parent_id,
                     run_id="run-grpo-standard-adapter-test",
                     output="runs/grpo/grpo-standard-smoke-v1/run-grpo-standard-adapter-test")
    assert cli.cmd_run_registered_grpo(args) == 0
    output = tmp_path / args.output
    run = json.loads((output / "lineage.json").read_text(encoding="utf-8"))
    result = json.loads((output / "run.json").read_text(encoding="utf-8"))
    resolved = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "grpo_metrics.json").read_text(encoding="utf-8"))
    assert run["run_id"] == args.run_id and run["status"] == "completed"
    assert [item["id"] for item in run["inputs"]] == [parent_id, dataset_id]
    assert result["reward_contract"]["status"] == "verified"
    assert resolved["reward_contract"]["contract_id"] == result["reward_contract"]["contract_id"]
    assert metrics["status"] == "not_measured"
    assert result["merged_model"] and Path(result["merged_model"]).is_dir()

    merged_record = next(json.loads(path.read_text(encoding="utf-8"))
                         for path in (output / "model_artifacts").glob("*.json")
                         if json.loads(path.read_text(encoding="utf-8")).get("kind") == "merged")
    monkeypatch.setattr(cli, "run_evalscope", lambda command, dry_run=False, stream_output=False:
                        {"returncode": 0, "stream_output": stream_output})
    monkeypatch.setattr(cli, "collect_evalscope_reports", lambda _: {
        "gsm8k": {"accuracy": 0.4, "score": 0.4, "samples": 5},
        "math_500": {"accuracy": 0.6, "score": 0.6, "samples": 5},
    })
    eval_config = tmp_path / "configs/evaluation/math-v1-launcher-mini.yaml"
    eval_config.parent.mkdir(parents=True, exist_ok=True)
    eval_config.write_text(
        "protocol_version: math-eval-v1\nprompt_version: qwen3-math-eval-v1\n"
        "benchmarks: [gsm8k, math_500]\ngeneration:\n  do_sample: false\n  temperature: 0.0\n  max_tokens: 512\n",
        encoding="utf-8")
    eval_recipe = tmp_path / "recipes/mini/30-eval-math-v1.yaml"
    eval_recipe.parent.mkdir(parents=True, exist_ok=True)
    eval_recipe.write_text(
        "run_label: grpo-eval-test\nstage: grpo_eval\n"
        "model_config: configs/model/qwen3-0.6b-base.yaml\n"
        "evaluation_config: configs/evaluation/math-v1-launcher-mini.yaml\nseed: 42\n",
        encoding="utf-8")
    eval_args = Namespace(workspace=str(tmp_path), model_id=merged_record["model_id"],
                          run_id="run-grpo-eval-adapter-test",
                          output="runs/evaluations/run-grpo-eval-adapter-test")
    assert cli.cmd_run_registered_grpo_eval(eval_args) == 0
    eval_output = tmp_path / eval_args.output
    eval_report = json.loads((eval_output / "eval.json").read_text(encoding="utf-8"))
    eval_lineage = json.loads((eval_output / "evaluation_lineage.json").read_text(encoding="utf-8"))
    eval_run = json.loads((eval_output / "evaluation_run.json").read_text(encoding="utf-8"))
    assert eval_report["resolved_evaluation"]["profile_id"] == "grpo-math-eval-v1"
    assert eval_report["metric_statuses"]["gsm8k.accuracy"]["denominator"] == 5
    assert eval_report["grpo_training_evidence"]["training_run_id"] == args.run_id
    assert eval_report["grpo_training_evidence"]["metrics"]["grpo_training.reward"]["status"] == "not_measured"
    assert eval_lineage["model_produced_by_run_id"] == result["merge"]["run_id"]
    assert eval_lineage["metrics"]["grpo_training.reward"]["status"] == "not_measured"
    assert any(item["role"] == "grpo_training_diagnostics" for item in eval_run["inputs"])


def test_registered_grpo_oom_preserves_failure_and_does_not_invent_rollout_metrics(tmp_path, monkeypatch):
    _write_recipes(tmp_path)
    (tmp_path / "recipes/smoke/40-grpo-standard.yaml").write_text(
        "profile: standard\nseed: 42\ntraining:\n  max_steps: 1\n", encoding="utf-8")
    (tmp_path / "plugins/grpo_rewards.py").write_text(
        'orms["onegpu_accuracy"] = OneGPUAccuracyReward\n'
        'orms["onegpu_boxed_format"] = OneGPUBoxedFormatReward\n', encoding="utf-8")
    dataset_id = _grpo_contract_dataset(tmp_path)
    parent_id = _parent_model(tmp_path, "sft")
    monkeypatch.setattr(cli.shutil, "which", lambda _: "swift")
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.console_command_available", lambda _: True)
    monkeypatch.setattr(cli, "run_monitored_command", lambda *args, **kwargs: {"status": "OOM", "returncode": 1})
    monkeypatch.setattr(cli, "run_command", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("merge must not run after OOM")))
    args = Namespace(workspace=str(tmp_path), profile_id="grpo-standard-smoke-v1",
                     dataset_id=dataset_id, parent_model_id=parent_id,
                     run_id="run-grpo-oom-adapter-test",
                     output="runs/grpo/grpo-standard-smoke-v1/run-grpo-oom-adapter-test")
    assert cli.cmd_run_registered_grpo(args) == 4
    output = tmp_path / args.output
    run = json.loads((output / "lineage.json").read_text(encoding="utf-8"))
    result = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "oom" and run["outputs"] == []
    assert result["grpo_metrics"]["status"] == "not_measured"
    assert result["grpo_metrics"]["metric_statuses"]["reward"]["status"] == "not_measured"


def test_registered_grpo_pass_without_checkpoint_preserves_failed_run_without_merging(tmp_path, monkeypatch):
    _write_recipes(tmp_path)
    (tmp_path / "recipes/smoke/40-grpo-standard.yaml").write_text(
        "profile: standard\nseed: 42\ntraining:\n  max_steps: 1\n", encoding="utf-8")
    (tmp_path / "plugins/grpo_rewards.py").write_text(
        'orms["onegpu_accuracy"] = OneGPUAccuracyReward\n'
        'orms["onegpu_boxed_format"] = OneGPUBoxedFormatReward\n', encoding="utf-8")
    dataset_id = _grpo_contract_dataset(tmp_path)
    parent_id = _parent_model(tmp_path, "sft")
    monkeypatch.setattr(cli.shutil, "which", lambda _: "swift")
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.console_command_available", lambda _: True)
    monkeypatch.setattr(cli, "run_monitored_command", lambda *args, **kwargs:
                        {"status": "PASS", "returncode": 0, "command": args[0]})
    monkeypatch.setattr(cli, "run_command", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("merge must not run without a checkpoint")))
    args = Namespace(workspace=str(tmp_path), profile_id="grpo-standard-smoke-v1",
                     dataset_id=dataset_id, parent_model_id=parent_id,
                     run_id="run-grpo-no-checkpoint-test",
                     output="runs/grpo/grpo-standard-smoke-v1/run-grpo-no-checkpoint-test")
    assert cli.cmd_run_registered_grpo(args) == 4
    output = tmp_path / args.output
    run = json.loads((output / "lineage.json").read_text(encoding="utf-8"))
    result = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "failed" and run["outputs"] == []
    assert result["checkpoint"] is None and result["merged_model"] is None
    assert result["grpo_metrics"]["status"] == "not_measured"
    assert "没有可用 checkpoint" in result["train"]["detail"]
    assert result["merged_model"] is None
