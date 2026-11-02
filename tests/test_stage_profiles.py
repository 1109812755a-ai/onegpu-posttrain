import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from onegpu_posttrain.lineage import Digest, ModelLineage, sha256_file
from onegpu_posttrain.ui.server import create_app
from onegpu_posttrain.ui.stage_profiles import (
    StageProfile,
    resolve_training_memory_mode,
    resolve_training_parameters,
    stage_profile_catalog,
)


def _workspace_with_recipes(root):
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
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"training:\n{''.join('  ' + line for line in training.splitlines(keepends=True))}", encoding="utf-8")
    for relative in (
        "plugins/grpo_rewards.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")


def test_registry_describes_each_algorithm_and_its_stage_contracts(tmp_path):
    _workspace_with_recipes(tmp_path)
    catalog = stage_profile_catalog(tmp_path)
    profiles = {item.profile_id: item for item in catalog.profiles}

    assert set(profiles) == {
        "cpt-smoke-v1", "sft-mini-1024-v1", "dpo-smoke-v1", "simpo-smoke-v1",
        "sft-registered-v1", "grpo-standard-smoke-v1", "grpo-lowmem-smoke-v1",
    }
    assert profiles["cpt-smoke-v1"].parent_model_stages == ["base"]
    cpt_eval = profiles["cpt-smoke-v1"].evaluation_profiles[0]
    assert cpt_eval.protocol == "math-eval-v1"
    assert "cpt_holdout" in cpt_eval.dataset_roles
    assert cpt_eval.denominator_semantics["domain_ppl.perplexity"].startswith("模型实际预测 token 数")
    assert cpt_eval.metric_sources["domain_ppl.perplexity"].startswith("canonical CPT domain holdout")
    assert profiles["sft-mini-1024-v1"].dataset_stage == "sft"
    assert profiles["sft-registered-v1"].parent_model_stages == ["base", "cpt"]
    assert profiles["sft-registered-v1"].parent_model_rule == "registered_sft_parent"
    sft_eval = profiles["sft-mini-1024-v1"].evaluation_profiles[0]
    assert sft_eval.protocol == "math-eval-v1"
    assert sft_eval.denominator_semantics["gsm8k.accuracy"].startswith("EvalScope 实际完成样本数")
    assert profiles["dpo-smoke-v1"].dataset_roles == ["train_pairs"]
    assert profiles["simpo-smoke-v1"].algorithm == "simpo"
    assert profiles["dpo-smoke-v1"].approved_recipe != profiles["simpo-smoke-v1"].approved_recipe
    assert profiles["dpo-smoke-v1"].parent_model_stages == profiles["simpo-smoke-v1"].parent_model_stages == ["sft"]
    assert profiles["dpo-smoke-v1"].parent_model_rule == profiles["simpo-smoke-v1"].parent_model_rule == "registered_sft_merged"
    assert profiles["grpo-standard-smoke-v1"].parent_model_stages == ["sft"]
    assert profiles["grpo-standard-smoke-v1"].reward_contract == "plugins/grpo_rewards.py"
    grpo_eval = profiles["grpo-standard-smoke-v1"].evaluation_profiles[0]
    assert grpo_eval.profile_id == "grpo-math-eval-v1"
    assert grpo_eval.protocol == "math-eval-v1"
    assert "grpo_training.reward" in grpo_eval.metrics
    assert "trainer log points" in grpo_eval.denominator_semantics["grpo_training.reward"]
    assert "final_holdout" in profiles["sft-mini-1024-v1"].evaluation_profiles[0].dataset_roles
    assert all(profile.output_verification == "model_manifest_and_digest" for profile in catalog.profiles)
    assert all(any(item.profile_id == "choice-accuracy-v1" and item.dataset_roles == ["evaluation_only"]
                   for item in profile.evaluation_profiles) for profile in catalog.profiles)
    assert profiles["sft-mini-1024-v1"].execution_supported is True
    assert profiles["dpo-smoke-v1"].execution_supported is True
    sft_parameters = {item.key: item for item in profiles["sft-mini-1024-v1"].training_parameters}
    assert sft_parameters["max_length"].default == 1024
    assert sft_parameters["max_length"].choices == []
    assert (sft_parameters["max_length"].minimum, sft_parameters["max_length"].maximum) == (512, 4047)
    assert "max_prompt_length_candidate" in {
        item.key for item in profiles["grpo-standard-smoke-v1"].training_parameters
    }
    memory_modes = profiles["sft-mini-1024-v1"].training_memory_modes
    assert [item.mode_id for item in memory_modes] == ["lora", "qlora-4bit-nf4"]
    assert resolve_training_memory_mode(profiles["sft-mini-1024-v1"]).mode_id == "lora"
    assert resolve_training_memory_mode(profiles["sft-mini-1024-v1"], "qlora-4bit-nf4").settings["quant_bits"] == 4
    with pytest.raises(ValueError, match="qlora"):
        resolve_training_memory_mode(profiles["sft-mini-1024-v1"], "qlora-arbitrary")
    with pytest.raises(ValueError, match="不支持"):
        resolve_training_memory_mode(profiles["dpo-smoke-v1"], "qlora-4bit-nf4")


def test_stage_parameter_overrides_are_profile_scoped_and_bounded(tmp_path):
    _workspace_with_recipes(tmp_path)
    profiles = {item.profile_id: item for item in stage_profile_catalog(tmp_path).profiles}
    profile = profiles["sft-mini-1024-v1"]

    resolved = resolve_training_parameters(profile, {"max_length": 2048, "num_train_epochs": 2})
    assert resolved["max_length"] == 2048
    upper_edge = resolve_training_parameters(profile, {"max_length": 4047})
    assert upper_edge["max_length"] == 4047
    assert resolved["num_train_epochs"] == 2
    assert resolved["learning_rate"] == 0.0001

    with pytest.raises(ValueError, match="不支持的训练参数"):
        resolve_training_parameters(profile, {"per_device_train_batch_size": 2})
    with pytest.raises(ValueError, match="512 到 4047"):
        resolve_training_parameters(profile, {"max_length": 4048})
    with pytest.raises(ValueError, match="整数"):
        resolve_training_parameters(profile, {"max_length": 768.0})


def test_stage_profile_api_returns_typed_allowlisted_profiles(tmp_path):
    _workspace_with_recipes(tmp_path)
    client = TestClient(create_app(tmp_path))

    response = client.get("/api/training/stage-profiles")

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "1.0"
    assert {item["algorithm"] for item in payload["profiles"]} == {
        "cpt", "sft", "dpo", "simpo", "grpo",
    }
    sft = next(item for item in payload["profiles"] if item["stage"] == "sft")
    assert [item["mode_id"] for item in sft["training_memory_modes"]] == ["lora", "qlora-4bit-nf4"]
    assert sft["training_memory_modes"][0]["default"] is True


def test_stage_workbench_options_api_is_read_only_and_returns_bounded_catalog_shape(tmp_path):
    client = TestClient(create_app(tmp_path))
    response = client.get("/api/training/workbench-options")
    assert response.status_code == 200
    assert response.json() == {"datasets": [], "models": []}


def test_stage_workbench_model_options_expose_verified_handoff_identity(tmp_path):
    weights = tmp_path / "runs" / "train-one" / "merged" / "model.bin"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"model-weights")
    record = ModelLineage(model_id="model-sft-one", kind="merged", stage="sft",
        path=weights.relative_to(tmp_path).as_posix(), produced_by_run_id="run-train-one",
        digest=Digest(value=sha256_file(weights), status="available"))
    registry = tmp_path / "runs" / "train-one" / "model_artifacts" / "model-sft-one.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(record.model_dump_json(), encoding="utf-8")
    (tmp_path / "runs" / "train-one" / "lineage.json").write_text(
        '{"run_id":"run-train-one","run_type":"train","status":"completed"}', encoding="utf-8")

    response = TestClient(create_app(tmp_path)).get("/api/training/workbench-options")

    assert response.status_code == 200
    model = response.json()["models"][0]
    assert model["model_id"] == "model-sft-one"
    assert model["produced_by_run_id"] == "run-train-one"
    assert model["source_run_id"] == "run-train-one"
    assert model["digest"]["value"] == sha256_file(weights)
    assert model["path_available"] is True


def test_stage_workbench_resolves_merged_model_to_exact_training_parent(tmp_path):
    runs = tmp_path / "runs"
    parent_path = runs / "train-one" / "adapter" / "weights.bin"
    merged_path = runs / "merge-one" / "merged" / "weights.bin"
    parent_path.parent.mkdir(parents=True)
    merged_path.parent.mkdir(parents=True)
    parent_path.write_bytes(b"adapter")
    merged_path.write_bytes(b"merged")
    parent_digest = sha256_file(parent_path)
    merged_digest = sha256_file(merged_path)
    (runs / "train-one" / "lineage.json").write_text(
        '{"run_id":"run-train-one","run_type":"train","status":"completed"}', encoding="utf-8")
    (runs / "merge-one" / "lineage.json").write_text(
        '{"run_id":"run-merge-one","run_type":"merge","status":"completed",'
        '"inputs":[{"id":"model-adapter-one","kind":"model","digest":{"status":"available","value":"'
        + parent_digest + '"}}]}', encoding="utf-8")
    parent = ModelLineage(model_id="model-adapter-one", kind="adapter", stage="sft",
        path=parent_path.relative_to(tmp_path).as_posix(), produced_by_run_id="run-train-one",
        digest=Digest(value=parent_digest, status="available"))
    merged = ModelLineage(model_id="model-merged-one", kind="merged", stage="sft",
        path=merged_path.relative_to(tmp_path).as_posix(), produced_by_run_id="run-merge-one",
        parent_model_ids=["model-adapter-one"], digest=Digest(value=merged_digest, status="available"))
    for model in (parent, merged):
        entry = runs / model.produced_by_run_id.removeprefix("run-") / "model_artifacts" / f"{model.model_id}.json"
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text(model.model_dump_json(), encoding="utf-8")

    response = TestClient(create_app(tmp_path)).get("/api/training/workbench-options")

    assert response.status_code == 200
    models = {item["model_id"]: item for item in response.json()["models"]}
    assert models["model-adapter-one"]["source_run_id"] == "run-train-one"
    assert models["model-merged-one"]["source_run_id"] == "run-train-one"


def test_missing_recipe_and_reward_contract_become_blocked_not_hidden(tmp_path):
    catalog = stage_profile_catalog(tmp_path)
    profiles = {item.profile_id: item for item in catalog.profiles}
    assert profiles["sft-mini-1024-v1"].execution_supported is False
    assert "recipe" in profiles["sft-mini-1024-v1"].execution_blocker
    assert profiles["grpo-standard-smoke-v1"].execution_supported is False
    assert "reward contract" in profiles["grpo-standard-smoke-v1"].execution_blocker


def test_profile_dto_rejects_unknown_fields_and_stage_names():
    profile = stage_profile_catalog(".").profiles[0].model_dump(mode="json")
    invalid_stage = {**profile, "stage": "mystery"}
    with pytest.raises(ValidationError):
        StageProfile.model_validate(invalid_stage)
    extra = {**profile, "unexpected_command": "shell"}
    with pytest.raises(ValidationError):
        StageProfile.model_validate(extra)
