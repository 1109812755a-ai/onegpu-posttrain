import pytest
import yaml

from onegpu_posttrain.ui.resolved_stage_recipe import (
    materialize_stage_recipe,
    verify_materialized_stage_recipe,
)
from onegpu_posttrain.ui.stage_profiles import stage_profile_catalog
from onegpu_posttrain.lineage import ModelLineage, sha256_directory


def _workspace(root):
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
        path.write_text("training:\n" + "".join(f"  {line}\n" for line in training.splitlines()), encoding="utf-8")
    (root / "plugins").mkdir()
    (root / "plugins/grpo_rewards.py").write_text("fixture", encoding="utf-8")


def _registered_base(root):
    model_dir = root / "models" / "base"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model_type":"qwen2","architectures":["Qwen2ForCausalLM"]}', encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"fixture")
    digest = sha256_directory(model_dir)
    model_id = f"model-{digest[:20]}"
    registry = root / "runs" / "base-models" / "model_artifacts" / f"{model_id}.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(ModelLineage(model_id=model_id, kind="base", path=str(model_dir), stage="base",
                                    digest={"value": digest, "status": "available"}).model_dump_json(), encoding="utf-8")
    return model_id, model_dir


@pytest.mark.parametrize(("profile_id", "override", "expected"), [
    ("cpt-smoke-v1", {"max_steps": 4}, 4),
    ("dpo-smoke-v1", {"max_steps": 4}, 4),
    ("simpo-smoke-v1", {"max_steps": 4}, 4),
    ("grpo-standard-smoke-v1", {"max_steps": 4}, 4),
])
def test_resolved_recipe_is_run_bound_and_verifiable(tmp_path, profile_id, override, expected):
    _workspace(tmp_path)
    path, summary = materialize_stage_recipe(tmp_path, profile_id, "run-test-stage", override)

    assert summary["effective"]["max_steps"] == expected
    verified = verify_materialized_stage_recipe(
        tmp_path, path, profile_id=profile_id, run_id="run-test-stage",
    )
    assert verified["training"]["max_steps"] == expected


def test_resolved_recipe_rejects_tampering_and_out_of_range_parameters(tmp_path):
    _workspace(tmp_path)
    with pytest.raises(ValueError, match="max_steps"):
        materialize_stage_recipe(tmp_path, "dpo-smoke-v1", "run-out-of-range", {"max_steps": 100000})

    path, _ = materialize_stage_recipe(tmp_path, "dpo-smoke-v1", "run-tampered", {"max_steps": 4})
    recipe_path = tmp_path / path
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    recipe["training"]["max_steps"] = 5
    recipe_path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    with pytest.raises(ValueError, match="resolved recipe"):
        verify_materialized_stage_recipe(tmp_path, path, profile_id="dpo-smoke-v1", run_id="run-tampered")


def test_best_checkpoint_requires_validation_and_materializes_selection(tmp_path):
    _workspace(tmp_path)
    with pytest.raises(ValueError, match="训练期验证"):
        materialize_stage_recipe(tmp_path, "dpo-smoke-v1", "run-best-no-val", {}, checkpoint_policy="best")
    path, summary = materialize_stage_recipe(
        tmp_path, "dpo-smoke-v1", "run-best-with-val", {},
        validation={"mode": "registered", "status": "ready", "dataset_id": "ds-validation"},
        checkpoint_policy="best",
    )
    assert summary["checkpoint_selection"] == {
        "policy": "best", "metric": "loss", "greater_is_better": False,
    }
    recipe = verify_materialized_stage_recipe(
        tmp_path, path, profile_id="dpo-smoke-v1", run_id="run-best-with-val",
    )
    assert recipe["training"]["load_best_model_at_end"] is True
    assert recipe["training"]["metric_for_best_model"] == "loss"


def test_registered_sft_recipe_binds_verified_base_identity_and_path(tmp_path, monkeypatch):
    _workspace(tmp_path)
    model_id, model_path = _registered_base(tmp_path)
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight._validate_sft_model_config", lambda _: None)

    path, summary = materialize_stage_recipe(
        tmp_path, "sft-registered-v1", "run-base-sft", {}, parent_model_id=model_id,
    )
    recipe = verify_materialized_stage_recipe(
        tmp_path, path, profile_id="sft-registered-v1", run_id="run-base-sft",
    )

    assert recipe["parent_model_id"] == model_id
    assert recipe["parent_model_path"] == str(model_path.resolve())
    assert recipe["model"]["repo"] == str(model_path.resolve())
    assert recipe["_onegpu_ui"]["parent_model"]["model_id"] == model_id
    assert summary["config_sha256"]

    (model_path / "model.safetensors").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="Base 模型"):
        verify_materialized_stage_recipe(
            tmp_path, path, profile_id="sft-registered-v1", run_id="run-base-sft",
        )


def test_materialized_recipe_refuses_symlinked_launcher_directory(tmp_path):
    _workspace(tmp_path)
    profile = next(item for item in stage_profile_catalog(tmp_path).profiles if item.profile_id == "dpo-smoke-v1")
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    launcher_dir = tmp_path / ".onegpu-posttrain/launcher"
    launcher_dir.parent.mkdir()
    try:
        launcher_dir.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"当前 Windows 账户没有创建目录符号链接的权限：{exc}")

    with pytest.raises(ValueError, match="符号链接"):
        materialize_stage_recipe(tmp_path, profile.profile_id, "run-symlinked", {})
