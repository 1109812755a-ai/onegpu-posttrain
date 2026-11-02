"""Materialize and verify bounded stage recipe overrides for a single confirmed Run."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from ..lineage import sha256_file, sha256_json
from .stage_profiles import resolve_training_memory_mode, resolve_training_parameters, stage_profile_catalog

_RUN_ID = re.compile(r"^run-[a-z0-9][a-z0-9-]{0,119}$")


def materialize_stage_recipe(workspace: str | Path, profile_id: str, run_id: str,
                             overrides: dict[str, int | float], *,
                             validation: dict[str, Any] | None = None,
                             memory_mode_id: str | None = None,
                             checkpoint_policy: str = "final",
                             parent_model_id: str | None = None) -> tuple[str, dict[str, Any]]:
    """Create an immutable, workspace-owned resolved recipe; never accept a user path."""
    root = Path(workspace).resolve(strict=True)
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("无效阶段训练 run_id")
    profile = next((item for item in stage_profile_catalog(root).profiles if item.profile_id == profile_id), None)
    if profile is None or not profile.execution_supported:
        raise ValueError("阶段配方不可用")
    source = root / profile.approved_recipe
    if source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(root):
        raise ValueError("批准的阶段配方不存在或越界")
    recipe = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict) or not isinstance(recipe.get("training"), dict):
        raise ValueError("批准配方缺少 training 配置")
    parent_identity = None
    if profile.parent_model_rule == "registered_sft_parent":
        if not isinstance(parent_model_id, str) or not parent_model_id:
            raise ValueError("该 SFT 配方必须绑定已登记父模型")
        from .training_preflight import resolve_registered_sft_parent_path
        model_path, model = resolve_registered_sft_parent_path(root, parent_model_id)
        parent_identity = {"model_id": model.model_id, "sha256": model.digest.value}
        recipe["parent_model_id"] = model.model_id
        recipe["parent_model_path"] = str(model_path)
        model_config = recipe.setdefault("model", {})
        if not isinstance(model_config, dict):
            raise ValueError("批准 recipe 的 model 配置格式无效")
        model_config.update({"repo": str(model_path), "role": model.stage})
    effective = resolve_training_parameters(profile, overrides)
    if checkpoint_policy not in {"final", "best"}:
        raise ValueError("checkpoint_policy 仅允许 final 或 best")
    validation = validation or {"mode": "none", "status": "ready"}
    if checkpoint_policy == "best":
        if validation.get("mode") == "none" or validation.get("status") != "ready":
            raise ValueError("选择 best 必须启用有效的训练期验证数据")
        recipe["training"].update({
            "load_best_model_at_end": True,
            "metric_for_best_model": profile.checkpoint_metric,
            "greater_is_better": profile.checkpoint_greater_is_better,
        })
    checkpoint_selection = {
        "policy": checkpoint_policy,
        "metric": profile.checkpoint_metric if checkpoint_policy == "best" else None,
        "greater_is_better": profile.checkpoint_greater_is_better if checkpoint_policy == "best" else None,
    }
    memory_mode = resolve_training_memory_mode(profile, memory_mode_id)
    recipe["training"] = {**recipe["training"], **effective, **memory_mode.settings}
    recipe_digest = sha256_file(source)
    training_identity = {"profile_id": profile.profile_id,
                         "recipe_sha256": recipe_digest,
                         "training": recipe["training"]}
    if parent_identity is not None:
        training_identity["parent_model"] = parent_identity
    training_config_digest = sha256_json(training_identity)
    config_digest = sha256_json({"training_config_sha256": training_config_digest,
                                 "validation": validation,
                                 "checkpoint_selection": checkpoint_selection})
    recipe["_onegpu_ui"] = {
        "profile_id": profile.profile_id,
        "run_id": run_id,
        "approved_recipe": profile.approved_recipe,
        "approved_recipe_sha256": recipe_digest,
        "parameter_overrides": overrides,
        "memory_mode_id": memory_mode.mode_id,
        "validation": validation,
        "checkpoint_selection": checkpoint_selection,
        "parent_model": parent_identity,
        "training_config_sha256": training_config_digest,
        "effective_config_sha256": config_digest,
    }
    target = root / ".onegpu-posttrain" / "launcher" / "configs" / f"{run_id}.yaml"
    current = root
    for part in target.parent.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("resolved recipe 路径不能经过符号链接")
        current.mkdir(exist_ok=True)
        if not current.resolve(strict=True).is_relative_to(root):
            raise ValueError("resolved recipe 路径越界")
    if target.exists() or target.is_symlink() or not target.resolve(strict=False).is_relative_to(root):
        raise FileExistsError("本次 Run 的 resolved recipe 已存在或越界")
    target.write_text(yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return target.relative_to(root).as_posix(), {
        "defaults": {item.key: item.default for item in profile.training_parameters},
        "overrides": dict(overrides), "effective": effective,
        "config_sha256": config_digest, "validation": validation,
        "checkpoint_selection": checkpoint_selection,
        "memory_mode_id": memory_mode.mode_id, "memory_mode_settings": memory_mode.settings,
    }


def verify_materialized_stage_recipe(workspace: str | Path, path: str | Path, *,
                                     profile_id: str, run_id: str) -> dict[str, Any]:
    """Check a CLI-provided resolved recipe against its immutable approved profile."""
    root = Path(workspace).resolve(strict=True)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    original = candidate
    try:
        original.absolute().relative_to(root)
    except ValueError as exc:
        raise ValueError("resolved recipe 必须位于当前 workspace") from exc
    current = original
    while current != root and current != current.parent:
        if current.is_symlink():
            raise ValueError("resolved recipe 路径不能经过符号链接")
        current = current.parent
    candidate = candidate.resolve(strict=True)
    candidate.relative_to(root)
    expected = (root / ".onegpu-posttrain" / "launcher" / "configs" / f"{run_id}.yaml").resolve(strict=True)
    if candidate != expected or not candidate.is_file() or candidate.is_symlink():
        raise ValueError("resolved recipe 必须是该 Run 在本地 Launcher 下生成的配置")
    recipe = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    metadata = recipe.get("_onegpu_ui") if isinstance(recipe, dict) else None
    if not isinstance(metadata, dict) or metadata.get("profile_id") != profile_id or metadata.get("run_id") != run_id:
        raise ValueError("resolved recipe 的 profile/Run 身份不匹配")
    profile = next((item for item in stage_profile_catalog(root).profiles if item.profile_id == profile_id), None)
    if profile is None or metadata.get("approved_recipe") != profile.approved_recipe:
        raise ValueError("resolved recipe 未基于该 profile 的批准配方")
    source = root / profile.approved_recipe
    if source.is_symlink() or not source.is_file() or sha256_file(source) != metadata.get("approved_recipe_sha256"):
        raise ValueError("批准配方在预检后已变化")
    parent_identity = None
    if profile.parent_model_rule == "registered_sft_parent":
        declared_parent = metadata.get("parent_model")
        if not isinstance(declared_parent, dict):
            raise ValueError("resolved recipe 缺少登记父模型身份")
        from .training_preflight import resolve_registered_sft_parent_path
        parent_path, model = resolve_registered_sft_parent_path(root, str(declared_parent.get("model_id", "")))
        parent_identity = {"model_id": model.model_id, "sha256": model.digest.value}
        model_config = recipe.get("model")
        if (declared_parent != parent_identity or not isinstance(model_config, dict)
                or model_config.get("repo") != str(parent_path)
                or recipe.get("parent_model_id") != model.model_id
                or recipe.get("parent_model_path") != str(parent_path)):
            raise ValueError("resolved recipe 的父模型身份或路径已变化")
    overrides = metadata.get("parameter_overrides")
    if not isinstance(overrides, dict):
        raise ValueError("resolved recipe 参数变更摘要无效")
    effective = resolve_training_parameters(profile, overrides)
    memory_mode_id = metadata.get("memory_mode_id")
    if not isinstance(memory_mode_id, str):
        raise ValueError("resolved recipe 显存模式无效")
    memory_mode = resolve_training_memory_mode(profile, memory_mode_id)
    training = recipe.get("training")
    ui_checkpoint = metadata.get("checkpoint_selection")
    if not isinstance(ui_checkpoint, dict) or ui_checkpoint.get("policy") not in {"final", "best"}:
        raise ValueError("resolved recipe checkpoint selection 无效")
    if ui_checkpoint["policy"] == "best":
        if metadata.get("validation", {}).get("mode") == "none":
            raise ValueError("best checkpoint selection requires training validation")
        expected_best_args = {
            "load_best_model_at_end": True,
            "metric_for_best_model": profile.checkpoint_metric,
            "greater_is_better": profile.checkpoint_greater_is_better,
        }
    else:
        expected_best_args = {}
    expected_training = {**effective, **memory_mode.settings, **expected_best_args}
    if not isinstance(training, dict) or any(training.get(key) != value for key, value in expected_training.items()):
        raise ValueError("resolved recipe 的有效参数与批准边界不一致")
    training_identity = {"profile_id": profile.profile_id,
                         "recipe_sha256": metadata["approved_recipe_sha256"],
                         "training": training}
    if parent_identity is not None:
        training_identity["parent_model"] = parent_identity
    training_config_digest = sha256_json(training_identity)
    if metadata.get("training_config_sha256") != training_config_digest:
        raise ValueError("resolved recipe 训练配置摘要不匹配")
    validation = metadata.get("validation")
    if not isinstance(validation, dict) or validation.get("mode") not in {"none", "registered", "random_split"}:
        raise ValueError("resolved recipe 验证配置无效")
    expected_digest = sha256_json({"training_config_sha256": training_config_digest,
                                   "validation": validation,
                                   "checkpoint_selection": ui_checkpoint})
    if expected_digest != metadata.get("effective_config_sha256"):
        raise ValueError("resolved recipe 配置摘要不匹配")
    return recipe
