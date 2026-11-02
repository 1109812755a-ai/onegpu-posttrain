"""Read-only stage prerequisites for the targeted-data workbench."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .launcher import _gpu_snapshot
from .sft_generation_setup import _profiles, _validated_local_configs
from .stage_profiles import stage_profile_catalog, stage_requirements_from_profiles


STAGE_REQUIREMENTS: dict[str, dict[str, Any]] = stage_requirements_from_profiles()


def stage_capabilities(
    workspace: str | Path,
    *,
    synthetic_generation_enabled: bool,
    experiment_launcher_enabled: bool,
) -> dict[str, Any]:
    """Describe actual local prerequisites without starting provider/training work."""
    root = Path(workspace).resolve()
    hardware = _gpu_snapshot()
    provider_profiles = _profiles(root)
    configured_provider = any(
        not profile["is_example"] and profile["credential_env_configured"]
        for profile in provider_profiles
    )
    sft_configs = _validated_local_configs(root)
    profile_catalog = stage_profile_catalog(root)
    profiles_by_stage: dict[str, list[dict[str, Any]]] = {}
    for profile in profile_catalog.profiles:
        capability_stage = "preference" if profile.stage in {"dpo", "simpo"} else profile.stage
        profiles_by_stage.setdefault(capability_stage, []).append(profile.model_dump(mode="json"))
    stages = []
    for stage, contract in STAGE_REQUIREMENTS.items():
        recipes = [
            {"ref": relative, "available": _workspace_file(root, relative)}
            for relative in contract["recipes"]
        ]
        reward_ref = contract["reward_contract"]
        reward_available = _workspace_file(root, reward_ref) if reward_ref else None
        generation_available = stage in {"sft", "cpt", "grpo"} and synthetic_generation_enabled
        stage_profiles = profiles_by_stage.get(stage, [])
        adapters_available = bool(stage_profiles) and all(
            profile.get("execution_supported") is True for profile in stage_profiles
        )
        training_ui_available = experiment_launcher_enabled and adapters_available
        evaluation_available = all(
            _workspace_file(root, ref)
            for ref in _evaluation_refs(stage)
        )
        generation_blockers = []
        if not generation_available:
            generation_blockers.append(
                (f"请使用 --enable-synthetic-generation 显式启用 {stage.upper()} 生成"
                 if stage in {"sft", "cpt", "grpo"} else
                 "当前未接通该阶段的数据生成适配器；可导入已登记 canonical 数据集继续训练")
            )
        training_blockers = []
        if not training_ui_available:
            training_blockers.append(
                "请使用 --enable-experiment-launcher 显式启用本地阶段 Launcher" if adapters_available
                else "阶段 recipe 或 Web Launcher adapter 尚未就绪"
            )
        if not all(item["available"] for item in recipes):
            training_blockers.append("阶段训练配方缺失")
        if reward_ref and not reward_available:
            training_blockers.append("GRPO 可执行 reward 插件缺失")
        if not evaluation_available:
            training_blockers.append("阶段评测配置/协议缺失")
        training_blockers.extend(
            profile.get("execution_blocker") for profile in stage_profiles
            if profile.get("execution_blocker")
        )
        evaluation_blockers = [] if evaluation_available else ["阶段评测配置/协议缺失"]
        stages.append({
            "stage": stage,
            "dataset_schema": contract["dataset_schema"],
            "parent_model_stage": contract["parent_stage"],
            "parent_model": {"required": True, "status": "select_registered_artifact"},
            "recipes": recipes,
            "verifier": {
                "contract": contract["verifier"],
                "status": "preflight_required",
                "generation_quality_adapter": ("sft_candidate_v1" if stage == "sft" and generation_available else
                                               "cpt_corpus_v1" if stage == "cpt" and generation_available else None),
            },
            "reward_contract": ({
                "ref": reward_ref, "available": reward_available,
                # File presence is not proof that the selected plan/dataset uses this
                # exact executable contract, so callers must still verify its hashes.
                "status": "present_unverified" if reward_available else "missing",
            } if reward_ref else None),
            "hardware": {key: hardware.get(key) for key in (
                "available", "status", "name", "total_mib", "used_mib", "free_mib", "reason", "observed_at",
            ) if key in hardware},
            "evaluation": {"supported_metrics": contract["evaluators"], "available": evaluation_available},
            "generation_ui_available": generation_available,
            "training_ui_available": training_ui_available,
            # A stage catalog cannot verify a selected bundle or grant launch permission.
            "data_generation_ready": False,
            "training_ready": not training_blockers,
            "ready_for_execution": not training_blockers,
            "evaluation_ready": not evaluation_blockers,
            "generation_blockers": generation_blockers,
            "generation_modes": _generation_modes(
                stage, generation_available, configured_provider, sft_configs
            ),
            "training_blockers": training_blockers,
            "evaluation_blockers": evaluation_blockers,
            "blockers": training_blockers,
            "profiles": stage_profiles,
        })
    return {"schema_version": "1.0", "read_only": True, "stages": stages}


def _generation_modes(
    stage: str, adapter_available: bool, provider_available: bool,
    sft_configs: dict[str, list[dict[str, str]]],
) -> dict[str, dict[str, Any]]:
    """Describe catalog-level prerequisites, without granting launch permission.

    Dataset identity, selected provider, budget and targeting evidence must still
    be checked by the eventual request-specific preflight.
    """
    modes: dict[str, dict[str, Any]] = {}
    for mode in ("ordinary", "targeted"):
        blockers: list[str] = []
        pending_checks = ["所选 Provider、预算及输入身份需在具体预检中核验"]
        if not adapter_available:
            blockers.append(_adapter_blocker(stage, mode))
        if not provider_available:
            blockers.append("未发现已配置凭据的真实 Provider")
        if stage == "sft":
            if not sft_configs["templates"] or not sft_configs["protocols"]:
                pending_checks.append("基于种子创建新方案需检查 SFT 模板与候选输出协议；主题生成使用内置协议")
        else:
            blockers.extend(_stage_generation_blockers(stage))
        if stage == "cpt":
            pending_checks.append("本地污染索引仅覆盖示例数据；事实正确性不会自动验证")
        if mode == "targeted":
            pending_checks.append("需核验定向计划和角色隔离证据")
        if stage == "grpo":
            pending_checks.extend([
                "候选输出仅包含 prompt；登记/训练前需独立绑定可信 reference 与 reward 证据",
                "定向模式仅使用精确计划绑定的 train_seed，不读取 diagnostic/final holdout",
            ])
        modes[mode] = {
            "adapter_available": adapter_available,
            "domain_adapter": _domain_adapter_status(stage, mode),
            "candidate_protocol": _candidate_protocol_status(stage),
            "quality_verifier": _quality_verifier_status(stage),
            "provider_configured": provider_available,
            "status": "blocked" if blockers else "preflight_required",
            "blockers": blockers,
            "pending_checks": pending_checks,
            "provider_calls": 0,
            "note": "只读能力检查；具体配置仍需单独预检，不能作为启动许可。",
        }
    return modes


def _adapter_blocker(stage: str, mode: str) -> str:
    if stage in {"sft", "cpt", "grpo"}:
        return f"{stage.upper()} 生成任务服务未启用；请使用 --enable-synthetic-generation 显式启用"
    labels = {"cpt": "CPT 领域语料", "preference": "Preference 同题候选与排序", "grpo": "GRPO prompt"}
    return f"{labels[stage]}的模型生成尚未接入共享 Web 任务分发（{mode}）"


def _domain_adapter_status(stage: str, mode: str) -> str:
    """Expose reusable domain code without implying a runnable Web generation path."""
    if stage == "sft":
        return "shared_sft_job" if mode in {"ordinary", "targeted"} else "unavailable"
    if stage == "cpt":
        return "shared_cpt_job" if mode in {"ordinary", "targeted"} else "unavailable"
    if stage == "grpo":
        return "prompt_only_candidate_job" if mode in {"ordinary", "targeted"} else "unavailable"
    if stage == "preference" and mode == "targeted":
        return "targeted_pair_builder_only"
    if stage == "grpo" and mode == "targeted":
        return "seed_derivation_only"
    return "not_implemented"


def _candidate_protocol_status(stage: str) -> str:
    if stage in {"sft", "cpt"}:
        return "available"  # Built-in prompt-only protocol or the snapshotted bundle protocol.
    if stage == "grpo":
        return "prompt_only_not_trainable_without_reference"
    return "not_connected_to_generation_dispatch"


def _quality_verifier_status(stage: str) -> str:
    if stage == "sft":
        return "available_for_review"
    if stage == "preference":
        return "targeted_reference_verifier_only"
    if stage == "grpo":
        return "prompt_schema_only_reference_required"
    if stage == "cpt":
        return "available_static_checks_factuality_not_verified"
    return "unavailable"


def _stage_generation_blockers(stage: str) -> list[str]:
    if stage == "cpt":
        return []
    blockers = ["该阶段的候选输出协议尚未接入普通/定向生成分发"]
    if stage == "preference":
        blockers.append("Preference 定向配对器可复用；普通入口、排序证据契约与 Web 质检尚未接通")
    elif stage == "grpo":
        # Prompt-only generation is available, but it is deliberately not a
        # trainable dataset path until trusted reference evidence is attached.
        pass
    return blockers


def _workspace_file(root: Path, relative: str) -> bool:
    path = root / relative
    return not path.is_symlink() and path.is_file() and path.resolve().is_relative_to(root)


def _evaluation_refs(stage: str) -> tuple[str, ...]:
    if stage == "cpt":
        return ("configs/evaluation/math-v1.yaml",)
    if stage == "sft":
        return ("configs/evaluation/math-v1-launcher-mini.yaml",)
    if stage == "preference":
        return ("configs/evaluation/math-v1.yaml",)
    return ("configs/evaluation/math-v1.yaml", "plugins/grpo_rewards.py")
