"""Closed, inspectable profiles for stage training and evaluation workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

StageId = Literal["cpt", "sft", "dpo", "simpo", "grpo"]
AlgorithmId = Literal["cpt", "sft", "dpo", "simpo", "grpo"]
ParentModelRule = Literal["external_base", "approved_mini_cpt", "registered_sft_parent", "registered_sft_merged"]


class EvaluationProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str
    protocol: str
    dataset_roles: list[str]
    metrics: list[str]
    metric_sources: dict[str, str]
    denominator_semantics: dict[str, str]


class TrainingParameter(BaseModel):
    """One server-owned, bounded training parameter exposed by a profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    label_zh: str
    kind: Literal["int", "float"]
    default: int | float
    minimum: int | float
    maximum: int | float
    choices: list[int | float] = Field(default_factory=list)


class TrainingMemoryMode(BaseModel):
    """A server-owned, user-selectable training memory preset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode_id: str
    label_zh: str
    description_zh: str
    default: bool = False
    settings: dict[str, int | float | str | bool] = Field(default_factory=dict)


class StageProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str
    stage: StageId
    algorithm: AlgorithmId
    approved_recipe: str
    parent_model_stages: list[str]
    parent_model_rule: ParentModelRule
    dataset_stage: str
    dataset_roles: list[str]
    adapter_id: str
    output_artifact_kinds: list[str]
    output_verification: str
    evaluation_profiles: list[EvaluationProfile]
    execution_supported: bool
    training_validation_supported: bool = True
    validation_dataset_roles: list[str] = Field(default_factory=lambda: ["validation", "diagnostic_dev"])
    validation_min_samples: int = Field(default=2, ge=1, le=1000)
    checkpoint_metric: str = "loss"
    checkpoint_greater_is_better: bool = False
    execution_blocker: str | None = None
    reward_contract: str | None = None
    training_parameters: list[TrainingParameter] = Field(default_factory=list)
    training_memory_modes: list[TrainingMemoryMode] = Field(default_factory=list)


class StageProfileCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    profiles: list[StageProfile]


_UNIFIED_EVAL = EvaluationProfile(
    profile_id="shared-math-eval-v1",
    protocol="math-eval-v1",
    dataset_roles=["benchmark", "diagnostic_dev", "final_holdout"],
    metrics=["gsm8k.accuracy", "math_500.accuracy", "internal_holdout.accuracy"],
    metric_sources={
        "gsm8k.accuracy": "EvalScope GSM8K report",
        "math_500.accuracy": "EvalScope MATH-500 report",
        "internal_holdout.accuracy": "internal math holdout evaluator",
    },
    denominator_semantics={
        "gsm8k.accuracy": "EvalScope 实际完成样本数；不是请求上限",
        "math_500.accuracy": "EvalScope 实际完成样本数；不是请求上限",
        "internal_holdout.accuracy": "内部评测器实际判分样本数",
    },
)
_SFT_EVAL = _UNIFIED_EVAL.model_copy(update={"profile_id": "sft-math-eval-v1"})
_CPT_EVAL = EvaluationProfile(
    profile_id="cpt-math-v1",
    protocol="math-eval-v1",
    dataset_roles=["cpt_holdout", "benchmark", "diagnostic_dev", "final_holdout"],
    metrics=["domain_ppl.perplexity", "gsm8k.accuracy", "math_500.accuracy"],
    metric_sources={
        "domain_ppl.perplexity": "canonical CPT domain holdout + token-weighted NLL evaluator",
        "gsm8k.accuracy": "EvalScope GSM8K report",
        "math_500.accuracy": "EvalScope MATH-500 report",
    },
    denominator_semantics={
        "domain_ppl.perplexity": "模型实际预测 token 数（predicted_tokens）；记录样本数和截断样本数另列",
        "gsm8k.accuracy": "EvalScope 实际完成样本数；不是请求上限",
        "math_500.accuracy": "EvalScope 实际完成样本数；不是请求上限",
    },
)
_PREFERENCE_EVAL = EvaluationProfile(
    profile_id="preference-heldout-v1",
    protocol="response-mean-logprob-v1",
    dataset_roles=["preference_eval"],
    metrics=["preference.accuracy", "preference.margin"],
    metric_sources={
        "preference.accuracy": "held-out chosen/rejected pair evaluator",
        "preference.margin": "held-out chosen-minus-rejected score evaluator",
    },
    denominator_semantics={
        "preference.accuracy": "實際完成評分的 preference pairs 數",
        "preference.margin": "實際完成評分的 preference pairs 數",
    },
)
_GRPO_EVAL = EvaluationProfile(
    profile_id="grpo-math-eval-v1",
    protocol="math-eval-v1",
    dataset_roles=["benchmark", "diagnostic_dev", "final_holdout"],
    metrics=[
        "gsm8k.accuracy", "math_500.accuracy", "internal_holdout.accuracy",
        "grpo_training.reward", "grpo_training.reward_std", "grpo_training.frac_reward_zero_std",
        "grpo_training.kl", "grpo_training.completions_mean_length",
        "grpo_training.completions_clipped_ratio",
        "grpo_training.reward_components.onegpu_accuracy",
        "grpo_training.reward_components.onegpu_boxed_format",
    ],
    metric_sources={
        "gsm8k.accuracy": "EvalScope GSM8K report",
        "math_500.accuracy": "EvalScope MATH-500 report",
        "internal_holdout.accuracy": "internal math holdout evaluator",
        "grpo_training.reward": "producing GRPO training Run trainer_state.json",
        "grpo_training.reward_std": "producing GRPO training Run trainer_state.json",
        "grpo_training.frac_reward_zero_std": "producing GRPO training Run trainer_state.json",
        "grpo_training.kl": "producing GRPO training Run trainer_state.json; N/A when beta=0",
        "grpo_training.completions_mean_length": "producing GRPO training Run trainer_state.json",
        "grpo_training.completions_clipped_ratio": "producing GRPO training Run trainer_state.json",
        "grpo_training.reward_components.onegpu_accuracy": "producing GRPO training Run reward component logs",
        "grpo_training.reward_components.onegpu_boxed_format": "producing GRPO training Run reward component logs",
    },
    denominator_semantics={
        "gsm8k.accuracy": "EvalScope 实际完成样本数",
        "math_500.accuracy": "EvalScope 实际完成样本数",
        "internal_holdout.accuracy": "内部评测器实际判分样本数",
        "grpo_training.reward": "训练日志中实际记录的 trainer log points；不是训练样本数",
        "grpo_training.reward_std": "训练日志中实际记录的 trainer log points；不是 rollout 数",
        "grpo_training.frac_reward_zero_std": "trainer 记录的聚合诊断，不推断 prompt-level denominator",
        "grpo_training.kl": "训练日志点；beta=0 时 not_applicable",
        "grpo_training.completions_mean_length": "训练日志点；单位 token",
        "grpo_training.completions_clipped_ratio": "训练日志点；比率",
        "grpo_training.reward_components.onegpu_accuracy": "训练日志中实际记录该 reward 的 log points",
        "grpo_training.reward_components.onegpu_boxed_format": "训练日志中实际记录该 reward 的 log points",
    },
)
_CHOICE_EVAL = EvaluationProfile(
    profile_id="choice-accuracy-v1",
    protocol="choice-accuracy-v1",
    dataset_roles=["evaluation_only"],
    metrics=["choice_accuracy.single_choice", "choice_accuracy.multiple_choice",
             "choice_accuracy.true_false", "choice_accuracy.overall"],
    metric_sources={
        "choice_accuracy.single_choice": "registered objective-choice suite; exact option match",
        "choice_accuracy.multiple_choice": "registered objective-choice suite; exact answer-set match",
        "choice_accuracy.true_false": "registered objective-choice suite; exact normalized truth match",
        "choice_accuracy.overall": "registered objective-choice suite; exact match across all records",
    },
    denominator_semantics={
        "choice_accuracy.single_choice": "该 suite 中实际完成的单选题数；解析失败仍计入分母",
        "choice_accuracy.multiple_choice": "该 suite 中实际完成的多选题数；答案集合须完全匹配",
        "choice_accuracy.true_false": "该 suite 中实际完成的判断题数；解析失败仍计入分母",
        "choice_accuracy.overall": "该 suite 中实际完成的全部客观题数；解析失败仍计入分母",
    },
)

_PARAMETER_RULES: dict[str, dict[str, tuple[str, int | float, int | float, tuple[int | float, ...]]]] = {
    "cpt": {
        "max_length": ("int", 512, 4047, ()),
        "max_steps": ("int", 1, 100, ()),
        "learning_rate": ("float", 0.000001, 0.0005, ()),
        "gradient_accumulation_steps": ("int", 1, 32, ()),
    },
    "sft": {
        "max_length": ("int", 512, 4047, ()),
        "num_train_epochs": ("int", 1, 3, ()),
        "learning_rate": ("float", 0.000001, 0.0005, ()),
        "gradient_accumulation_steps": ("int", 1, 32, ()),
    },
    "dpo": {
        "max_length": ("int", 512, 4047, ()),
        "max_steps": ("int", 1, 100, ()),
        "learning_rate": ("float", 0.000001, 0.0005, ()),
        "gradient_accumulation_steps": ("int", 1, 32, ()),
    },
    "simpo": {
        "max_length": ("int", 512, 4047, ()),
        "max_steps": ("int", 1, 100, ()),
        "learning_rate": ("float", 0.000001, 0.0005, ()),
        "gradient_accumulation_steps": ("int", 1, 32, ()),
    },
    "grpo": {
        "max_prompt_length_candidate": ("int", 128, 512, (128, 256, 512)),
        "max_completion_length_candidate": ("int", 128, 512, (128, 256, 512)),
        "max_steps": ("int", 1, 100, ()),
        "learning_rate": ("float", 0.0000001, 0.0001, ()),
        "gradient_accumulation_steps": ("int", 1, 16, ()),
    },
}


def resolve_training_parameters(profile: StageProfile, overrides: dict[str, int | float] | None = None) -> dict[str, int | float]:
    """Validate user overrides against a profile's bounded parameter declarations."""
    overrides = overrides or {}
    specs = {item.key: item for item in profile.training_parameters}
    unknown = sorted(set(overrides) - set(specs))
    if unknown:
        raise ValueError(f"不支持的训练参数：{', '.join(unknown)}")
    resolved = {key: item.default for key, item in specs.items()}
    for key, value in overrides.items():
        spec = specs[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"训练参数 {key} 必须是 {spec.kind} 数值")
        if spec.kind == "int" and not isinstance(value, int):
            raise ValueError(f"训练参数 {key} 必须是整数")
        if value < spec.minimum or value > spec.maximum:
            raise ValueError(f"训练参数 {key} 必须在 {spec.minimum} 到 {spec.maximum} 之间")
        if spec.choices and value not in spec.choices:
            raise ValueError(f"训练参数 {key} 只允许：{', '.join(map(str, spec.choices))}")
        resolved[key] = value
    return resolved


def resolve_training_memory_mode(profile: StageProfile, mode_id: str | None = None) -> TrainingMemoryMode:
    """Resolve a mode ID to its approved server-owned settings."""
    modes = profile.training_memory_modes
    if not modes:
        if mode_id not in (None, "lora"):
            raise ValueError(f"阶段 {profile.stage} 不支持训练显存模式 {mode_id}")
        return TrainingMemoryMode(mode_id="lora", label_zh="标准 LoRA", description_zh="使用批准配方的默认 LoRA 配置。", default=True)
    selected_id = mode_id or next((item.mode_id for item in modes if item.default), modes[0].mode_id)
    selected = next((item for item in modes if item.mode_id == selected_id), None)
    if selected is None:
        raise ValueError(f"不支持的训练显存模式：{selected_id}")
    return selected

_PROFILE_DEFINITIONS = (
    StageProfile(
        profile_id="cpt-smoke-v1", stage="cpt", algorithm="cpt",
        approved_recipe="recipes/smoke/10-peft-cpt.yaml", parent_model_stages=["base"], parent_model_rule="external_base",
        dataset_stage="cpt", dataset_roles=["train"], adapter_id="cpt-cli-v1",
        output_artifact_kinds=["adapter", "merged_model"], output_verification="model_manifest_and_digest",
        evaluation_profiles=[_CPT_EVAL, _UNIFIED_EVAL, _CHOICE_EVAL], execution_supported=True,
        validation_dataset_roles=["diagnostic_dev"],
    ),
    StageProfile(
        profile_id="sft-mini-1024-v1", stage="sft", algorithm="sft",
        approved_recipe="recipes/mini/23-sft-1024-fallback.yaml", parent_model_stages=["cpt"], parent_model_rule="approved_mini_cpt",
        dataset_stage="sft", dataset_roles=["train"], adapter_id="mini-sft-launcher-v1",
        output_artifact_kinds=["adapter", "merged_model"], output_verification="model_manifest_and_digest",
        evaluation_profiles=[_SFT_EVAL, _CHOICE_EVAL], execution_supported=True,
        training_memory_modes=[
            TrainingMemoryMode(mode_id="lora", label_zh="标准 LoRA", description_zh="使用批准配方的默认 LoRA 配置。", default=True),
            TrainingMemoryMode(mode_id="qlora-4bit-nf4", label_zh="低显存 QLoRA（4-bit NF4）",
                               description_zh="固定 BNB 4-bit NF4、双重量化与 FP16 计算；长序列仍可能 OOM。",
                               settings={"quant_method": "bnb", "quant_bits": 4,
                                         "bnb_4bit_compute_dtype": "float16", "bnb_4bit_quant_type": "nf4",
                                         "bnb_4bit_use_double_quant": True}),
        ],
    ),
    StageProfile(
        profile_id="sft-registered-v1", stage="sft", algorithm="sft",
        approved_recipe="recipes/mini/24-sft-registered-parent.yaml",
        parent_model_stages=["base", "cpt"], parent_model_rule="registered_sft_parent",
        dataset_stage="sft", dataset_roles=["train"], adapter_id="registered-sft-v1",
        output_artifact_kinds=["adapter", "merged_model"], output_verification="model_manifest_and_digest",
        evaluation_profiles=[_SFT_EVAL, _CHOICE_EVAL], execution_supported=True,
        training_memory_modes=[
            TrainingMemoryMode(mode_id="lora", label_zh="标准 LoRA", description_zh="使用批准配方的默认 LoRA 配置。", default=True),
            TrainingMemoryMode(mode_id="qlora-4bit-nf4", label_zh="低显存 QLoRA（4-bit NF4）",
                               description_zh="固定 BNB 4-bit NF4、双重量化与 FP16 计算；长序列仍可能 OOM。",
                               settings={"quant_method": "bnb", "quant_bits": 4,
                                         "bnb_4bit_compute_dtype": "float16", "bnb_4bit_quant_type": "nf4",
                                         "bnb_4bit_use_double_quant": True}),
        ],
    ),
    StageProfile(
        profile_id="dpo-smoke-v1", stage="dpo", algorithm="dpo",
        approved_recipe="recipes/smoke/30-dpo.yaml", parent_model_stages=["sft"], parent_model_rule="registered_sft_merged",
        dataset_stage="preference", dataset_roles=["train_pairs"], adapter_id="dpo-cli-v1",
        output_artifact_kinds=["adapter", "merged_model"], output_verification="model_manifest_and_digest",
        evaluation_profiles=[_PREFERENCE_EVAL, _UNIFIED_EVAL, _CHOICE_EVAL], execution_supported=True,
    ),
    StageProfile(
        profile_id="simpo-smoke-v1", stage="simpo", algorithm="simpo",
        approved_recipe="recipes/smoke/31-simpo-original.yaml", parent_model_stages=["sft"], parent_model_rule="registered_sft_merged",
        dataset_stage="preference", dataset_roles=["train_pairs"], adapter_id="simpo-cli-v1",
        output_artifact_kinds=["adapter", "merged_model"], output_verification="model_manifest_and_digest",
        evaluation_profiles=[_PREFERENCE_EVAL, _UNIFIED_EVAL, _CHOICE_EVAL], execution_supported=True,
    ),
    StageProfile(
        profile_id="grpo-standard-smoke-v1", stage="grpo", algorithm="grpo",
        approved_recipe="recipes/smoke/40-grpo-standard.yaml", parent_model_stages=["sft"], parent_model_rule="registered_sft_merged",
        dataset_stage="grpo", dataset_roles=["train_prompts"], adapter_id="grpo-standard-cli-v1",
        output_artifact_kinds=["adapter", "merged_model"], output_verification="model_manifest_and_digest",
        evaluation_profiles=[_GRPO_EVAL, _CHOICE_EVAL], execution_supported=True, reward_contract="plugins/grpo_rewards.py",
    ),
    StageProfile(
        profile_id="grpo-lowmem-smoke-v1", stage="grpo", algorithm="grpo",
        approved_recipe="recipes/smoke/41-grpo-lowmem.yaml", parent_model_stages=["sft"], parent_model_rule="registered_sft_merged",
        dataset_stage="grpo", dataset_roles=["train_prompts"], adapter_id="grpo-lowmem-cli-v1",
        output_artifact_kinds=["adapter", "merged_model"], output_verification="model_manifest_and_digest",
        evaluation_profiles=[_GRPO_EVAL, _CHOICE_EVAL], execution_supported=True, reward_contract="plugins/grpo_rewards.py",
    ),
)


def stage_profile_catalog(workspace: str | Path) -> StageProfileCatalog:
    """Return only approved profiles, with paths constrained to the workspace."""
    root = Path(workspace).resolve()
    profiles = []
    for definition in _PROFILE_DEFINITIONS:
        profile = definition.model_copy(deep=True)
        recipe = _workspace_file(root, profile.approved_recipe)
        reward = _workspace_file(root, profile.reward_contract) if profile.reward_contract else True
        if not recipe:
            profile.execution_supported = False
            profile.execution_blocker = "批准的阶段 recipe 不存在或不在 workspace 内。"
        if not reward:
            profile.execution_supported = False
            profile.execution_blocker = "GRPO reward contract 不存在或不在 workspace 内。"
        if recipe is not None:
            try:
                recipe_data = yaml.safe_load(recipe.read_text(encoding="utf-8"))
                training = recipe_data.get("training") if isinstance(recipe_data, dict) else None
                if not isinstance(training, dict):
                    raise TypeError("approved recipe 缺少 training 配置")
                parameters = []
                for key, (kind, minimum, maximum, choices) in _PARAMETER_RULES[profile.stage].items():
                    default = training.get(key)
                    if isinstance(default, bool) or not isinstance(default, (int, float)):
                        raise TypeError(f"approved recipe 的 {key} 默认值缺失或类型无效")
                    if default < minimum or default > maximum or (choices and default not in choices):
                        raise ValueError(f"approved recipe 的 {key} 默认值超出 profile 允许范围")
                    parameters.append(TrainingParameter(
                        key=key, label_zh=_PARAMETER_LABELS[key], kind=kind, default=default,
                        minimum=minimum, maximum=maximum, choices=list(choices),
                    ))
                profile.training_parameters = parameters
            except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
                profile.execution_supported = False
                profile.execution_blocker = f"批准 recipe 参数配置无效：{str(exc)[:160]}"
        profiles.append(profile)
    return StageProfileCatalog(profiles=profiles)


_PARAMETER_LABELS = {
    "max_length": "最大序列长度（tokens）",
    "num_train_epochs": "训练轮数",
    "max_steps": "最大训练步数",
    "learning_rate": "学习率",
    "gradient_accumulation_steps": "梯度累积步数",
    "max_prompt_length_candidate": "最大提示长度（tokens）",
    "max_completion_length_candidate": "最大回答长度（tokens）",
}


def _workspace_file(root: Path, relative: str | None) -> Path | None:
    if not relative:
        return None
    path = root / relative
    try:
        resolved = path.resolve(strict=True)
        if path.is_symlink() or not resolved.is_file() or not resolved.is_relative_to(root):
            return None
        return resolved
    except (OSError, RuntimeError):
        return None


def stage_requirements_from_profiles() -> dict[str, dict[str, object]]:
    """Compatibility projection for the existing read-only prerequisites endpoint."""
    grouped: dict[str, list[StageProfile]] = {}
    for profile in _PROFILE_DEFINITIONS:
        capability_stage = "preference" if profile.stage in {"dpo", "simpo"} else profile.stage
        grouped.setdefault(capability_stage, []).append(profile)
    result: dict[str, dict[str, object]] = {}
    for stage, profiles in grouped.items():
        first = profiles[0]
        result[stage] = {
            "dataset_schema": (
                "preference-pair-v1" if stage == "preference"
                else "cpt-corpus-v1" if stage == "cpt"
                else f"{first.dataset_stage}-v1"
            ),
            "parent_stage": first.parent_model_stages[0],
            "recipes": list(dict.fromkeys(profile.approved_recipe for profile in profiles)),
            "evaluators": list(dict.fromkeys(evaluation.profile_id for profile in profiles for evaluation in profile.evaluation_profiles)),
            "verifier": first.output_verification,
            "reward_contract": first.reward_contract,
        }
    return result
