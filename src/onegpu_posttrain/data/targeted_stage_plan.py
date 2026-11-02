"""Immutable, stage-discriminated plans for targeted post-training data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..lineage import sha256_json
from .targeted import DataIdentity, EvaluationContract, MetricTarget, ModelIdentity


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class EvidenceRef(_FrozenModel):
    evidence_id: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_kind: Literal["scenario_report", "verification_contract", "reward_contract", "quality_report"]


class StageDataBudget(_FrozenModel):
    """Keep the desired usable amount separate from hard request limits."""

    unit: Literal["tokens", "records", "pairs", "prompts"]
    expected_usable_count: int = Field(ge=1)
    max_candidates: int = Field(ge=1)
    max_requests: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    max_retries: int = Field(default=0, ge=0)
    timeout_seconds: float = Field(default=120, gt=0, le=600, allow_inf_nan=False)


class CPTDomainCoverage(_FrozenModel):
    """A minimum token share for a named corpus subtopic, not a quality claim."""

    subtopic: str = Field(min_length=1, max_length=120)
    minimum_token_share: float = Field(ge=0, le=1, allow_inf_nan=False)


class CPTTarget(_FrozenModel):
    stage: Literal["cpt"]
    domain: str = Field(min_length=1, max_length=200)
    subtopics: tuple[str, ...] = Field(min_length=1)
    coverage_targets: tuple["CPTDomainCoverage", ...] = ()
    target_unit: Literal["tokens"] = "tokens"
    evidence: EvidenceRef | None = None

    @model_validator(mode="after")
    def validate_coverage_targets(self) -> "CPTTarget":
        topics = [item.subtopic for item in self.coverage_targets]
        if len(topics) != len(set(topics)):
            raise ValueError("CPT coverage_targets 中的 subtopic 不能重复")
        if any(topic not in self.subtopics for topic in topics):
            raise ValueError("CPT coverage_targets 必须属于已声明的 subtopics")
        if sum(item.minimum_token_share for item in self.coverage_targets) > 1.0:
            raise ValueError("CPT coverage_targets 的 minimum_token_share 总和不能超过 1")
        return self


class SFTTarget(_FrozenModel):
    stage: Literal["sft"]
    scenario_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    target_unit: Literal["records"] = "records"
    evidence: EvidenceRef | None = None


class PreferenceTarget(_FrozenModel):
    stage: Literal["preference"]
    scenario_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    target_unit: Literal["pairs"] = "pairs"
    evidence: EvidenceRef | None = None
    verification_contract: EvidenceRef | None = None

    @model_validator(mode="after")
    def validate_verification_contract(self) -> PreferenceTarget:
        if self.verification_contract and self.verification_contract.evidence_kind != "verification_contract":
            raise ValueError("Preference verification_contract 必须引用 verification_contract 证据")
        return self


class GRPOTarget(_FrozenModel):
    stage: Literal["grpo"]
    scenario_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    target_unit: Literal["prompts"] = "prompts"
    evidence: EvidenceRef | None = None
    reward_contract: EvidenceRef | None = None
    target_metric: MetricTarget | None = None
    guardrail_metrics: tuple[MetricTarget, ...] = ()

    @model_validator(mode="after")
    def validate_reward_contract(self) -> GRPOTarget:
        if self.reward_contract and self.reward_contract.evidence_kind != "reward_contract":
            raise ValueError("GRPO reward_contract 必须引用 reward_contract 证据")
        metric_names = [item.name for item in self.guardrail_metrics]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("GRPO guardrail_metrics 名称不能重复")
        if self.target_metric and self.target_metric.name in metric_names:
            raise ValueError("GRPO 目标指标不能同时声明为 guardrail")
        return self


StageTarget = Annotated[
    Union[CPTTarget, SFTTarget, PreferenceTarget, GRPOTarget],
    Field(discriminator="stage"),
]


class TargetedStagePlan(_FrozenModel):
    """A content-addressed plan revision; execution runs get separate run IDs."""

    schema_version: Literal["1.0"] = "1.0"
    version: int = Field(default=1, ge=1)
    target: StageTarget
    training_seed: DataIdentity
    starting_model: ModelIdentity
    diagnostic_evaluation: EvaluationContract
    final_evaluation: EvaluationContract
    provider_profile_id: str = Field(min_length=1)
    budget: StageDataBudget
    training_budget: int = Field(ge=1)
    comparison_design: Literal["before_after_pipeline", "matched_budget_ab"] = "before_after_pipeline"
    seed: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_stage_contract(self) -> TargetedStagePlan:
        expected_units = {
            "cpt": "tokens",
            "sft": "records",
            "preference": "pairs",
            "grpo": "prompts",
        }
        if self.training_seed.role != "train_seed":
            raise ValueError("定向计划只能引用 train_seed 数据，不可使用诊断集或最终 holdout")
        if self.budget.unit != expected_units[self.target.stage]:
            raise ValueError(f"{self.target.stage} 目标单位必须为 {expected_units[self.target.stage]}")
        if self.final_evaluation.dataset_id == self.training_seed.dataset_id:
            raise ValueError("final_holdout 必须与训练种子分离")
        if self.diagnostic_evaluation.role != "diagnostic_dev":
            raise ValueError("diagnostic_evaluation 必须使用 diagnostic_dev 角色")
        if self.final_evaluation.role != "final_holdout":
            raise ValueError("final_evaluation 必须使用 final_holdout 角色")
        if len({self.training_seed.dataset_id, self.diagnostic_evaluation.dataset_id,
                self.final_evaluation.dataset_id}) != 3:
            raise ValueError("训练种子、diagnostic_dev 和 final_holdout 必须相互独立")
        return self

    @property
    def config_sha256(self) -> str:
        return sha256_json(self.model_dump(mode="json"))

    @property
    def plan_id(self) -> str:
        return f"plan-targeted-{self.config_sha256[:20]}"


def save_targeted_stage_plan(plan: TargetedStagePlan, output_path: str | Path) -> dict[str, str | int]:
    """Persist one immutable revision. Existing files are never overwritten."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": plan.schema_version,
        "plan_id": plan.plan_id,
        "plan_version": plan.version,
        "config_sha256": plan.config_sha256,
        "plan": plan.model_dump(mode="json"),
        "status": "registered_not_executed",
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return {
        "plan_id": plan.plan_id,
        "plan_version": plan.version,
        "config_sha256": plan.config_sha256,
        "path": str(output),
    }


def registered_targeted_stage_plan_path(workspace: str | Path, plan_id: str, version: int) -> Path:
    if not plan_id.startswith("plan-targeted-") or not plan_id[14:].isalnum() or version < 1:
        raise ValueError("invalid targeted stage plan identity")
    root = Path(workspace).resolve(strict=True)
    return root / "artifacts" / "targeted-plans" / plan_id / f"v{version}" / "plan.json"


def register_targeted_stage_plan(workspace: str | Path, plan: TargetedStagePlan) -> dict[str, str | int]:
    output = registered_targeted_stage_plan_path(workspace, plan.plan_id, plan.version)
    root = Path(workspace).resolve(strict=True)
    if not output.resolve(strict=False).is_relative_to(root):
        raise ValueError("targeted stage plan path escapes workspace")
    return save_targeted_stage_plan(plan, output)


def load_registered_targeted_stage_plan(workspace: str | Path, plan_id: str, version: int) -> TargetedStagePlan:
    root = Path(workspace).resolve(strict=True)
    path = registered_targeted_stage_plan_path(root, plan_id, version)
    if path.is_symlink() or not path.is_file() or not path.resolve(strict=True).is_relative_to(root):
        raise FileNotFoundError("registered targeted stage plan not found")
    return load_targeted_stage_plan(path)


def load_targeted_stage_plan(path: str | Path) -> TargetedStagePlan:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("plan"), dict):
        raise TypeError("stage plan 必须是已登记的 YAML/JSON 对象")
    plan = TargetedStagePlan.model_validate(payload["plan"])
    if (
        payload.get("plan_id") != plan.plan_id
        or payload.get("plan_version") != plan.version
        or payload.get("config_sha256") != plan.config_sha256
    ):
        raise ValueError("stage plan 身份或摘要不匹配")
    return plan
