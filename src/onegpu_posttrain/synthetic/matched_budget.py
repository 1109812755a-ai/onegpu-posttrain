"""Strict comparability checks for matched-budget ordinary-vs-targeted data."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..data.targeted import MetricTarget
from ..lineage import sha256_json
from .effects import Guardrail, MetricDelta, _delta


class MatchedBudgetArm(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    arm: Literal["ordinary", "targeted"]
    parent_model_id: str = Field(min_length=1)
    trained_model_id: str = Field(min_length=1)
    training_dataset_id: str = Field(min_length=1)
    training_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_config_without_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    actual_training_budget: int = Field(ge=1)
    budget_unit: Literal["tokens", "steps", "examples"]
    eval_id: str = Field(min_length=1)
    eval_dataset_id: str = Field(min_length=1)
    eval_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    eval_item_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    eval_protocol_version: str = Field(min_length=1)
    eval_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metrics: dict[str, float]


class MatchedBudgetReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    report_id: str
    status: Literal["not_comparable", "guardrail_regression", "improved", "no_improvement"]
    comparable: bool
    reasons_not_comparable: list[str]
    ordinary_model_id: str
    targeted_model_id: str
    shared_parent_model_id: str
    ordinary_dataset_id: str
    targeted_dataset_id: str
    matched_training_budget: int | None
    budget_unit: str | None
    target_metric: str
    target_delta: MetricDelta
    target_threshold_status: Literal["pass", "fail", "not_evaluated"]
    guardrails: dict[str, MetricDelta]
    ordinary_cost_status: Literal["known", "estimate", "unknown"]
    targeted_cost_status: Literal["known", "estimate", "unknown"]
    ordinary_estimated_cost_usd: float | None = Field(default=None, ge=0)
    targeted_estimated_cost_usd: float | None = Field(default=None, ge=0)
    pretreatment_model_used_as_control: Literal[False] = False


def compare_matched_budget_arms(
    *,
    ordinary: MatchedBudgetArm,
    targeted: MatchedBudgetArm,
    target: MetricTarget,
    guardrails: list[Guardrail],
    ordinary_cost_status: Literal["known", "estimate", "unknown"] = "unknown",
    targeted_cost_status: Literal["known", "estimate", "unknown"] = "unknown",
    ordinary_estimated_cost_usd: float | None = None,
    targeted_estimated_cost_usd: float | None = None,
) -> MatchedBudgetReport:
    reasons: list[str] = []
    if ordinary.arm != "ordinary" or targeted.arm != "targeted":
        reasons.append("arm labels do not identify ordinary control and targeted treatment")
    if ordinary.parent_model_id != targeted.parent_model_id:
        reasons.append("starting parent checkpoints differ")
    if ordinary.trained_model_id == ordinary.parent_model_id or targeted.trained_model_id == targeted.parent_model_id:
        reasons.append("both arms must contain an actually trained model, not a pretreatment checkpoint")
    if ordinary.actual_training_budget != targeted.actual_training_budget:
        reasons.append("actual training budgets differ")
    if ordinary.budget_unit != targeted.budget_unit:
        reasons.append("training budget units differ")
    if ordinary.training_config_without_dataset_sha256 != targeted.training_config_without_dataset_sha256:
        reasons.append("non-data training recipe/configuration differs")
    if ordinary.training_dataset_sha256 == targeted.training_dataset_sha256:
        reasons.append("ordinary and targeted training data content is identical")
    if (ordinary.eval_dataset_id, ordinary.eval_dataset_sha256, ordinary.eval_item_set_sha256,
            ordinary.eval_protocol_version, ordinary.eval_config_sha256) != (
            targeted.eval_dataset_id, targeted.eval_dataset_sha256, targeted.eval_item_set_sha256,
            targeted.eval_protocol_version, targeted.eval_config_sha256):
        reasons.append("evaluation dataset, question set, or protocol differs")
    if target.name not in ordinary.metrics or target.name not in targeted.metrics:
        reasons.append(f"target metric {target.name!r} is missing from one arm")

    comparable = not reasons
    target_delta = _delta(
        ordinary.metrics.get(target.name) if comparable else None,
        targeted.metrics.get(target.name) if comparable else None,
        target.direction,
    )
    target_value = targeted.metrics.get(target.name) if comparable else None
    if target_value is None:
        threshold_status = "not_evaluated"
    else:
        passes = target_value >= target.threshold if target.direction == "higher" else target_value <= target.threshold
        threshold_status = "pass" if passes else "fail"
    guard_delta: dict[str, MetricDelta] = {}
    for guard in guardrails:
        before_value = ordinary.metrics.get(guard.name) if comparable else None
        after_value = targeted.metrics.get(guard.name) if comparable else None
        if before_value is None or after_value is None:
            if comparable:
                reasons.append(f"guardrail metric {guard.name!r} is missing from one arm")
            guard_delta[guard.name] = _delta(None, None, guard.direction)
        else:
            guard_delta[guard.name] = _delta(before_value, after_value, guard.direction, guard.max_regression)
    if reasons:
        comparable = False
        target_delta = _delta(None, None, target.direction)
        threshold_status = "not_evaluated"
        guard_delta = {guard.name: _delta(None, None, guard.direction) for guard in guardrails}
        status = "not_comparable"
    elif any(metric.regressed for metric in guard_delta.values()):
        status = "guardrail_regression"
    else:
        delta = target_delta.delta
        improved = delta > 0 if target.direction == "higher" else delta < 0
        status = "improved" if improved and threshold_status == "pass" else "no_improvement"

    report_identity = {
        "ordinary": ordinary.model_dump(mode="json"),
        "targeted": targeted.model_dump(mode="json"),
        "target": target.model_dump(mode="json"),
        "guardrails": [item.model_dump(mode="json") for item in guardrails],
        "status": status,
        "reasons": reasons,
    }
    return MatchedBudgetReport(
        report_id=f"matched-budget-{sha256_json(report_identity)[:20]}",
        status=status,
        comparable=comparable,
        reasons_not_comparable=reasons,
        ordinary_model_id=ordinary.trained_model_id,
        targeted_model_id=targeted.trained_model_id,
        shared_parent_model_id=ordinary.parent_model_id,
        ordinary_dataset_id=ordinary.training_dataset_id,
        targeted_dataset_id=targeted.training_dataset_id,
        matched_training_budget=ordinary.actual_training_budget if comparable else None,
        budget_unit=ordinary.budget_unit if comparable else None,
        target_metric=target.name,
        target_delta=target_delta,
        target_threshold_status=threshold_status,
        guardrails=guard_delta,
        ordinary_cost_status=ordinary_cost_status,
        targeted_cost_status=targeted_cost_status,
        ordinary_estimated_cost_usd=ordinary_estimated_cost_usd,
        targeted_estimated_cost_usd=targeted_estimated_cost_usd,
    )
