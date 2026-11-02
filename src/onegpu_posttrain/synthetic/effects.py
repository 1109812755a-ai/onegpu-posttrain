"""Evidence-bounded before/after reporting for targeted data experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..data.targeted import MetricTarget, ScenarioReport
from ..lineage import sha256_json


class Guardrail(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    direction: Literal["higher", "lower"]
    max_regression: float = Field(default=0, ge=0, allow_inf_nan=False)


class MetricDelta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    before: float | None = None
    after: float | None = None
    delta: float | None = None
    direction: Literal["higher", "lower"]
    status: Literal["measured", "not_evaluated"]
    regressed: bool | None = None
    regression_amount: float | None = None


class TrainingResourceEvidence(BaseModel):
    """Observed SFT budget and hardware fields; missing telemetry stays unknown."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    actual_training_records: int | None = Field(default=None, ge=0)
    actual_training_tokens: int | None = Field(default=None, ge=0)
    wall_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    peak_gpu_memory_mib: int | None = Field(default=None, ge=0)


class TargetedEffectReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    report_id: str
    status: Literal[
        "not_executed", "not_comparable", "over_budget", "guardrail_regression",
        "improved", "no_improvement",
    ]
    before_eval_id: str
    after_eval_id: str | None
    before_model_id: str
    after_model_id: str | None
    dataset_id: str
    dataset_sha256: str
    protocol_version: str
    scenario_id: str
    target_metric: str
    target_threshold: float
    target_threshold_status: Literal["pass", "fail", "not_evaluated"]
    target: MetricDelta
    guardrails: dict[str, MetricDelta]
    training_dataset_id: str
    training_dataset_sha256: str
    targeted_plan_id: str | None = Field(default=None, pattern=r"^plan-targeted-[0-9a-f]{20}$")
    targeted_plan_version: int | None = Field(default=None, ge=1)
    training_run_id: str | None = None
    training_resources: TrainingResourceEvidence = Field(default_factory=TrainingResourceEvidence)
    actual_added_records: int = Field(ge=0)
    planned_training_budget: int = Field(ge=1)
    actual_training_budget: int = Field(ge=0)
    budget_status: Literal["within_budget", "over_budget", "not_executed"]
    cost_status: Literal["known", "estimate", "unknown"]
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    comparison_limitations: list[str] = Field(default_factory=list)
    automatic_next_round_started: Literal[False] = False

    @model_validator(mode="after")
    def validate_targeted_plan_identity(self):
        if (self.targeted_plan_id is None) != (self.targeted_plan_version is None):
            raise ValueError("targeted plan ID and version must be supplied together")
        return self


def save_effect_report(report: TargetedEffectReport, path: str | Path) -> None:
    """Persist immutable effect evidence without replacing a prior report."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report.model_dump(mode="json"), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _scenario(report: ScenarioReport, scenario_id: str) -> tuple[float | None, int | None]:
    item = next((row for row in report.scenarios if row.scenario_id == scenario_id), None)
    if item is None or item.denominator is None or item.denominator == 0:
        return None, item.denominator if item else None
    return item.accuracy, item.denominator


def _delta(before: float | None, after: float | None, direction: str, tolerance: float = 0) -> MetricDelta:
    if before is None or after is None:
        return MetricDelta(before=before, after=after, direction=direction, status="not_evaluated")
    amount = before - after if direction == "higher" else after - before
    regressed = amount > tolerance
    return MetricDelta(before=before, after=after, delta=after - before, direction=direction,
                       status="measured", regressed=regressed, regression_amount=max(0.0, amount))


def build_before_after_report(
    *,
    before: ScenarioReport,
    after: ScenarioReport | None,
    target_scenario_id: str,
    target: MetricTarget,
    guardrails: list[Guardrail],
    before_guard_metrics: dict[str, float],
    after_guard_metrics: dict[str, float] | None,
    training_dataset_id: str,
    training_dataset_sha256: str,
    actual_added_records: int,
    planned_training_budget: int,
    actual_training_budget: int | None,
    cost_status: Literal["known", "estimate", "unknown"],
    estimated_cost_usd: float | None = None,
    targeted_plan_id: str | None = None,
    targeted_plan_version: int | None = None,
    training_run_id: str | None = None,
    training_resources: TrainingResourceEvidence | dict | None = None,
) -> TargetedEffectReport:
    """Compare only identical evaluation data/protocol; preserve missing outcomes."""
    before_value, before_denominator = _scenario(before, target_scenario_id)
    base = {
        "before_eval_id": before.eval_id,
        "before_model_id": before.model_id,
        "dataset_id": before.dataset_id,
        "dataset_sha256": before.dataset_sha256,
        "protocol_version": before.protocol_version,
        "scenario_id": target_scenario_id,
        "target_metric": target.name,
        "target_threshold": target.threshold,
        "target_threshold_status": "not_evaluated",
        "target": _delta(before_value, None, target.direction),
        "guardrails": {
            item.name: _delta(before_guard_metrics.get(item.name),
                              None if after_guard_metrics is None else after_guard_metrics.get(item.name),
                              item.direction, item.max_regression)
            for item in guardrails
        },
        "training_dataset_id": training_dataset_id,
        "training_dataset_sha256": training_dataset_sha256,
        "targeted_plan_id": targeted_plan_id,
        "targeted_plan_version": targeted_plan_version,
        "training_run_id": training_run_id,
        "training_resources": (training_resources.model_dump(mode="json")
                               if isinstance(training_resources, TrainingResourceEvidence)
                               else training_resources or {}),
        "actual_added_records": actual_added_records,
        "planned_training_budget": planned_training_budget,
        "actual_training_budget": actual_training_budget or 0,
        "budget_status": "not_executed" if actual_training_budget is None else (
            "within_budget" if actual_training_budget <= planned_training_budget else "over_budget"
        ),
        "cost_status": cost_status,
        "estimated_cost_usd": estimated_cost_usd,
        "comparison_limitations": [],
    }
    if after is None or actual_training_budget is None:
        base["comparison_limitations"].append("定向训练或复测尚未执行；不推断收益")
        status = "not_executed"
        after_eval_id = None
        after_model_id = None
    else:
        after_value, after_denominator = _scenario(after, target_scenario_id)
        same_evaluation = (
            before.dataset_id == after.dataset_id
            and before.dataset_sha256 == after.dataset_sha256
            and before.protocol_version == after.protocol_version
            and before.scenario_definition_id == after.scenario_definition_id
            and before.scenario_definition_sha256 == after.scenario_definition_sha256
        )
        if not same_evaluation:
            status = "not_comparable"
            base["comparison_limitations"].append("前后评测数据或协议身份不同，指标差异不可比较")
            base["target"] = _delta(None, None, target.direction)
            base["guardrails"] = {item.name: _delta(None, None, item.direction) for item in guardrails}
        elif before_value is None or after_value is None:
            status = "not_comparable"
            base["comparison_limitations"].append("目标场景缺少有效分母，不能计算前后变化")
            base["target"] = _delta(before_value, after_value, target.direction)
            base["guardrails"] = {item.name: _delta(
                before_guard_metrics.get(item.name),
                None if after_guard_metrics is None else after_guard_metrics.get(item.name), item.direction,
            ) for item in guardrails}
        else:
            base["target"] = _delta(before_value, after_value, target.direction)
            threshold_pass = after_value >= target.threshold if target.direction == "higher" else after_value <= target.threshold
            base["target_threshold_status"] = "pass" if threshold_pass else "fail"
            base["guardrails"] = {
                item.name: _delta(before_guard_metrics.get(item.name),
                                  None if after_guard_metrics is None else after_guard_metrics.get(item.name),
                                  item.direction, item.max_regression)
                for item in guardrails
            }
            guard_regression = any(metric.regressed for metric in base["guardrails"].values())
            target_delta = base["target"].delta
            improved_in_direction = target_delta > 0 if target.direction == "higher" else target_delta < 0
            if base["budget_status"] == "over_budget":
                status = "over_budget"
                base["comparison_limitations"].append("实际训练预算超过登记预算；不将结果标成预算内收益")
            elif guard_regression:
                status = "guardrail_regression"
            elif improved_in_direction and threshold_pass:
                status = "improved"
            else:
                status = "no_improvement"
            if before_denominator != after_denominator:
                base["comparison_limitations"].append("前后场景分母不同；请将样本量差异纳入解释")
        after_eval_id = after.eval_id
        after_model_id = after.model_id

    identity = {
        **{key: value.model_dump(mode="json") if isinstance(value, MetricDelta) else {
            name: metric.model_dump(mode="json") for name, metric in value.items()
        } if key in {"target", "guardrails"} else value for key, value in base.items()},
        "status": status,
        "after_eval_id": after_eval_id,
        "after_model_id": after_model_id,
    }
    return TargetedEffectReport(
        report_id=f"targeted-effect-{sha256_json(identity)[:20]}",
        status=status,
        after_eval_id=after_eval_id,
        after_model_id=after_model_id,
        **base,
    )
