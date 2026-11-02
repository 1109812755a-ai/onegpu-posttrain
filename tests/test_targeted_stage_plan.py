from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from onegpu_posttrain.data.targeted import DataIdentity, EvaluationContract, ModelIdentity
from onegpu_posttrain.data.targeted_stage_plan import (
    TargetedStagePlan,
    load_targeted_stage_plan,
    save_targeted_stage_plan,
)


SHA = "a" * 64


def stage_plan(stage: str, **overrides) -> TargetedStagePlan:
    targets = {
        "cpt": {"stage": "cpt", "domain": "代数", "subtopics": ["方程"]},
        "sft": {"stage": "sft", "scenario_id": "algebra"},
        "preference": {"stage": "preference", "scenario_id": "algebra"},
        "grpo": {"stage": "grpo", "scenario_id": "algebra"},
    }
    units = {"cpt": "tokens", "sft": "records", "preference": "pairs", "grpo": "prompts"}
    value = {
        "target": targets[stage],
        "training_seed": {"dataset_id": "train-v1", "content_sha256": SHA, "role": "train_seed"},
        "starting_model": {"model_id": "base-model", "content_sha256": SHA},
        "diagnostic_evaluation": {
            "dataset_id": "diagnostic-v1", "dataset_sha256": "d" * 64,
            "protocol_version": "eval-v1", "config_sha256": "e" * 64,
            "role": "diagnostic_dev",
        },
        "final_evaluation": {
            "dataset_id": "holdout-v1", "dataset_sha256": "b" * 64,
            "protocol_version": "eval-v1", "config_sha256": "c" * 64,
            "role": "final_holdout",
        },
        "provider_profile_id": "local-profile",
        "budget": {
            "unit": units[stage], "expected_usable_count": 100,
            "max_candidates": 160, "max_requests": 80,
            "max_output_tokens": 512,
        },
        "training_budget": 20,
        "seed": 42,
    }
    value.update(overrides)
    return TargetedStagePlan.model_validate(value)


@pytest.mark.parametrize("stage", ["cpt", "sft", "preference", "grpo"])
def test_stage_plan_supports_four_distinct_target_contracts(stage):
    plan = stage_plan(stage)
    assert plan.target.stage == stage
    assert plan.plan_id.startswith("plan-targeted-")
    assert plan.plan_id == f"plan-targeted-{plan.config_sha256[:20]}"
    assert plan.model_dump(mode="json")["budget"]["expected_usable_count"] == 100


def test_stage_target_rejects_fields_from_another_stage():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        stage_plan("sft", target={"stage": "sft", "scenario_id": "algebra", "domain": "must-not-cross"})


def test_target_units_are_distinct_from_hard_generation_budgets():
    plan = stage_plan("preference")
    assert plan.budget.expected_usable_count == 100
    assert plan.budget.max_candidates == 160
    assert plan.budget.max_requests == 80
    with pytest.raises(ValidationError, match="目标单位"):
        stage_plan("preference", budget={
            "unit": "records", "expected_usable_count": 100,
            "max_candidates": 160, "max_requests": 80, "max_output_tokens": 512,
        })


def test_plan_rejects_non_training_seed_and_holdout_overlap():
    with pytest.raises(ValidationError, match="train_seed"):
        stage_plan("sft", training_seed={"dataset_id": "dev", "content_sha256": SHA, "role": "diagnostic_dev"})
    with pytest.raises(ValidationError, match="final_holdout"):
        stage_plan("sft", final_evaluation={
            "dataset_id": "train-v1", "dataset_sha256": SHA,
            "protocol_version": "eval-v1", "config_sha256": SHA,
            "role": "final_holdout",
        })


def test_plan_revision_is_immutable_and_identity_changes(tmp_path):
    revision1 = stage_plan("cpt", version=1)
    revision2 = stage_plan("cpt", version=2)
    assert revision1.plan_id != revision2.plan_id
    assert revision1.config_sha256 != revision2.config_sha256

    path1 = tmp_path / "plan-v1.json"
    path2 = tmp_path / "plan-v2.json"
    saved1 = save_targeted_stage_plan(revision1, path1)
    save_targeted_stage_plan(revision2, path2)
    assert saved1["plan_version"] == 1
    assert load_targeted_stage_plan(path1) == revision1
    assert load_targeted_stage_plan(path2) == revision2
    with pytest.raises(FileExistsError):
        save_targeted_stage_plan(revision2, path1)

    payload = json.loads(path1.read_text(encoding="utf-8"))
    payload["plan"]["target"]["domain"] = "篡改"
    path1.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="身份或摘要不匹配"):
        load_targeted_stage_plan(path1)


def test_cpt_plan_accepts_domain_coverage_targets_without_qa_fields():
    plan = stage_plan("cpt", target={
        "stage": "cpt", "domain": "数学", "subtopics": ["代数", "几何"],
        "coverage_targets": [
            {"subtopic": "代数", "minimum_token_share": 0.4},
            {"subtopic": "几何", "minimum_token_share": 0.2},
        ],
    })
    assert plan.target.coverage_targets[0].subtopic == "代数"
    assert "scenario_id" not in plan.target.model_dump()


def test_grpo_plan_keeps_target_metric_and_guardrail_thresholds_explicit():
    plan = stage_plan("grpo", target={
        "stage": "grpo", "scenario_id": "algebra",
        "target_metric": {"name": "gsm8k.accuracy", "direction": "higher", "threshold": 0.5,
                          "threshold_source": "用户确认的试点目标", "minimum_samples": 20},
        "guardrail_metrics": [{"name": "math_500.accuracy", "direction": "higher", "threshold": 0.2,
                               "threshold_source": "基线保护线", "minimum_samples": 10}],
    })
    assert plan.target.target_metric.threshold == 0.5
    assert plan.target.guardrail_metrics[0].threshold_source == "基线保护线"

    with pytest.raises(ValidationError, match="不能同时声明为 guardrail"):
        stage_plan("grpo", target={
            "stage": "grpo", "scenario_id": "algebra",
            "target_metric": {"name": "gsm8k.accuracy", "direction": "higher", "threshold": 0.5,
                              "threshold_source": "用户", "minimum_samples": 20},
            "guardrail_metrics": [{"name": "gsm8k.accuracy", "direction": "higher", "threshold": 0.2,
                                   "threshold_source": "用户", "minimum_samples": 20}],
        })


@pytest.mark.parametrize("coverage", [
    [{"subtopic": "未知", "minimum_token_share": 0.2}],
    [{"subtopic": "代数", "minimum_token_share": 0.8}, {"subtopic": "几何", "minimum_token_share": 0.4}],
    [{"subtopic": "代数", "minimum_token_share": 0.2}, {"subtopic": "代数", "minimum_token_share": 0.2}],
])
def test_cpt_plan_rejects_invalid_domain_coverage(coverage):
    with pytest.raises(ValidationError):
        stage_plan("cpt", target={
            "stage": "cpt", "domain": "数学", "subtopics": ["代数", "几何"],
            "coverage_targets": coverage,
        })
