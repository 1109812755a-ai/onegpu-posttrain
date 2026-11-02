from __future__ import annotations

import pytest

from onegpu_posttrain.data.targeted import DataIdentity, EvaluationContract, MetricTarget, ModelIdentity
from onegpu_posttrain.data.targeted_stage_plan import TargetedStagePlan
from onegpu_posttrain.lineage import ArtifactRef, EvaluationLineage, RunRecord
from onegpu_posttrain.synthetic.stage_effects import build_stage_effect_report


SHA = "a" * 64


def _plan(*, target_metric=True):
    target = {"stage": "grpo", "scenario_id": "algebra"}
    if target_metric:
        target["target_metric"] = {
            "name": "gsm8k.accuracy", "direction": "higher", "threshold": 0.5,
            "threshold_source": "pilot acceptance target", "minimum_samples": 5,
        }
        target["guardrail_metrics"] = [{
            "name": "math_500.accuracy", "direction": "higher", "threshold": 0.2,
            "threshold_source": "baseline floor", "minimum_samples": 5,
        }]
    return TargetedStagePlan.model_validate({
        "target": target,
        "training_seed": {"dataset_id": "seed", "content_sha256": SHA, "role": "train_seed"},
        "starting_model": {"model_id": "model-start", "content_sha256": SHA},
        "diagnostic_evaluation": {"dataset_id": "benchmark-gsm8k", "dataset_sha256": "d" * 64,
                                  "protocol_version": "math-v1", "config_sha256": "e" * 64,
                                  "role": "diagnostic_dev"},
        "final_evaluation": {"dataset_id": "final-holdout", "dataset_sha256": "b" * 64,
                             "protocol_version": "math-v1", "config_sha256": "c" * 64,
                             "role": "final_holdout"},
        "provider_profile_id": "local", "budget": {"unit": "prompts", "expected_usable_count": 5,
            "max_candidates": 8, "max_requests": 4, "max_output_tokens": 512},
        "training_budget": 10, "seed": 42,
    })


def _evaluation(plan, *, eval_id, model_id, accuracy, math_accuracy=0.4, config="e" * 64):
    return EvaluationLineage(
        eval_id=eval_id, produced_by_run_id=("run-grpo-effect-test" if model_id == "model-grpo"
                                            else f"run-{eval_id.removeprefix('eval-')}"),
        model_produced_by_run_id="run-grpo-effect-test" if model_id == "model-grpo" else None,
        model_id=model_id, dataset_ids=["benchmark-gsm8k"], protocol_version="math-v1",
        config_sha256=config, targeted_plan_id=plan.plan_id if model_id != "model-start" else None,
        targeted_plan_version=plan.version if model_id != "model-start" else None,
        status="completed", metrics={
            "evalscope.gsm8k.accuracy": {"value": accuracy, "denominator": 10},
            "evalscope.math_500.accuracy": {"value": math_accuracy, "denominator": 10},
        },
    )


def _training_run(plan):
    return RunRecord(
        run_id="run-grpo-effect-test", run_type="train", status="completed",
        started_at="2026-10-25T00:00:00Z", targeted_plan_id=plan.plan_id,
        targeted_plan_version=plan.version,
        inputs=[ArtifactRef(id="ds-grpo-train", kind="dataset", role="train",
                            digest={"value": SHA, "status": "available"})],
    )


def test_grpo_stage_effect_report_requires_matched_eval_and_uses_explicit_thresholds():
    plan = _plan()
    before = _evaluation(plan, eval_id="eval-before", model_id="model-start", accuracy=0.4)
    after = _evaluation(plan, eval_id="eval-after", model_id="model-grpo", accuracy=0.6, math_accuracy=0.3)
    report = build_stage_effect_report(plan=plan, training_run=_training_run(plan),
                                       model_id="model-grpo", before=before, after=after)
    assert report["status"] == "target_pass"
    assert report["target_metric"]["threshold_source"] == "pilot acceptance target"
    assert report["target_metric"]["delta"] == pytest.approx(0.2)
    assert report["guardrail_metrics"]["math_500.accuracy"]["threshold_status"] == "pass"
    assert report["training_datasets"] == [{"dataset_id": "ds-grpo-train", "content_sha256": SHA}]


def test_grpo_effect_report_does_not_infer_target_without_baseline_or_target_config():
    plan = _plan()
    after = _evaluation(plan, eval_id="eval-after", model_id="model-grpo", accuracy=0.9)
    no_baseline = build_stage_effect_report(plan=plan, training_run=_training_run(plan),
                                            model_id="model-grpo", before=None, after=after)
    assert no_baseline["status"] == "not_comparable"
    assert no_baseline["target_state"] == "not_comparable"

    unconfigured_plan = _plan(target_metric=False)
    after = _evaluation(unconfigured_plan, eval_id="eval-after", model_id="model-grpo", accuracy=0.9)
    no_target = build_stage_effect_report(
        plan=unconfigured_plan, training_run=_training_run(unconfigured_plan),
        model_id="model-grpo", before=_evaluation(unconfigured_plan, eval_id="eval-before",
                                                   model_id="model-start", accuracy=0.4), after=after,
    )
    assert no_target["target_state"] == "not_configured"
    assert "不判定达标" in no_target["comparison_limitations"][-1]


def test_grpo_effect_report_rejects_evaluation_from_another_plan():
    plan = _plan()
    after = _evaluation(plan, eval_id="eval-after", model_id="model-grpo", accuracy=0.6)
    after = after.model_copy(update={"targeted_plan_id": "plan-targeted-00000000000000000000"})
    with pytest.raises(ValueError, match="targeted plan identity"):
        build_stage_effect_report(plan=plan, training_run=_training_run(plan),
                                  model_id="model-grpo", before=None, after=after)


def test_grpo_effect_report_requires_completed_eval_and_plan_evaluation_contract():
    plan = _plan()
    training_run = _training_run(plan)
    after = _evaluation(plan, eval_id="eval-after", model_id="model-grpo", accuracy=0.6)
    incomplete = after.model_copy(update={"status": "failed"})
    with pytest.raises(ValueError, match="completed post-training evaluation"):
        build_stage_effect_report(plan=plan, training_run=training_run,
                                  model_id="model-grpo", before=None, after=incomplete)

    wrong_contract = after.model_copy(update={"config_sha256": "f" * 64})
    report = build_stage_effect_report(plan=plan, training_run=training_run,
                                       model_id="model-grpo", before=None, after=wrong_contract)
    assert report["status"] == "not_comparable"
    assert report["target_state"] == "not_comparable"
    assert any("训练后评测未匹配" in item for item in report["comparison_limitations"])


def test_grpo_effect_report_rejects_training_run_from_another_plan():
    plan = _plan()
    after = _evaluation(plan, eval_id="eval-after", model_id="model-grpo", accuracy=0.6)
    wrong_run = _training_run(plan).model_copy(update={"targeted_plan_version": plan.version + 1})
    with pytest.raises(ValueError, match="training Run does not match"):
        build_stage_effect_report(plan=plan, training_run=wrong_run,
                                  model_id="model-grpo", before=None, after=after)
