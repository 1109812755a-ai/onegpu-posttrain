from __future__ import annotations

import pytest
from pydantic import ValidationError

from onegpu_posttrain.data.targeted import (
    MetricTarget,
    ScenarioDiagnosis,
    ScenarioReport,
)
from onegpu_posttrain.synthetic.effects import (
    Guardrail,
    TargetedEffectReport,
    TrainingResourceEvidence,
    build_before_after_report,
    save_effect_report,
)

SHA = "a" * 64
DEF_ID = "scenario-def-" + "b" * 20


def _report(eval_id, model_id, *, correct, dataset_id="diag-v1", protocol="math-v1", guard=None):
    return ScenarioReport(
        eval_id=eval_id,
        model_id=model_id,
        dataset_id=dataset_id,
        dataset_sha256=SHA,
        protocol_version=protocol,
        predictions_sha256="c" * 64,
        scenario_definition_id=DEF_ID,
        scenario_definition_sha256="d" * 64,
        label_source_sha256="e" * 64,
        threshold_source="fixture only",
        minimum_samples_required=2,
        input_record_count=4,
        prediction_count=4,
        scenarios=[ScenarioDiagnosis(
            scenario_id="algebra", sample_count=4, denominator=4, correct=correct,
            incorrect=4 - correct, accuracy=correct / 4, status="available",
        )],
    )


def _build(before, after, *, before_guard=None, after_guard=None, actual_budget: int | None = 5):
    return build_before_after_report(
        before=before,
        after=after,
        target_scenario_id="algebra",
        target=MetricTarget(name="accuracy", direction="higher", threshold=0.7,
                            threshold_source="fixture only", minimum_samples=2),
        guardrails=[Guardrail(name="holdout_accuracy", direction="higher", max_regression=0.05)],
        before_guard_metrics={"holdout_accuracy": 0.8 if before_guard is None else before_guard},
        after_guard_metrics=None if after is None else {
            "holdout_accuracy": 0.8 if after_guard is None else after_guard,
        },
        training_dataset_id="generated-sft-v1",
        training_dataset_sha256="f" * 64,
        actual_added_records=12,
        planned_training_budget=5,
        actual_training_budget=actual_budget,
        cost_status="unknown",
    )


def test_missing_execution_remains_not_executed_without_invented_delta(tmp_path):
    report = _build(_report("eval-before", "model-before", correct=2), None, actual_budget=None)
    assert report.status == "not_executed"
    assert report.target.delta is None
    assert report.target_threshold_status == "not_evaluated"
    assert report.automatic_next_round_started is False
    assert report.cost_status == "unknown"
    output = tmp_path / "effects" / "not-executed.json"
    save_effect_report(report, output)
    assert '"status": "not_executed"' in output.read_text(encoding="utf-8")


def test_same_protocol_improvement_reports_target_guard_and_actual_budgets():
    report = _build(
        _report("eval-before", "model-before", correct=2),
        _report("eval-after", "model-after", correct=3),
    )
    assert report.status == "improved"
    assert report.target.delta == 0.25
    assert report.target_threshold_status == "pass"
    assert report.actual_added_records == 12
    assert report.actual_training_budget == 5
    assert report.budget_status == "within_budget"


def test_guardrail_regression_is_preserved_even_when_target_improves():
    report = _build(
        _report("eval-before", "model-before", correct=2),
        _report("eval-after", "model-after", correct=4),
        after_guard=0.7,
    )
    assert report.status == "guardrail_regression"
    assert report.guardrails["holdout_accuracy"].regressed is True


def test_changed_eval_identity_is_not_comparable():
    report = _build(
        _report("eval-before", "model-before", correct=2),
        _report("eval-after", "model-after", correct=4, dataset_id="other-dataset"),
    )
    assert report.status == "not_comparable"
    assert report.target.delta is None


def test_over_budget_run_is_not_labeled_as_budgeted_improvement():
    report = _build(
        _report("eval-before", "model-before", correct=2),
        _report("eval-after", "model-after", correct=4),
        actual_budget=8,
    )
    assert report.status == "over_budget"
    assert report.budget_status == "over_budget"


def test_report_retains_targeted_plan_run_and_observed_training_resources():
    plan_id = "plan-targeted-0123456789abcdefabcd"
    report = build_before_after_report(
        before=_report("eval-before", "model-before", correct=2),
        after=_report("eval-after", "model-after", correct=3),
        target_scenario_id="algebra",
        target=MetricTarget(name="accuracy", direction="higher", threshold=0.7,
                            threshold_source="fixture only", minimum_samples=2),
        guardrails=[], before_guard_metrics={}, after_guard_metrics={},
        training_dataset_id="ds-exact-registered-sft", training_dataset_sha256="f" * 64,
        actual_added_records=3, planned_training_budget=5, actual_training_budget=3,
        cost_status="unknown", targeted_plan_id=plan_id, targeted_plan_version=2,
        training_run_id="run-targeted-sft-offline-001",
        training_resources=TrainingResourceEvidence(actual_training_records=3, actual_training_tokens=91,
                                                     wall_seconds=12.5, peak_gpu_memory_mib=4096),
    )
    assert report.training_dataset_id == "ds-exact-registered-sft"
    assert report.targeted_plan_id == plan_id and report.targeted_plan_version == 2
    assert report.training_run_id == "run-targeted-sft-offline-001"
    assert report.training_resources.actual_training_records == 3
    assert report.training_resources.actual_training_tokens == 91
    assert report.training_resources.peak_gpu_memory_mib == 4096


def test_report_rejects_unpaired_targeted_plan_identity():
    report = _build(_report("eval-before", "model-before", correct=2), None)
    with pytest.raises(ValidationError, match="ID and version"):
        TargetedEffectReport.model_validate(
            report.model_dump() | {"targeted_plan_id": "plan-targeted-0123456789abcdefabcd"}
        )
