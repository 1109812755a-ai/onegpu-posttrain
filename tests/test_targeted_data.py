from __future__ import annotations

import pytest
from pydantic import ValidationError

from onegpu_posttrain.data.targeted import (
    DataIdentity,
    EvaluationContract,
    EvaluationIdentity,
    GenerationBudget,
    MetricTarget,
    ModelIdentity,
    ScenarioDefinition,
    ScenarioDiagnosis,
    ScenarioLabel,
    ScenarioReport,
    TargetedGenerationPlan,
)

SHA = "a" * 64


def scenario_definition(**updates):
    values = {
        "taxonomy_id": "math-skills",
        "taxonomy_version": "v1",
        "dataset_id": "math-dev-v1",
        "dataset_sha256": SHA,
        "label_source": "metadata",
        "label_field": "topic",
        "scenarios": [ScenarioLabel(scenario_id="algebra", title="代数", values=["algebra"])],
    }
    values.update(updates)
    return ScenarioDefinition(**values)


def diagnosis(scenario_id="algebra", *, samples=4, denominator=4, correct=2, incorrect=2,
              parse_failed=0, unsupported=0, excluded=0, accuracy=0.5, status="available"):
    return ScenarioDiagnosis(
        scenario_id=scenario_id, sample_count=samples, denominator=denominator,
        correct=correct, incorrect=incorrect, parse_failed=parse_failed,
        unsupported=unsupported, excluded=excluded, accuracy=accuracy, status=status,
    )


def generation_plan(**updates):
    values = {
        "scenario_report_id": "scenario-report-" + "b" * 20,
        "scenario_report_sha256": SHA,
        "scenario_definition_id": "scenario-def-" + "c" * 20,
        "scenario_definition_sha256": SHA,
        "role_registry_sha256": SHA,
        "target_scenario_id": "algebra",
        "target": MetricTarget(name="accuracy", direction="higher", threshold=0.8,
                               threshold_source="project experiment hypothesis", minimum_samples=4),
        "diagnostic": EvaluationIdentity(eval_id="eval-dev", model_id="model-sft", dataset_id="math-dev-v1",
                                         dataset_sha256=SHA, protocol_version="internal-math-v1",
                                         config_sha256=SHA, role="diagnostic_dev"),
        "training_seed": DataIdentity(dataset_id="math-train-v1", content_sha256=SHA, role="train_seed"),
        "starting_model": ModelIdentity(model_id="model-sft", content_sha256=SHA),
        "final_evaluation": EvaluationContract(dataset_id="math-holdout-v1", dataset_sha256=SHA,
                                               protocol_version="internal-math-v1", config_sha256=SHA,
                                               role="final_holdout"),
        "stage": "sft",
        "comparison_design": "before_after_pipeline",
        "provider_profile_id": "local-fake",
        "prompt_template_sha256": SHA,
        "budget": GenerationBudget(max_requests=10, max_output_tokens=512, target_candidates=5),
        "training_budget": 100,
        "seed": 42,
    }
    values.update(updates)
    return TargetedGenerationPlan(**values)


def test_scenario_definition_identity_changes_with_label_contract():
    original = scenario_definition()
    changed = scenario_definition(taxonomy_version="v2")
    assert original.definition_id != changed.definition_id
    assert original.config_sha256 != changed.config_sha256


def test_targeted_plan_identity_changes_with_generation_budget():
    original = generation_plan()
    changed = generation_plan(budget=GenerationBudget(max_requests=11, max_output_tokens=512, target_candidates=5))
    assert original.plan_id != changed.plan_id
    assert original.config_sha256 != changed.config_sha256


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        scenario_definition(unreviewed_rule="ignore correctness")


def test_empty_scenario_is_not_evaluated_not_zero_accuracy():
    row = diagnosis(samples=0, denominator=0, correct=0, incorrect=0, accuracy=None, status="not_evaluated")
    report = ScenarioReport(
        eval_id="eval-dev", model_id="model-sft", dataset_id="math-dev-v1", dataset_sha256=SHA,
        protocol_version="internal-math-v1", predictions_sha256=SHA,
        scenario_definition_id="scenario-def-" + "c" * 20, scenario_definition_sha256=SHA,
        label_source_sha256=SHA, threshold_source="project experiment hypothesis", minimum_samples_required=4,
        input_record_count=0, prediction_count=0, scenarios=[row],
    )
    assert report.scenarios[0].accuracy is None
    assert report.scenarios[0].status == "not_evaluated"
    assert report.report_id == report.model_copy().report_id


def test_nonempty_scenario_requires_denominator():
    with pytest.raises(ValidationError, match="denominator"):
        diagnosis(denominator=None)


def test_missing_denominator_cannot_be_labeled_insufficient_or_available():
    with pytest.raises(ValidationError, match="denominator"):
        diagnosis(samples=0, denominator=None, correct=0, incorrect=0, accuracy=None,
                  status="insufficient_evidence")


def test_accuracy_must_be_computed_from_denominator():
    with pytest.raises(ValidationError, match="correct / denominator"):
        diagnosis(accuracy=0.75)


def test_low_denominator_is_reported_as_insufficient_evidence():
    row = diagnosis(samples=2, denominator=2, correct=1, incorrect=1, accuracy=0.5,
                    status="insufficient_evidence")
    ScenarioReport(
        eval_id="eval-dev", model_id="model-sft", dataset_id="math-dev-v1", dataset_sha256=SHA,
        protocol_version="internal-math-v1", predictions_sha256=SHA,
        scenario_definition_id="scenario-def-" + "c" * 20, scenario_definition_sha256=SHA,
        label_source_sha256=SHA, threshold_source="project experiment hypothesis", minimum_samples_required=4,
        input_record_count=2, prediction_count=2, scenarios=[row],
    )


def test_plan_rejects_role_leakage_and_model_mismatch():
    with pytest.raises(ValidationError, match="train_seed"):
        generation_plan(training_seed=DataIdentity(dataset_id="math-dev-v1", content_sha256=SHA, role="diagnostic_dev"))
    with pytest.raises(ValidationError):
        generation_plan(starting_model=ModelIdentity(model_id="another-model", content_sha256=SHA))
    with pytest.raises(ValidationError, match="分离"):
        generation_plan(final_evaluation=EvaluationContract(dataset_id="math-dev-v1", dataset_sha256=SHA,
                                                            protocol_version="internal-math-v1",
                                                            config_sha256=SHA, role="final_holdout"))

