from __future__ import annotations

from onegpu_posttrain.data.targeted import MetricTarget
from onegpu_posttrain.synthetic.effects import Guardrail
from onegpu_posttrain.synthetic.matched_budget import (
    MatchedBudgetArm,
    compare_matched_budget_arms,
)

SHA = "a" * 64


def _arm(arm, *, model, data_hash, accuracy, budget=100):
    return MatchedBudgetArm(
        arm=arm,
        parent_model_id="model-sft-start",
        trained_model_id=model,
        training_dataset_id=f"dataset-{arm}",
        training_dataset_sha256=data_hash,
        training_config_without_dataset_sha256="b" * 64,
        actual_training_budget=budget,
        budget_unit="tokens",
        eval_id=f"eval-{arm}",
        eval_dataset_id="fixed-diagnostic-v1",
        eval_dataset_sha256="c" * 64,
        eval_item_set_sha256="d" * 64,
        eval_protocol_version="math-v1",
        eval_config_sha256="e" * 64,
        metrics={"accuracy": accuracy, "holdout_accuracy": 0.8},
    )


def _compare(ordinary, targeted):
    return compare_matched_budget_arms(
        ordinary=ordinary,
        targeted=targeted,
        target=MetricTarget(name="accuracy", direction="higher", threshold=0.7,
                            threshold_source="fixture", minimum_samples=2),
        guardrails=[Guardrail(name="holdout_accuracy", direction="higher", max_regression=0.05)],
    )


def test_matched_start_budget_protocol_and_data_condition_show_comparable_delta():
    report = _compare(
        _arm("ordinary", model="model-ordinary", data_hash=SHA, accuracy=0.5),
        _arm("targeted", model="model-targeted", data_hash="f" * 64, accuracy=0.75),
    )
    assert report.comparable is True
    assert report.status == "improved"
    assert report.target_delta.delta == 0.25
    assert report.matched_training_budget == 100
    assert report.pretreatment_model_used_as_control is False


def test_changed_parent_budget_config_or_eval_set_are_not_comparable():
    ordinary = _arm("ordinary", model="model-ordinary", data_hash=SHA, accuracy=0.5)
    targeted = _arm("targeted", model="model-targeted", data_hash="f" * 64, accuracy=0.75)
    variants = [
        targeted.model_copy(update={"parent_model_id": "other-start"}),
        targeted.model_copy(update={"actual_training_budget": 99}),
        targeted.model_copy(update={"training_config_without_dataset_sha256": "1" * 64}),
        targeted.model_copy(update={"eval_item_set_sha256": "2" * 64}),
    ]
    for variant in variants:
        report = _compare(ordinary, variant)
        assert report.status == "not_comparable"
        assert report.target_delta.delta is None
        assert report.reasons_not_comparable


def test_pretreatment_checkpoint_cannot_fake_a_matched_budget_control():
    ordinary = _arm("ordinary", model="model-sft-start", data_hash=SHA, accuracy=0.5)
    targeted = _arm("targeted", model="model-targeted", data_hash="f" * 64, accuracy=0.75)
    report = _compare(ordinary, targeted)
    assert report.status == "not_comparable"
    assert any("actually trained model" in reason for reason in report.reasons_not_comparable)


def test_guardrail_regression_overrides_target_gain_and_cost_stays_unknown():
    ordinary = _arm("ordinary", model="model-ordinary", data_hash=SHA, accuracy=0.5)
    targeted = _arm("targeted", model="model-targeted", data_hash="f" * 64, accuracy=0.75).model_copy(
        update={"metrics": {"accuracy": 0.75, "holdout_accuracy": 0.6}},
    )
    report = _compare(ordinary, targeted)
    assert report.status == "guardrail_regression"
    assert report.targeted_cost_status == "unknown"
