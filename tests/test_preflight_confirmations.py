import pytest

from onegpu_posttrain.ui.preflight_confirmations import StagePreflightConfirmations
from onegpu_posttrain.ui.training_preflight import StagePreflightRequest
from onegpu_posttrain.ui.training_validation import TrainingValidationConfig


def _request(dataset_id="ds-sft-1"):
    return StagePreflightRequest(stage="sft", profile_id="sft-mini-1024-v1",
                                 dataset_ids=[dataset_id], parent_model_id="model-cpt-1")


def _summary(*, content_hash="a" * 64):
    return {
        "stage": "sft", "profile_id": "sft-mini-1024-v1", "algorithm": "sft",
        "status": "ready_for_confirmation", "approved_recipe": {"path": "recipes/mini/23-sft-1024-fallback.yaml",
                                                                    "sha256": "b" * 64},
        "datasets": [{"dataset_id": "ds-sft-1", "manifest_sha256": "c" * 64,
                       "content_sha256": content_hash}],
        "parent_model": {"model_id": "model-cpt-1", "digest": "d" * 64},
        "training_parameters": {"config_sha256": "e" * 64},
        "checks": [], "blockers": [], "warnings": [],
    }


def test_confirmation_is_bound_and_consumed_once():
    confirmations = StagePreflightConfirmations()
    request = _request()
    issued = confirmations.issue(request, _summary())

    accepted = confirmations.consume(issued["confirmation_token"], request, _summary())
    assert accepted["output_location"].startswith("runs/sft/sft-mini-1024-v1/run-")
    assert "confirmation_token" not in accepted
    with pytest.raises(ValueError, match="不存在|已使用"):
        confirmations.consume(issued["confirmation_token"], request, _summary())


def test_confirmation_rejects_request_or_evidence_tampering():
    confirmations = StagePreflightConfirmations()
    request = _request()
    issued = confirmations.issue(request, _summary())

    with pytest.raises(ValueError, match="请求与预检"):
        confirmations.consume(issued["confirmation_token"], _request("ds-other"), _summary())

    issued = confirmations.issue(request, _summary())
    with pytest.raises(ValueError, match="摘要或预检结果已变化"):
        confirmations.consume(issued["confirmation_token"], request, _summary(content_hash="e" * 64))

    issued = confirmations.issue(request, _summary())
    changed_config = _summary()
    changed_config["training_parameters"]["config_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="摘要或预检结果已变化"):
        confirmations.consume(issued["confirmation_token"], request, changed_config)


def test_confirmation_binds_parameter_overrides():
    confirmations = StagePreflightConfirmations()
    request = _request().model_copy(update={"parameter_overrides": {"max_length": 768}})
    issued = confirmations.issue(request, _summary())
    with pytest.raises(ValueError, match="階段|阶段|profile|数据集|请求"):
        confirmations.consume(issued["confirmation_token"],
                              request.model_copy(update={"parameter_overrides": {"max_length": 1024}}),
                              _summary())


def test_confirmation_binds_validation_mode_ratio_seed_and_dataset():
    confirmations = StagePreflightConfirmations()
    request = _request().model_copy(update={
        "validation": TrainingValidationConfig(mode="random_split", ratio=0.1, seed=42),
    })
    issued = confirmations.issue(request, _summary())
    changed = request.model_copy(update={
        "validation": TrainingValidationConfig(mode="random_split", ratio=0.2, seed=42),
    })
    with pytest.raises(ValueError, match="请求与预检"):
        confirmations.consume(issued["confirmation_token"], changed, _summary())


def test_confirmation_expiration_is_enforced():
    now = [10.0]
    confirmations = StagePreflightConfirmations(ttl_seconds=5, clock=lambda: now[0])
    request = _request()
    issued = confirmations.issue(request, _summary())
    now[0] = 15.0

    with pytest.raises(ValueError, match="已过期"):
        confirmations.consume(issued["confirmation_token"], request, _summary())
