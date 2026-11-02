from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from onegpu_posttrain.ui.targeted_workflow import (
    TargetedWorkflowState,
    WorkflowEvidence,
    load_workflow_state,
    save_workflow_state,
    workflow_state_path,
)


PLAN_ID = "plan-targeted-0123456789abcdefabcd"


def state(**updates):
    value = {
        "plan_id": PLAN_ID, "plan_version": 1, "stage": "sft",
        "starting_model_id": "model-start", "initial_dataset_ids": ("dataset-seed",),
    }
    value.update(updates)
    return TargetedWorkflowState.model_validate(value)


def test_workflow_snapshot_persists_partial_and_failed_steps_without_fabricated_outputs(tmp_path):
    original = state()
    with_partial = original.append(WorkflowEvidence(
        operation_id="gen-1", kind="generation", status="partial", execution_id="exec-1",
        artifact_ids=("candidate-bundle-1",), reason="request budget exhausted",
    ))
    with_failure = with_partial.append(WorkflowEvidence(
        operation_id="train-1", kind="training", status="failed", run_id="run-train-1",
        reason="OOM before checkpoint",
    ))
    path = workflow_state_path(tmp_path, PLAN_ID, 1)
    save_workflow_state(with_failure, path)
    restored = load_workflow_state(path)

    assert restored == with_failure
    assert original.evidence == ()
    assert restored.evidence[-1].model_id is None
    assert [item.status for item in restored.evidence] == ["partial", "failed"]


def test_successful_steps_require_real_output_ids():
    with pytest.raises(ValidationError, match="真实输出身份"):
        WorkflowEvidence(operation_id="train-1", kind="training", status="succeeded", run_id="run-1")
    with pytest.raises(ValidationError, match="真实输出身份"):
        WorkflowEvidence(operation_id="eval-1", kind="evaluation", status="succeeded", run_id="run-2")


def test_blocked_and_failed_steps_require_explanation():
    with pytest.raises(ValidationError, match="必须说明原因"):
        WorkflowEvidence(operation_id="grpo", kind="training", status="blocked")


def test_append_rejects_duplicate_operation_ids():
    initial = state().append(WorkflowEvidence(
        operation_id="quality-1", kind="quality", status="succeeded", run_id="run-quality-1",
    ))
    with pytest.raises(ValidationError, match="operation_id 必须唯一"):
        initial.append(WorkflowEvidence(
            operation_id="quality-1", kind="quality", status="succeeded", run_id="run-quality-2",
        ))


def test_state_digest_mismatch_and_unsafe_plan_ids_are_rejected(tmp_path):
    path = tmp_path / "state.json"
    save_workflow_state(state(), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["state"]["stage"] = "grpo"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="摘要不匹配"):
        load_workflow_state(path)
    with pytest.raises(ValueError, match="无效 plan_id"):
        workflow_state_path(tmp_path, "..\\escape", 1)


def test_workflow_rejects_broken_dataset_and_model_lineage():
    with pytest.raises(ValidationError, match="not-registered"):
        state(evidence=(
            WorkflowEvidence(
                operation_id="train-1", kind="training", status="succeeded", run_id="run-1",
                input_dataset_ids=("not-registered",), input_model_ids=("model-start",), model_id="model-out",
            ),
        ))
    with pytest.raises(ValidationError, match="model-never-produced"):
        state(evidence=(
            WorkflowEvidence(
                operation_id="eval-1", kind="evaluation", status="succeeded", run_id="run-2",
                input_model_ids=("model-never-produced",), evaluation_id="eval-1",
            ),
        ))


def test_workflow_accepts_complete_dataset_to_model_to_evaluation_chain():
    value = state().append(WorkflowEvidence(
        operation_id="register-1", kind="registration", status="succeeded", run_id="run-register",
        dataset_id="dataset-generated",
    )).append(WorkflowEvidence(
        operation_id="train-1", kind="training", status="succeeded", run_id="run-train",
        input_dataset_ids=("dataset-generated",), input_model_ids=("model-start",), model_id="model-trained",
    )).append(WorkflowEvidence(
        operation_id="eval-1", kind="evaluation", status="succeeded", run_id="run-eval",
        input_model_ids=("model-trained",), evaluation_id="eval-after",
    ))
    assert [item.kind for item in value.evidence] == ["registration", "training", "evaluation"]
