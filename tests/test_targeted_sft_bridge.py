from __future__ import annotations

from pathlib import Path

import pytest

from onegpu_posttrain.data.targeted import DataIdentity, EvaluationContract, ModelIdentity
from onegpu_posttrain.data.targeted_plan import load_targeted_plan
from onegpu_posttrain.data.targeted_sft_bridge import validate_targeted_sft_bridge
from onegpu_posttrain.data.targeted_stage_plan import EvidenceRef, StageDataBudget, TargetedStagePlan
from onegpu_posttrain.synthetic.workflow_bundle import register_sft_workflow_bundle
from test_sft_workflow_bundle import _inputs


def make_shared_plan(bundle):
    legacy = load_targeted_plan(bundle["plan"])
    return TargetedStagePlan(
        target={"stage": "sft", "scenario_id": legacy.target_scenario_id,
                "evidence": {"evidence_id": legacy.scenario_report_id,
                             "content_sha256": legacy.scenario_report_sha256,
                             "evidence_kind": "scenario_report"}},
        training_seed=legacy.training_seed,
        starting_model=legacy.starting_model,
        diagnostic_evaluation={"dataset_id": legacy.diagnostic.dataset_id,
                               "dataset_sha256": legacy.diagnostic.dataset_sha256,
                               "protocol_version": legacy.diagnostic.protocol_version,
                               "config_sha256": legacy.diagnostic.config_sha256,
                               "role": "diagnostic_dev"},
        final_evaluation=legacy.final_evaluation,
        provider_profile_id=legacy.provider_profile_id,
        budget={"unit": "records", "expected_usable_count": legacy.budget.target_candidates,
                "max_candidates": legacy.budget.target_candidates,
                "max_requests": legacy.budget.max_requests,
                "max_output_tokens": legacy.budget.max_output_tokens,
                "max_retries": legacy.budget.max_retries,
                "timeout_seconds": legacy.budget.timeout_seconds},
        training_budget=legacy.training_budget, seed=legacy.seed,
    )


def test_shared_sft_plan_matches_existing_seeded_generation_bundle(tmp_path):
    workspace, inputs = _inputs(tmp_path)
    bundle = register_sft_workflow_bundle(workspace, inputs=inputs)
    result = validate_targeted_sft_bridge(workspace, stage_plan=make_shared_plan(inputs), bundle_id=bundle.bundle_id)
    assert result["checks_passed"] == 17
    assert result["scenario_id"] == "algebra"
    assert result["stage_plan_id"].startswith("plan-targeted-")


def test_shared_sft_plan_cannot_silently_switch_seed_or_scenario(tmp_path):
    workspace, inputs = _inputs(tmp_path)
    bundle = register_sft_workflow_bundle(workspace, inputs=inputs)
    plan = make_shared_plan(inputs)
    changed = plan.model_copy(update={
        "target": plan.target.model_copy(update={"scenario_id": "arithmetic"}),
    })
    with pytest.raises(ValueError, match="scenario_id"):
        validate_targeted_sft_bridge(workspace, stage_plan=changed, bundle_id=bundle.bundle_id)
