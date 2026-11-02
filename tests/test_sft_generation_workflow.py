from __future__ import annotations

import json

from onegpu_posttrain.data.targeted_plan import load_targeted_plan
from onegpu_posttrain.synthetic.generation import prepare_generation
from onegpu_posttrain.synthetic.providers import FakeProvider
from onegpu_posttrain.synthetic.sft_workflow import (
    execute_sft_generation,
    preflight_sft_generation,
)
from onegpu_posttrain.synthetic.workflow_bundle import register_sft_workflow_bundle
from test_sft_workflow_bundle import _inputs


def _responses(inputs):
    setup = prepare_generation(
        plan_path=inputs["plan"], profile_path=inputs["profile"], template_path=inputs["template"],
        protocol_path=inputs["protocol"], definition_path=inputs["definition"],
        report_path=inputs["report"], registry_path=inputs["registry"],
        train_seed_path=inputs["train_seeds"],
        protected_dataset_paths=[inputs["diagnostic"], inputs["holdout"]],
    )
    seeds = {row["id"]: row for row in setup.selection.records}
    responses = []
    for seed in setup.selection.records:
        metadata = seed.get("metadata", {})
        answer = metadata["reference"]
        responses.append(json.dumps({
            "problem_id": metadata["problem_id"],
            "solution": f"Calculate directly. Final answer: {answer} \\boxed{{{answer}}}",
            "final_answer": answer,
        }))
    assert len(responses) == len(seeds)
    return responses


def test_preflight_has_no_provider_side_effect(tmp_path):
    workspace, inputs = _inputs(tmp_path)
    bundle = register_sft_workflow_bundle(workspace, inputs=inputs)
    summary = preflight_sft_generation(
        workspace, bundle.bundle_id, usable_target=2, max_requests=2, target_candidates=2,
    )
    assert summary["status"] == "preflight_ready"
    assert summary["provider_calls"] == 0
    assert summary["network_used"] is False
    assert "_plan_payload" not in summary


def test_fake_provider_executes_bounded_sft_pipeline_offline(tmp_path):
    workspace, inputs = _inputs(tmp_path)
    bundle = register_sft_workflow_bundle(workspace, inputs=inputs)
    original = load_targeted_plan(inputs["plan"])
    provider = FakeProvider(_responses(inputs))
    result = execute_sft_generation(
        workspace, bundle.bundle_id, execution_id="sftgen-test-0001",
        usable_target=2, max_requests=2, target_candidates=2, provider=provider,
    )

    assert provider.calls == 2
    assert result["execution_id"] == "sftgen-test-0001"
    assert result["bundle_id"] == original.plan_id
    assert result["candidate_count"] == 2
    assert result["request_count"] == 2
    assert result["network_used"] is False
    assert result["usable_records"] >= 1
    assert result["training_dataset_id"].startswith("ds-targeted-sft-")
    assert (workspace / result["raw_output_path"]).is_file()


def test_generation_budget_cannot_exceed_registered_plan(tmp_path):
    workspace, inputs = _inputs(tmp_path)
    bundle = register_sft_workflow_bundle(workspace, inputs=inputs)
    provider = FakeProvider()
    try:
        preflight_sft_generation(
            workspace, bundle.bundle_id, usable_target=1, max_requests=6, target_candidates=1,
        )
    except ValueError as exc:
        assert "cannot exceed" in str(exc)
    else:
        raise AssertionError("budget overrun must be rejected before generation")
    assert provider.calls == 0
