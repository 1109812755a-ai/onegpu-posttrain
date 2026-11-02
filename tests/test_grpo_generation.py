from __future__ import annotations

import pytest
from test_targeted_stage_plan import stage_plan

from onegpu_posttrain.data.targeted_grpo import ExecutableRewardContract
from onegpu_posttrain.synthetic.grpo_generation import generate_grpo_prompt_candidates
from onegpu_posttrain.synthetic.providers import FakeProvider, ProviderProfile


def _targeted_plan(contract):
    return stage_plan("grpo", budget={
        "unit": "prompts", "expected_usable_count": 1, "max_candidates": 2,
        "max_requests": 2, "max_output_tokens": 128,
    }, target={
        "stage": "grpo", "scenario_id": "algebra",
        "reward_contract": {
            "evidence_id": contract.contract_id,
            "content_sha256": contract.protocol_sha256,
            "evidence_kind": "reward_contract",
        },
    })


def test_ordinary_prompt_candidates_never_trust_answer_or_reward_labels():
    contract = ExecutableRewardContract.create()
    provider = FakeProvider(['{"raw_problem":"A generated arithmetic problem"}'])
    report = generate_grpo_prompt_candidates(
        [{"id": "topic-1", "problem": "addition"}], mode="ordinary",
        profile=ProviderProfile(profile_id="fake", adapter="fake", model="fixture"),
        provider=provider, contract=contract, request_budget=1, candidate_budget=1,
        max_output_tokens=128, timeout_seconds=177,
    )
    assert provider.calls == report["request_count"] == 1
    candidate = report["candidates"][0]
    assert candidate["raw_problem"] == "A generated arithmetic problem"
    assert candidate["reference_status"] == "missing"
    assert candidate["reward_evidence_status"] == "not_evaluated"
    assert candidate["trainable"] is False and report["trainable"] is False
    assert "answer" not in candidate and "solution" not in candidate
    assert report["provider_identity_status"] == "profile_declared_not_independently_verified"
    assert report["timeout_seconds"] == 177
    assert report["registered_model_association_status"] == "not_proven"
    assert report["registered_model_id"] is None and report["registered_model_sha256"] is None


def test_targeted_generation_requires_exact_reward_and_excludes_holdout():
    contract = ExecutableRewardContract.create()
    plan = _targeted_plan(contract)
    provider = FakeProvider(['{"raw_problem":"A train-seed variation"}'])
    report = generate_grpo_prompt_candidates(
        [
            {"id": "seed-1", "raw_problem": "one plus one", "metadata": {"data_role": "train_seed"}},
            {"id": "holdout-1", "raw_problem": "holdout", "metadata": {"data_role": "final_holdout"}},
        ], mode="targeted", profile=ProviderProfile(profile_id="fake", adapter="fake", model="fixture"),
        provider=provider, contract=contract, targeted_plan=plan, request_budget=2,
        candidate_budget=2, max_output_tokens=128,
    )
    assert report["rejections"] == {"non_training_ancestry": 1}
    assert report["candidates"][0]["source_record_id"] == "seed-1"
    assert report["candidates"][0]["targeted_plan_id"] == plan.plan_id

    other_contract = ExecutableRewardContract.create(("accuracy",))
    no_call_provider = FakeProvider()
    with pytest.raises(ValueError, match="reward contract or generation budget"):
        generate_grpo_prompt_candidates(
            [], mode="targeted", profile=ProviderProfile(profile_id="fake", adapter="fake", model="fixture"),
            provider=no_call_provider, contract=other_contract, targeted_plan=plan,
            request_budget=1, candidate_budget=1, max_output_tokens=64,
        )
    assert no_call_provider.calls == 0


def test_provider_cannot_return_answer_or_claim_reward_and_budget_is_bounded():
    contract = ExecutableRewardContract.create()
    provider = FakeProvider(['{"raw_problem":"question","answer":"42","reward":1}'])
    report = generate_grpo_prompt_candidates(
        [{"id": "topic-1", "problem": "algebra"}], mode="ordinary",
        profile=ProviderProfile(profile_id="fake", adapter="fake", model="fixture"),
        provider=provider, contract=contract, request_budget=1, candidate_budget=1,
        max_output_tokens=128,
    )
    assert report["candidates"] == []
    assert report["rejections"] == {"invalid_or_answer_bearing_provider_response": 1}

    never_called = FakeProvider()
    with pytest.raises(ValueError, match="safety limit"):
        generate_grpo_prompt_candidates(
            [], mode="ordinary", profile=ProviderProfile(profile_id="fake", adapter="fake", model="fixture"),
            provider=never_called, contract=contract, request_budget=10_001,
            candidate_budget=1, max_output_tokens=128,
        )
    assert never_called.calls == 0
