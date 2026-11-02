from __future__ import annotations

import pytest

from onegpu_posttrain.data.policy import (
    DataPolicy,
    PolicyInvariants,
    PolicyRules,
    SplitSnapshot,
    parse_policy,
    policy_diff,
    policy_templates,
)


def test_policy_roundtrip_identity_and_display_name():
    policy = DataPolicy(stage="sft")
    assert parse_policy(policy.artifact()) == policy
    assert policy.model_copy(update={"title": "另一个名字"}).policy_id == policy.policy_id
    changed = DataPolicy(stage="sft", rules=PolicyRules(correctness="report_only"))
    assert changed.policy_id != policy.policy_id
    assert policy_diff(policy, changed) == [{"field": "rules.correctness", "a": "filter", "b": "report_only"}]
    artifact = policy.artifact()
    artifact["config"]["resources"]["max_length"] = 20
    with pytest.raises(ValueError, match="摘要"):
        parse_policy(artifact)


@pytest.mark.parametrize("payload", [
    {"stage": "cpt"}, {"stage": "sft", "script": "do_stuff()"},
    {"stage": "sft", "rules": {"min_length_ratio": 0.5, "max_length_ratio": 2}},
    {"stage": "preference", "rules": {"correctness": "report_only"}},
    {"stage": "sft", "usage": "baseline", "rules": {"completeness": "report_only"}},
])
def test_policy_rejects_unsupported_or_misrepresented_settings(payload):
    with pytest.raises(ValueError):
        DataPolicy.model_validate(payload)


def test_invariants_and_templates():
    with pytest.raises(ValueError):
        PolicyInvariants(train_eval_separation=False)
    assert len(policy_templates()) == 4
    for artifact in policy_templates():
        assert parse_policy(artifact).usage == "exploratory"


def test_split_is_independent_of_quotas():
    snapshot = SplitSnapshot()
    before = [snapshot.pool(f"problem-{i}") for i in range(300)]
    policy = DataPolicy(stage="sft")
    policy.resources.target_count = 3
    assert before == [snapshot.pool(f"problem-{i}") for i in range(300)]
    assert set(before) == {"sft", "alignment", "holdout"}
