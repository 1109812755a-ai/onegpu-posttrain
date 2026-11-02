from __future__ import annotations

import json
import pytest

from test_targeted_stage_plan import stage_plan

from onegpu_posttrain.data.manifest import load_manifest
from onegpu_posttrain.data.targeted_preference import (
    build_targeted_preference_pairs,
    generate_targeted_preference_candidates,
    register_targeted_preference_pairs,
)
from onegpu_posttrain.synthetic.providers import FakeProvider, ProviderProfile
from onegpu_posttrain.evaluation import verification


@pytest.fixture(autouse=True)
def deterministic_optional_answer_verifier(monkeypatch):
    """Exercise pairing logic without requiring the optional math-verify extra."""
    monkeypatch.setattr(verification, "_verification_identity", lambda: ("math-verify", "test"))

    def verify(pairs, _timeout):
        return ["correct" if pair.prediction.strip() == pair.reference.strip() else "incorrect" for pair in pairs]

    monkeypatch.setattr(verification, "_run_bounded_batch", verify)


def _candidate(cid: str, answer: str, *, ref=2, role="train_seed", source="seed-1"):
    return {
        "candidate_id": cid, "problem_id": f"problem-{source}", "source_record_id": source,
        "source_role": role, "problem": "What is 1+1?", "completion": f"Final answer: {answer}",
        "reference_answer": ref, "provider_claimed_correct": answer == "2",
    }


def test_builds_only_opposite_independently_verified_pair_and_partial_denominator():
    plan = stage_plan("preference", budget={
        "unit": "pairs", "expected_usable_count": 2, "max_candidates": 10,
        "max_requests": 10, "max_output_tokens": 100,
    })
    result = build_targeted_preference_pairs([_candidate("good", "2"), _candidate("bad", "3")], plan=plan)
    assert result["status"] == "partial"
    assert result["candidate_count"] == 2
    assert result["usable_pairs"] == 1
    assert result["shortfall"] == 1
    row = result["records"][0]
    assert row["chosen"] == "Final answer: 2"
    assert row["rejected"] == "Final answer: 3"
    assert row["metadata"]["chosen_correct"] is True
    assert row["metadata"]["rejected_correct"] is False
    assert row["metadata"]["targeted_plan_id"] == plan.plan_id


def test_same_quality_unknown_duplicate_and_holdout_candidates_never_make_pairs():
    plan = stage_plan("preference")
    both_correct = [_candidate("a", "2"), {
        **_candidate("b", "2", source="seed-1"), "completion": r"Final answer: \boxed{2}",
    }]
    both_wrong = [_candidate("c", "4", source="seed-2"), _candidate("d", "5", source="seed-2")]
    unknown = [_candidate("e", "2", ref=None, source="seed-3"), _candidate("f", "3", ref=None, source="seed-3")]
    contaminated = [_candidate("g", "2", role="final_holdout", source="seed-4"), _candidate("h", "3", source="seed-4")]
    duplicate = [_candidate("i", "2", source="seed-5"), _candidate("j", "2", source="seed-5")]
    result = build_targeted_preference_pairs(both_correct + both_wrong + unknown + contaminated + duplicate, plan=plan)
    assert result["usable_pairs"] == 0
    assert result["records"] == []
    assert result["rejections"]["both_correct_or_no_wrong_candidate"] == 3
    assert result["rejections"]["both_incorrect_or_no_correct_candidate"] == 3
    assert result["rejections"]["verification_unverified"] == 2
    assert result["rejections"]["non_training_ancestry"] == 1
    assert result["rejections"]["duplicate_candidate_content"] == 1


def test_registration_binds_partial_pair_records_plan_and_manifest(tmp_path):
    plan = stage_plan("preference")
    report = build_targeted_preference_pairs([_candidate("good", "2"), _candidate("bad", "3")], plan=plan)
    registration = register_targeted_preference_pairs(
        report, plan=plan, workspace=tmp_path, output_root=tmp_path / "data/processed/generated",
        provider_profile_id="test-profile",
    )
    assert registration["record_count"] == 1
    assert registration["generation_status"] == "partial"
    manifest = load_manifest(tmp_path / registration["registration_path"] / "manifests/preference-model-generated-v1.yaml")
    assert manifest.dataset_id == registration["dataset_id"]
    assert manifest.metadata["targeted_plan_id"] == plan.plan_id
    assert manifest.lineage.parents == [plan.training_seed.dataset_id]
    expected_parent_ref = {
        "dataset_id": plan.training_seed.dataset_id,
        "content_sha256": plan.training_seed.content_sha256,
        "role": "train_seed",
    }
    assert registration["parent_dataset_refs"] == [expected_parent_ref]
    assert manifest.metadata["parent_dataset_refs"] == [expected_parent_ref]
    assert manifest.metadata["ordering_evidence_status"] == "verified"
    assert manifest.hashes["canonical_records_sha256"] == registration["canonical_sha256"]
    assert manifest.metadata["preference_quality_summary"] == {
        "status": "partial", "target_pairs": 100, "candidate_count": 2, "usable_pairs": 1,
        "shortfall": 99, "rejections": {}, "independently_verified_pairs": 1,
        "evidence_scope": "final-answer-only", "parent_dataset_refs": [expected_parent_ref],
    }
    registered_row = json.loads((tmp_path / registration["registration_path"] / "canonical/preference.jsonl").read_text(encoding="utf-8"))
    assert registered_row["metadata"]["ordering_evidence_status"] == "verified"
    assert registered_row["metadata"]["ordering_evidence_method"] == "independent_final_answer_reference_check"
    assert registered_row["metadata"]["data_role"] == "train_pairs"
    assert registered_row["metadata"]["source_data_role"] == "train_seed"
    build_lineage = load_manifest(tmp_path / registration["registration_path"] / "manifests/preference-model-generated-v1.yaml")
    build_run = tmp_path / registration["registration_path"] / "runs" / build_lineage.lineage.produced_by_run_id / "lineage.json"
    assert json.loads(build_run.read_text(encoding="utf-8"))["targeted_plan_id"] == plan.plan_id
    from onegpu_posttrain.cli import _dataset_targeted_plan
    assert _dataset_targeted_plan(tmp_path / registration["registration_path"] / "canonical/preference.jsonl") == (
        plan.plan_id, plan.version,
    )


def test_registration_rejects_pair_rows_without_bound_independent_evidence(tmp_path):
    plan = stage_plan("preference")
    report = build_targeted_preference_pairs([_candidate("good", "2"), _candidate("bad", "3")], plan=plan)
    report["records"][0]["metadata"]["chosen_correct"] = False
    with pytest.raises(ValueError, match="independent chosen/rejected evidence"):
        register_targeted_preference_pairs(
            report, plan=plan, workspace=tmp_path,
            output_root=tmp_path / "data/processed/generated", provider_profile_id="test-profile",
        )


def test_provider_generation_uses_only_train_seeds_and_pairs_after_independent_check():
    plan = stage_plan("preference", budget={
        "unit": "pairs", "expected_usable_count": 2, "max_candidates": 4,
        "max_requests": 4, "max_output_tokens": 100,
    })
    profile = ProviderProfile(profile_id="fake-pref", adapter="fake", model="fixture")
    provider = FakeProvider([
        '{"completion":"Work it out. Final answer: 2"}',
        '{"completion":"Work it out. Final answer: 3"}',
    ])
    generated = generate_targeted_preference_candidates([
        {"id": "seed-1", "raw_problem": "What is 1+1?", "answer": "2",
         "metadata": {"data_role": "train_seed", "problem_id": "p1"}},
        {"id": "holdout-1", "raw_problem": "1+1?", "answer": "2",
         "metadata": {"data_role": "final_holdout", "problem_id": "p2"}},
    ], plan=plan, profile=profile, provider=provider)
    assert provider.calls == generated["request_count"] == 2
    assert generated["rejected_seeds"] == {"non_training_ancestry": 1}
    assert generated["network_used"] is False
    assert generated["correctness_status"] == "not_independently_verified"
    paired = build_targeted_preference_pairs(generated["candidates"], plan=plan)
    assert paired["usable_pairs"] == 1
    assert paired["records"][0]["metadata"]["parent_record_ids"] == ["seed-1"]


def test_provider_request_budget_and_schema_errors_are_partial_not_pairs():
    plan = stage_plan("preference", budget={
        "unit": "pairs", "expected_usable_count": 2, "max_candidates": 4,
        "max_requests": 4, "max_output_tokens": 100,
    })
    profile = ProviderProfile(profile_id="fake-pref", adapter="fake", model="fixture")
    provider = FakeProvider(["not-json"])
    generated = generate_targeted_preference_candidates([
        {"id": "seed-1", "raw_problem": "What is 1+1?", "answer": "2",
         "metadata": {"data_role": "train_seed", "problem_id": "p1"}},
    ], plan=plan, profile=profile, provider=provider, max_requests=1)
    assert generated["status"] == "partial"
    assert generated["request_count"] == 1
    assert generated["request_errors"] == {"provider_or_response_error": 1}
    assert generated["candidates"] == []
    assert build_targeted_preference_pairs([], plan=plan)["usable_pairs"] == 0


def test_provider_budget_cannot_exceed_registered_plan():
    plan = stage_plan("preference")
    provider = FakeProvider()
    profile = ProviderProfile(profile_id="fake-pref", adapter="fake", model="fixture")
    with pytest.raises(ValueError, match="cannot exceed"):
        generate_targeted_preference_candidates([], plan=plan, profile=profile,
                                                provider=provider, max_requests=plan.budget.max_requests + 1)
    assert provider.calls == 0


def test_exhausted_request_budget_never_fabricates_a_preference_pair():
    plan = stage_plan("preference", budget={
        "unit": "pairs", "expected_usable_count": 1, "max_candidates": 4,
        "max_requests": 4, "max_output_tokens": 100,
    })
    profile = ProviderProfile(profile_id="fake-pref", adapter="fake", model="fixture")
    provider = FakeProvider(['{"completion":"Final answer: 2"}'])
    generated = generate_targeted_preference_candidates([{
        "id": "seed-1", "raw_problem": "What is 1+1?", "answer": "2",
        "metadata": {"data_role": "train_seed", "problem_id": "p1"},
    }], plan=plan, profile=profile, provider=provider, max_requests=1)

    paired = build_targeted_preference_pairs(generated["candidates"], plan=plan)
    assert generated["stop_reason"] == "request_budget_exhausted"
    assert generated["request_count"] == 1
    assert generated["candidate_count"] == 1
    assert paired["usable_pairs"] == 0
    assert paired["records"] == []
    assert paired["shortfall"] == 1


def test_pair_target_budget_rejects_extra_verified_candidates_without_fabrication():
    plan = stage_plan("preference", budget={
        "unit": "pairs", "expected_usable_count": 1, "max_candidates": 4,
        "max_requests": 4, "max_output_tokens": 100,
    })
    candidates = [
        _candidate("good-1", "2", source="seed-1"), _candidate("bad-1", "3", source="seed-1"),
        _candidate("good-2", "2", source="seed-2"), _candidate("bad-2", "3", source="seed-2"),
    ]

    result = build_targeted_preference_pairs(candidates, plan=plan)
    assert result["usable_pairs"] == 1
    assert len(result["records"]) == 1
    assert result["rejections"] == {"target_budget_reached": 2}
    assert result["shortfall"] == 0
