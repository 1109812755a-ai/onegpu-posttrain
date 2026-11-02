from __future__ import annotations

from test_targeted_stage_plan import stage_plan

from onegpu_posttrain.data.manifest import load_manifest
from onegpu_posttrain.data.targeted_grpo import (
    ExecutableRewardContract,
    build_targeted_grpo_prompts,
    evaluate_reward_contract,
    prepare_imported_targeted_grpo_prompts,
    register_targeted_grpo_prompts,
    summarize_rollout_rewards,
    validate_reward_contract,
)


def _plan(contract):
    return stage_plan("grpo", target={
        "stage": "grpo", "scenario_id": "algebra",
        "reward_contract": {"evidence_id": contract.contract_id,
                            "content_sha256": contract.protocol_sha256,
                            "evidence_kind": "reward_contract"},
    }, budget={"unit": "prompts", "expected_usable_count": 2,
               "max_candidates": 5, "max_requests": 5, "max_output_tokens": 256})


def test_reward_contract_is_content_versioned_deterministic_and_executable():
    contract = ExecutableRewardContract.create()
    validate_reward_contract(contract)
    assert evaluate_reward_contract(contract, r"Final answer: \boxed{2}", "2") == {
        "accuracy": 1.0, "boxed_format": 1.0,
    }
    assert evaluate_reward_contract(contract, "Final answer: 3", "2") == {
        "accuracy": 0.0, "boxed_format": 0.0,
    }


def test_grpo_prompt_builder_requires_contract_and_rejects_holdout_or_missing_reference():
    contract = ExecutableRewardContract.create()
    plan = _plan(contract)
    result = build_targeted_grpo_prompts([
        {"id": "seed-1", "problem_id": "p1", "problem": "1+1?", "answer": "2"},
        {"id": "holdout", "problem_id": "p2", "problem": "2+2?", "answer": "4", "role": "final_holdout"},
        {"id": "unverified", "problem_id": "p3", "problem": "3+3?"},
    ], plan=plan, contract=contract)
    assert result["status"] == "partial"
    assert result["usable_prompts"] == 1
    assert result["rejections"]["non_training_ancestry"] == 1
    assert result["rejections"]["missing_prompt_or_trusted_reference"] == 1
    assert result["records"][0]["metadata"]["reward_contract_id"] == contract.contract_id
    assert result["quality"]["trusted_reference_coverage"] == {
        "status": "measured", "numerator": 1, "denominator": 3,
        "value": 1 / 3, "unit": "ratio",
    }
    assert result["quality"]["prompt_length_chars"]["denominator"] == 1
    assert result["quality"]["prompt_length_tokens"]["status"] == "not_measured"
    assert result["quality"]["truncation"]["status"] == "not_measured"
    assert result["quality"]["reward_variance"]["status"] == "not_measured"


def test_grpo_import_preserves_only_explicit_matching_train_ancestry():
    contract = ExecutableRewardContract.create()
    plan = _plan(contract)
    generated = build_targeted_grpo_prompts([
        {"id": "seed-1", "problem": "1+1?", "answer": "2"},
        {"id": "seed-2", "problem": "2+2?", "answer": "4"},
    ], plan=plan, contract=contract)
    imported = prepare_imported_targeted_grpo_prompts(
        [generated["records"][0], generated["records"][0],
         generated["records"][1] | {"metadata": generated["records"][1]["metadata"] | {"data_role": "final_holdout"}}],
        plan=plan, contract=contract,
    )
    assert imported["status"] == "partial"
    assert imported["candidate_count"] == 3
    assert imported["usable_prompts"] == 1
    assert imported["rejections"]["duplicate_record_id"] == 1
    assert imported["rejections"]["mismatched_plan_reward_or_training_role"] == 1
    assert imported["quality"]["trusted_reference_coverage"]["denominator"] == 3


def test_grpo_rollout_metrics_keep_true_denominators_and_zero_std_not_zero_metric():
    contract = ExecutableRewardContract.create()
    report = summarize_rollout_rewards(contract, [
        {"prompt_id": "p1", "completion": r"\boxed{2}", "reference_answer": "2",
         "completion_tokens": 4, "truncated": False},
        {"prompt_id": "p1", "completion": "3", "reference_answer": "2"},
        {"prompt_id": "p2", "completion": r"\boxed{4}", "reference_answer": "4"},
        {"prompt_id": "p3", "completion": "unscorable", "reference_answer": ""},
        {"completion": r"\boxed{2}", "reference_answer": "2"},
    ])
    assert report["verified_rollout_count"] == 3
    assert report["rollout_count"] == 5
    assert report["prompt_count"] == 2
    assert report["prompt_count_with_repeated_rollouts"] == 1
    assert report["zero_std_prompt_count"] == 0
    assert report["zero_std_ratio"] == 0
    assert report["verifier_coverage"] == {
        "status": "measured", "numerator": 3, "denominator": 5, "value": 0.6, "unit": "ratio",
    }
    assert report["completion_length_chars"]["denominator"] == 4
    assert report["completion_length_tokens"]["denominator"] == 1
    assert report["truncation"] == {
        "status": "measured", "truncated_count": 0, "known_count": 1,
        "denominator": 5, "value": 0, "unit": "ratio",
    }
    assert report["reward_variances"]["accuracy"] is not None


def test_grpo_registered_partial_dataset_binds_contract_hash_and_plan(tmp_path):
    contract = ExecutableRewardContract.create()
    plan = _plan(contract)
    report = build_targeted_grpo_prompts([
        {"id": "seed-1", "problem": "1+1?", "answer": "2"},
    ], plan=plan, contract=contract)
    result = register_targeted_grpo_prompts(
        report, plan=plan, contract=contract, workspace=tmp_path,
        output_root=tmp_path / "data/processed/generated", provider_profile_id="fake",
    )
    manifest = load_manifest(tmp_path / result["registration_path"] / "manifests/grpo-user-upload-v1.yaml")
    assert manifest.dataset_id == result["dataset_id"]
    assert manifest.metadata["targeted_plan_id"] == plan.plan_id
    assert manifest.metadata["reward_contract_protocol_sha256"] == contract.protocol_sha256
    assert manifest.metadata["targeted_grpo_quality"]["usable_prompt_count"] == 1
    assert manifest.lineage.parents == [plan.training_seed.dataset_id]
    assert result["record_count"] == 1


def test_grpo_registration_rejects_candidate_with_forged_reward_evidence(tmp_path):
    contract = ExecutableRewardContract.create()
    plan = _plan(contract)
    report = build_targeted_grpo_prompts([
        {"id": "seed-1", "problem": "1+1?", "answer": "2"},
    ], plan=plan, contract=contract)
    report["records"][0]["metadata"]["reward_contract_implementation_sha256"] = "0" * 64
    try:
        register_targeted_grpo_prompts(
            report, plan=plan, contract=contract, workspace=tmp_path,
            output_root=tmp_path / "data/processed/generated", provider_profile_id="fake",
        )
    except ValueError as exc:
        assert "mismatched plan or reward evidence" in str(exc)
    else:
        raise AssertionError("tampered GRPO reward evidence must not be registered")

