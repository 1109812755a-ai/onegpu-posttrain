import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from onegpu_posttrain.data.manifest import sha256_file
from onegpu_posttrain.data.quality_review import (
    HumanReviewDecisions,
    QualityReview,
    QualityReviewBundleEntry,
    QualityReviewBundleIndex,
    QualityReviewCheck,
    QualityReviewProfile,
    ReviewCheckRule,
    ReviewerIdentity,
    _semantic_checks,
    apply_manual_review_decisions,
    classify_dataset_usage,
    load_quality_review_bundle,
    load_quality_review_profile,
    quality_review_evidence_sha256,
)

ROOT = Path(__file__).parents[1]


def test_versioned_review_profile_loads_and_hashes_stably():
    path = ROOT / "configs/data/quality-review/mini-review-v1.yaml"
    profile = load_quality_review_profile(path)
    assert profile.profile_id == "mini-stage-quality-review"
    assert set(profile.stages) == {"cpt", "cpt_holdout", "sft", "preference", "preference_eval", "grpo", "eval"}
    assert profile.profile_sha256 == load_quality_review_profile(path).profile_sha256
    assert all(rule.minimum is None or rule.maximum is None or rule.minimum <= rule.maximum
               for stage in profile.stages.values() for rule in stage.checks)


@pytest.mark.parametrize("payload", [
    {"schema_version": "0.9", "profile_id": "p", "profile_version": "1", "description": "x",
     "stages": {"sft": {"checks": [{"check_id": "x", "method": "m"}]}}},
    {"schema_version": "1.0", "profile_id": "p", "profile_version": "1", "description": "x",
     "composite_score": "average", "stages": {"sft": {"checks": [{"check_id": "x", "method": "m"}]}}},
])
def test_review_profile_rejects_unversioned_or_composite_policy(payload):
    with pytest.raises(ValidationError):
        QualityReviewProfile.model_validate(payload)


def test_required_check_cannot_ignore_unsupported_evidence():
    payload = {"schema_version": "1.0", "profile_id": "p", "profile_version": "1", "description": "x",
               "stages": {"sft": {"checks": [{"check_id": "x", "method": "m",
                                                  "requirement": "required",
                                                  "unsupported_action": "not_applicable"}]}}}
    with pytest.raises(ValidationError, match="required checks"):
        QualityReviewProfile.model_validate(payload)


@pytest.mark.parametrize("usage", ["baseline", "exploratory", "intentionally_noisy"])
def test_dataset_usage_classification_preserves_supported_values(usage):
    assert classify_dataset_usage({"usage": usage}) == usage


@pytest.mark.parametrize("metadata", [None, {}, {"usage": "unknown"}, {"usage": None}])
def test_missing_or_unknown_usage_is_legacy_unknown(metadata):
    assert classify_dataset_usage(metadata) == "legacy_unknown"


def test_sft_semantic_review_uses_only_explicit_reference_and_keeps_coverage(monkeypatch):
    from onegpu_posttrain.evaluation import verification

    monkeypatch.setattr(verification, "_verification_identity", lambda: (None, None))
    result = _semantic_checks("sft", [
        {"messages": [{"role": "assistant", "content": "Final answer: 4"}],
         "metadata": {"reference_answer": "4", "correctness_math_verify": False}},
        {"messages": [{"role": "assistant", "content": "No answer"}],
         "metadata": {"reference_answer": "4"}},
        {"messages": [{"role": "assistant", "content": "Final answer: 5"}], "metadata": {}},
        {"messages": [{"role": "assistant", "content": "Final answer: 5"}],
         "metadata": {"reference_answer": "4", "correctness_math_verify": True}},
    ])
    assert result["counts"] == {"correct": 1, "unparseable": 1, "unverified": 2}
    assert result["denominator"] == 4
    assert result["coverage"] == pytest.approx(2 / 4)


def test_preference_semantic_review_checks_both_responses_not_upstream_labels(monkeypatch):
    from onegpu_posttrain.evaluation import verification

    monkeypatch.setattr(verification, "_verification_identity", lambda: (None, None))
    result = _semantic_checks("preference", [{
        "chosen": "Final answer: 4", "rejected": "Final answer: 5",
        "metadata": {"reference_answer": "4", "chosen_correct": False, "rejected_correct": True},
    }])
    assert result["counts"] == {"correct": 1, "unverified": 1}
    assert result["denominator"] == 2


def test_semantic_review_keeps_timeout_and_worker_failure_out_of_incorrect(monkeypatch):
    from onegpu_posttrain.evaluation import verification

    monkeypatch.setattr(verification, "_verification_identity", lambda: ("math-verify", "0.9.0"))
    monkeypatch.setattr(verification, "_run_bounded_batch", lambda pairs, timeout: ["timeout", "verifier_error"])
    result = _semantic_checks("sft", [
        {"messages": [{"role": "assistant", "content": "Final answer: 1/2"}],
         "metadata": {"reference_answer": "0.5"}},
        {"messages": [{"role": "assistant", "content": "Final answer: x+1"}],
         "metadata": {"reference_answer": "x+2"}},
    ])
    assert result["counts"] == {"timeout": 1, "verifier_error": 1}
    assert result["counts"].get("incorrect", 0) == 0
    assert result["decision_coverage"] == 0
    assert result["coverage"] == 1


def test_manual_decision_preserves_unavailable_evidence_and_requires_rationale():
    check = QualityReviewCheck(
        check_id="human_reasoning_review", evidence_status="not_measured", method="manual_review",
        decision="pending", rationale="等待人工审核",
    )
    updated = apply_manual_review_decisions(
        [check], [ReviewCheckRule(check_id="human_reasoning_review", method="manual_review")],
        {"human_reasoning_review": {"decision": "pass", "rationale": "Reviewed 4 records; no issue."}},
    )
    assert updated[0].decision == "pass"
    assert updated[0].evidence_status == "not_measured"
    assert updated[0].rationale.startswith("Reviewed")
    with pytest.raises(ValidationError, match="rationale"):
        HumanReviewDecisions.model_validate({"reviewer_id": "reviewer", "decisions": {
            "openr1.sft_train": {"human_reasoning_review": {"decision": "pass"}},
        }})


def test_manual_decision_cannot_override_automated_check():
    check = QualityReviewCheck(
        check_id="independent_semantic_correctness", evidence_status="unsupported",
        method="reference_answer_verification", decision="pending", rationale="Verifier unavailable.",
    )
    with pytest.raises(ValueError, match="cannot replace automated"):
        apply_manual_review_decisions(
            [check], [ReviewCheckRule(check_id=check.check_id, method="reference_answer_verification")],
            {check.check_id: {"decision": "pass", "rationale": "I guess it is fine."}},
        )


def _write_review_bundle(tmp_path: Path, review_path_value="reviews/sft.json"):
    review_dir = tmp_path / "reviews"
    review_dir.mkdir(exist_ok=True)
    checks = [QualityReviewCheck(check_id="manual", method="manual_review",
        evidence_status="not_measured", decision="pending", rationale="Pending.")]
    review_payload = QualityReview(
        review_id="review-1", dataset_key="openr1.sft_train", dataset_id="ds-sft",
        stage="sft", content_sha256="a" * 64, quality_report_id="quality-1",
        quality_report_sha256="b" * 64, profile_id="profile", profile_version="1",
        profile_sha256="c" * 64, checker_name="checker", checker_version="1",
        usage="baseline",
        reviewer=ReviewerIdentity(reviewer_id="reviewer", method="human_review"),
        checks=checks,
        evidence_sha256=quality_review_evidence_sha256(checks),
    ).model_dump(mode="json")
    review_file = review_dir / "sft.json"
    review_file.write_text(json.dumps(review_payload), encoding="utf-8")
    entry = QualityReviewBundleEntry(
        dataset_key="openr1.sft_train", dataset_id="ds-sft", stage="sft",
        content_sha256="a" * 64, quality_report_id="quality-1", quality_report_sha256="b" * 64,
        review_id="review-1", review_path=review_path_value,
        review_sha256=sha256_file(review_file), profile_sha256="c" * 64,
        usage="baseline", outcome="pending",
    )
    index = QualityReviewBundleIndex(
        assessment_id="assessment-1", created_at="2026-10-19T00:00:00Z",
        profile_id="profile", profile_version="1", profile_sha256="c" * 64,
        source_quality_assessment_id="quality-assessment-1", datasets=[entry],
    )
    (tmp_path / "index.json").write_text(index.model_dump_json(), encoding="utf-8")
    return entry


def test_review_bundle_loader_checks_hash_identity_and_containment(tmp_path):
    _write_review_bundle(tmp_path)
    _, index = load_quality_review_bundle(tmp_path)
    assert index.datasets[0].dataset_id == "ds-sft"
    (tmp_path / "reviews/sft.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_quality_review_bundle(tmp_path)


def test_review_bundle_loader_rejects_escaping_path(tmp_path):
    (tmp_path.parent / "outside.json").write_text("{}", encoding="utf-8")
    _write_review_bundle(tmp_path, "../outside.json")
    with pytest.raises(ValueError):
        load_quality_review_bundle(tmp_path)


def test_review_bundle_index_rejects_duplicate_dataset_keys(tmp_path):
    entry = _write_review_bundle(tmp_path)
    with pytest.raises(ValidationError, match="duplicate dataset keys"):
        QualityReviewBundleIndex(
            assessment_id="assessment-1", created_at="2026-10-19T00:00:00Z",
            profile_id="profile", profile_version="1", profile_sha256="c" * 64,
            source_quality_assessment_id="quality-assessment-1", datasets=[entry, entry],
        )


def test_verifier_configuration_is_part_of_quality_evidence_digest():
    base = QualityReviewCheck(
        check_id="independent_semantic_correctness", metric_id="semantic_correctness",
        evidence_status="measured", value=1.0, unit="verified_response_ratio",
        numerator=1, denominator=1, coverage=1.0, method=(
            "reference_answer_verification;math-answer-verification-v1;math-verify@0.9.0;"
            "batch=32;deadline=60.0s;max-chars=4096;inner-timeout=disabled"
        ), decision="pending", rationale="Verifier version and settings are bound.",
    )
    same = base.model_copy(deep=True)
    changed_protocol = base.model_copy(update={"method": base.method.replace("0.9.0", "0.9.1")})
    digest = quality_review_evidence_sha256([base])
    assert digest == quality_review_evidence_sha256([same])
    assert digest != quality_review_evidence_sha256([changed_protocol])
