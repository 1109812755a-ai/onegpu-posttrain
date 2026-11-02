from __future__ import annotations

import json

from onegpu_posttrain.data.tokenization import WhitespaceTokenizer
from onegpu_posttrain.synthetic.candidates import validate_candidate_response
from onegpu_posttrain.synthetic.providers import OutputProtocol

SHA = "a" * 64
PROTOCOL = OutputProtocol(
    protocol_id="sft-candidate", version="v1",
    required_fields=["problem_id", "solution", "final_answer"],
)


def _validate(payload, *, finish_reason="stop", max_tokens=100):
    return validate_candidate_response(
        json.dumps(payload),
        source_record_id="seed-1",
        expected_problem_id="problem-1",
        candidate_slot=0,
        request_sha256=SHA,
        finish_reason=finish_reason,
        question="What is 2+2?",
        protocol=PROTOCOL,
        tokenizer=WhitespaceTokenizer(),
        max_training_tokens=max_tokens,
    )


def test_well_formed_but_mathematically_wrong_answer_is_not_marked_verified():
    candidate = _validate({
        "problem_id": "problem-1", "solution": "Compute 2+2=5. Final answer: 5",
        "final_answer": "5",
    })
    assert candidate.format_status == "valid"
    assert candidate.answer_consistency == "match"
    assert candidate.correctness_status == "unverified"
    assert candidate.validation_status == "candidate_ready_for_quality_review"
    assert "verified" not in candidate.validation_status


def test_final_answer_conflict_is_reported_without_claiming_correctness():
    candidate = _validate({
        "problem_id": "problem-1", "solution": "Compute 2+2. Final answer: 4",
        "final_answer": "5",
    })
    assert candidate.answer_consistency == "mismatch"


def test_length_finish_reason_rejects_even_parseable_json():
    candidate = _validate({
        "problem_id": "problem-1", "solution": "Final answer: 4", "final_answer": "4",
    }, finish_reason="length")
    assert candidate.completion_status == "truncated"
    assert candidate.validation_status == "rejected_incomplete"


def test_actual_chat_rendered_length_is_measured_and_gated():
    candidate = _validate({
        "problem_id": "problem-1", "solution": "work " * 20 + "Final answer: 4",
        "final_answer": "4",
    }, max_tokens=5)
    assert candidate.rendered_tokens is not None
    assert candidate.length_status == "over_limit"
    assert candidate.validation_status == "rejected_length"


def test_malformed_and_wrong_problem_identity_are_rejected():
    malformed = validate_candidate_response(
        "[]", source_record_id="seed-1", expected_problem_id="problem-1", candidate_slot=0,
        request_sha256=SHA, finish_reason="stop", question="question", protocol=PROTOCOL,
        tokenizer=WhitespaceTokenizer(), max_training_tokens=100,
    )
    wrong_identity = _validate({
        "problem_id": "other-problem", "solution": "Final answer: 4", "final_answer": "4",
    })
    assert malformed.format_status == "invalid"
    assert malformed.validation_status == "rejected_format"
    assert wrong_identity.validation_status == "rejected_identity"
