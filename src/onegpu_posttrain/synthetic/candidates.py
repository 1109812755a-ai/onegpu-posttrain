"""Structural validation for generated candidates; never claims correctness."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..data.tokenization import TokenizerBackend
from ..evaluation.answers import answers_equal, extract_answer
from .providers import OutputProtocol


class CandidateResponseSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_characters: int = Field(ge=0)
    returned_fields: list[str] = Field(default_factory=list)
    solution_characters: int | None = Field(default=None, ge=0)
    final_answer_characters: int | None = Field(default=None, ge=0)


class CandidateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal["1.0"] = "1.0"
    candidate_id: str = Field(pattern=r"^candidate-[0-9a-f]{20}$")
    source_record_id: str = Field(min_length=1)
    problem_id: str = Field(min_length=1)
    candidate_slot: int = Field(ge=0)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_summary: CandidateResponseSummary
    finish_reason: str | None = None
    completion_status: Literal["complete", "truncated", "unknown"]
    solution: str | None = None
    final_answer: str | None = None
    answer_consistency: Literal["match", "mismatch", "unparseable", "unverified"]
    correctness_status: Literal["unverified"] = "unverified"
    rendered_tokens: int | None = Field(default=None, ge=0)
    max_training_tokens: int | None = Field(default=None, ge=1)
    length_status: Literal["within_limit", "over_limit", "unavailable"]
    format_status: Literal["valid", "invalid"]
    validation_status: Literal[
        "candidate_ready_for_quality_review", "rejected_format", "rejected_identity",
        "rejected_incomplete", "rejected_length",
    ]


def validate_candidate_response(
    response_text: str,
    *,
    source_record_id: str,
    expected_problem_id: str,
    candidate_slot: int,
    request_sha256: str,
    finish_reason: str | None,
    question: str,
    protocol: OutputProtocol,
    tokenizer: TokenizerBackend | None,
    max_training_tokens: int | None,
) -> CandidateRecord:
    """Parse and classify one response without treating it as verified data."""
    response_sha256 = hashlib.sha256(response_text.encode("utf-8")).hexdigest()
    candidate_id = f"candidate-{hashlib.sha256((request_sha256 + response_sha256).encode()).hexdigest()[:20]}"
    try:
        payload = json.loads(response_text)
        if not isinstance(payload, dict):
            raise TypeError("candidate response must be a JSON object")
        if any(not isinstance(payload.get(name), str) or not payload[name].strip()
               for name in protocol.required_fields):
            raise ValueError("required candidate fields must be non-empty strings")
        problem_id = payload.get("problem_id")
        if not isinstance(problem_id, str) or not problem_id.strip():
            raise ValueError("problem_id is required")
        solution = payload.get("solution")
        final_answer = payload.get("final_answer")
        if not isinstance(solution, str) or not isinstance(final_answer, str):
            raise TypeError("solution and final_answer are required by the candidate schema")
        format_status = "valid"
    except (json.JSONDecodeError, TypeError, ValueError):
        payload = {}
        problem_id = expected_problem_id
        solution = None
        final_answer = None
        format_status = "invalid"

    if format_status == "invalid":
        answer_consistency = "unverified"
    else:
        parsed_answer = extract_answer(solution)
        if parsed_answer is None:
            answer_consistency = "unparseable"
        else:
            answer_consistency = "match" if answers_equal(parsed_answer, final_answer) else "mismatch"

    completion_status: Literal["complete", "truncated", "unknown"]
    if finish_reason and finish_reason.lower() in {"length", "max_tokens", "max_output_tokens"}:
        completion_status = "truncated"
    elif finish_reason and finish_reason.lower() in {"stop", "end_turn", "completed"}:
        completion_status = "complete"
    else:
        completion_status = "unknown"

    rendered_tokens = None
    if tokenizer is not None and solution is not None:
        rendered_tokens = tokenizer.count_messages([
            {"role": "user", "content": question},
            {"role": "assistant", "content": solution},
        ])
    if rendered_tokens is None or max_training_tokens is None:
        length_status: Literal["within_limit", "over_limit", "unavailable"] = "unavailable"
    else:
        length_status = "within_limit" if rendered_tokens <= max_training_tokens else "over_limit"

    if format_status == "invalid":
        validation_status = "rejected_format"
    elif problem_id != expected_problem_id:
        validation_status = "rejected_identity"
    elif completion_status != "complete":
        validation_status = "rejected_incomplete"
    elif length_status == "over_limit":
        validation_status = "rejected_length"
    else:
        validation_status = "candidate_ready_for_quality_review"

    return CandidateRecord(
        candidate_id=candidate_id,
        source_record_id=source_record_id,
        problem_id=expected_problem_id,
        candidate_slot=candidate_slot,
        request_sha256=request_sha256,
        response_summary=CandidateResponseSummary(
            response_sha256=response_sha256,
            response_characters=len(response_text),
            returned_fields=sorted(payload) if isinstance(payload, dict) else [],
            solution_characters=len(solution) if solution is not None else None,
            final_answer_characters=len(final_answer) if final_answer is not None else None,
        ),
        finish_reason=finish_reason,
        completion_status=completion_status,
        solution=solution,
        final_answer=final_answer,
        answer_consistency=answer_consistency,
        rendered_tokens=rendered_tokens,
        max_training_tokens=max_training_tokens,
        length_status=length_status,
        format_status=format_status,
        validation_status=validation_status,
    )
