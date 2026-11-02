"""Export verified generated candidates through the existing data-policy selector."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from ..data.decontaminate import BenchmarkIndex
from ..data.policy import DataPolicy
from ..data.policy_build import observe_candidates, select_record
from ..data.roles import check_generated_candidate
from ..data.targeted import DataRoleRegistry
from ..data.tokenization import TokenizerBackend
from .candidates import CandidateRecord
from .verification import AnswerAssessment, assess_candidate_answer


@dataclass(frozen=True)
class CandidatePolicyInput:
    candidate: CandidateRecord
    question: str
    reference_answer: Any | None


def export_candidates_with_policy(
    inputs: list[CandidatePolicyInput],
    *,
    policy: DataPolicy,
    tokenizer: TokenizerBackend,
    registry: DataRoleRegistry,
    protected_index: BenchmarkIndex,
    protected_index_id: str,
    protected_index_sha256: str,
) -> dict[str, Any]:
    """Select stage data with the same rules as existing imported data.

    Candidate correctness is independently assessed against a train-source
    reference. Unknown correctness is preserved for exploratory SFT only; it
    is never manufactured into a Preference side.
    """
    roles = {record.record_id: record for record in registry.records}
    grouped: dict[str, list[tuple[CandidatePolicyInput, AnswerAssessment]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    seen_responses: set[tuple[str, str]] = set()
    for item in inputs:
        candidate = item.candidate
        source = roles.get(candidate.source_record_id)
        if source is None or source.role != "train_seed" or source.problem_id != candidate.problem_id:
            rejected["unregistered_or_non_train_seed_ancestor"] += 1
            continue
        if candidate.validation_status != "candidate_ready_for_quality_review":
            rejected[candidate.validation_status] += 1
            continue
        if candidate.answer_consistency == "mismatch":
            rejected["final_answer_solution_mismatch"] += 1
            continue
        duplicate_key = (candidate.source_record_id, candidate.response_summary.response_sha256)
        if duplicate_key in seen_responses:
            rejected["duplicate_candidate"] += 1
            continue
        seen_responses.add(duplicate_key)
        assessment = assess_candidate_answer(item.candidate.solution or "", item.reference_answer)
        grouped[candidate.source_record_id].append((item, assessment))

    records: list[dict[str, Any]] = []
    diagnostics: Counter[str] = Counter()
    lengths: dict[str, list[int]] = {}
    assessment_counts: Counter[str] = Counter()
    for source_id, pairs in sorted(grouped.items()):
        source = roles[source_id]
        question = pairs[0][0].question
        if any(pair[0].question != question for pair in pairs):
            rejected["candidate_question_conflict"] += len(pairs)
            continue
        if any(pair[0].candidate.problem_id != source.problem_id for pair in pairs):
            rejected["candidate_problem_identity_conflict"] += len(pairs)
            continue
        assessment_counts.update(pair[1].status for pair in pairs)
        row = {
            "problem": question,
            "answer": str(pairs[0][0].reference_answer) if pairs[0][0].reference_answer is not None else None,
            "parent_record_id": source_id,
            "source": "targeted_synthetic",
            "uuid": source_id,
            "generations": [pair[0].candidate.solution or "" for pair in pairs],
            "candidate_ids": [pair[0].candidate.candidate_id for pair in pairs],
            "correctness_math_verify": [
                True if pair[1].status == "correct" else False if pair[1].status == "incorrect" else None
                for pair in pairs
            ],
            "is_reasoning_complete": [pair[0].candidate.completion_status == "complete" for pair in pairs],
            "finish_reasons": [pair[0].candidate.finish_reason or "unknown" for pair in pairs],
        }
        pid = source.problem_id
        observations = observe_candidates(row, question, tokenizer)
        record, reason = select_record(policy, row, pid, question, observations, diagnostics, lengths)
        if reason:
            rejected[reason] += 1
            continue
        assert record is not None
        metadata = record.setdefault("metadata", {})
        metadata["parent_record_ids"] = [source_id]
        candidate_by_id = {pair[0].candidate.candidate_id: pair[1] for pair in pairs}
        metadata["answer_assessments"] = [
            {"candidate_id": candidate_id, "status": assessment.status,
             "method": assessment.method, "coverage": assessment.coverage}
            for candidate_id in metadata.get("synthetic_candidate_ids", [])
            if (assessment := candidate_by_id.get(candidate_id)) is not None
        ]
        check = check_generated_candidate(
            record,
            registry=registry,
            protected_index=protected_index,
            protected_index_id=protected_index_id,
            protected_index_sha256=protected_index_sha256,
        )
        if not check.accepted:
            rejected.update(check.reasons)
            continue
        records.append(record)

    # The shared policy selector is deterministic by problem/content. Refuse
    # any identity collision rather than silently deduplicating or overwriting.
    record_ids = [record["id"] for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("policy export produced duplicate canonical record IDs")
    return {
        "stage": policy.stage,
        "policy_id": policy.policy_id,
        "policy_config_sha256": policy.config_sha256,
        "records": records,
        "selected_count": len(records),
        "assessment_counts": dict(assessment_counts),
        "rejected": dict(rejected),
        "selection_diagnostics": dict(diagnostics),
        "lengths": lengths,
        "contamination_index_id": protected_index_id,
        "contamination_index_sha256": protected_index_sha256,
        "candidate_parent_ids": sorted(grouped),
        "candidate_content_sha256": hashlib.sha256(
            "\n".join(sorted(item.candidate.response_summary.response_sha256 for item in inputs)).encode()
        ).hexdigest(),
    }
