from __future__ import annotations

import hashlib
import json
import sys
from types import ModuleType

from onegpu_posttrain.data.decontaminate import BenchmarkIndex
from onegpu_posttrain.data.policy import DataPolicy, PolicyRules, PolicyTokenizer
from onegpu_posttrain.data.targeted import DataRoleRegistry, RoleRecord
from onegpu_posttrain.data.tokenization import WhitespaceTokenizer
from onegpu_posttrain.evaluation import verification
from onegpu_posttrain.synthetic.candidates import validate_candidate_response
from onegpu_posttrain.synthetic.policy_export import (
    CandidatePolicyInput,
    export_candidates_with_policy,
)
from onegpu_posttrain.synthetic.providers import OutputProtocol

SHA = "a" * 64
PROTOCOL = OutputProtocol(
    protocol_id="candidate", version="v1", required_fields=["problem_id", "solution", "final_answer"],
)
QUESTION = "A fresh algebra seed problem: what is two plus two?"


def _candidate(solution: str, final_answer: str, *, slot: int):
    response = json.dumps({
        "problem_id": "seed-problem", "solution": solution, "final_answer": final_answer,
    })
    return validate_candidate_response(
        response, source_record_id="seed-record", expected_problem_id="seed-problem",
        candidate_slot=slot, request_sha256=hashlib.sha256(f"request-{slot}".encode()).hexdigest(),
        finish_reason="stop", question=QUESTION, protocol=PROTOCOL,
        tokenizer=WhitespaceTokenizer(), max_training_tokens=128,
    )


def _export(stage, candidates, reference="4", *, protected_text="An unrelated protected benchmark question"):
    registry = DataRoleRegistry(records=[RoleRecord(
        dataset_id="seed-dataset", dataset_sha256=SHA, record_id="seed-record",
        problem_id="seed-problem", role="train_seed",
    )])
    policy = DataPolicy(stage=stage, tokenizer=PolicyTokenizer(test_only=True),
                        rules=PolicyRules(min_length_ratio=0.5, max_length_ratio=2.0) if stage == "preference" else PolicyRules())
    return export_candidates_with_policy(
        [CandidatePolicyInput(candidate=item, question=QUESTION, reference_answer=reference)
         for item in candidates],
        policy=policy, tokenizer=WhitespaceTokenizer(), registry=registry,
        protected_index=BenchmarkIndex([("holdout", protected_text)]),
        protected_index_id="index-v1", protected_index_sha256=SHA,
    )


def test_sft_exports_verified_candidate_with_candidate_and_seed_lineage():
    result = _export("sft", [
        _candidate("2+2=4. Final answer: 4", "4", slot=0),
        _candidate("2+2=5. Final answer: 5", "5", slot=1),
    ])
    assert result["selected_count"] == 1
    row = result["records"][0]
    assert row["metadata"]["parent_record_ids"] == ["seed-record"]
    assert len(row["metadata"]["synthetic_candidate_ids"]) == 1
    assert row["metadata"]["answer_assessments"][0]["status"] == "correct"


def test_preference_needs_same_problem_known_correct_and_wrong_sides(monkeypatch):
    # Keep the preference contract deterministic without requiring the optional extra.
    verifier = ModuleType("math_verify")
    verifier.parse = lambda value: value
    verifier.verify = lambda gold, pred: gold == pred
    monkeypatch.setitem(sys.modules, "math_verify", verifier)
    monkeypatch.setattr(verification, "_verification_identity", lambda: ("math-verify", "test"))
    monkeypatch.setattr(
        verification, "_run_bounded_batch",
        lambda pairs, _timeout: ["correct" if p.prediction == p.reference else "incorrect" for p in pairs],
    )
    result = _export("preference", [
        _candidate("2+2=4. Final answer: 4", "4", slot=0),
        _candidate("2+2=5. Final answer: 5", "5", slot=1),
    ])
    assert result["selected_count"] == 1
    row = result["records"][0]
    assert row["chosen"].endswith("4")
    assert row["rejected"].endswith("5")
    assert len(row["metadata"]["synthetic_candidate_ids"]) == 2
    assert row["metadata"]["parent_record_ids"] == ["seed-record"]


def test_preference_does_not_fabricate_missing_wrong_or_unknown_side():
    only_correct = _export("preference", [_candidate("2+2=4. Final answer: 4", "4", slot=0)])
    unknown_reference = _export("preference", [
        _candidate("2+2=4. Final answer: 4", "4", slot=0),
        _candidate("2+2=5. Final answer: 5", "5", slot=1),
    ], reference=None)
    assert only_correct["records"] == []
    assert only_correct["rejected"]["no_known_wrong_candidate"] == 1
    assert unknown_reference["records"] == []
    assert unknown_reference["rejected"]["no_known_correct_candidate"] == 1


def test_duplicate_candidate_and_protected_seed_text_are_rejected():
    candidate = _candidate("2+2=4. Final answer: 4", "4", slot=0)
    duplicate = _export("sft", [candidate, candidate])
    contaminated = _export("sft", [candidate], protected_text=QUESTION)
    assert duplicate["rejected"]["duplicate_candidate"] == 1
    assert contaminated["records"] == []
    assert contaminated["rejected"]["protected_text_match"] == 1
