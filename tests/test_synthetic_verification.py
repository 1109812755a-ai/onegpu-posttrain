from __future__ import annotations

import builtins
from types import ModuleType

from onegpu_posttrain.synthetic.verification import assess_candidate_answer


def test_missing_trusted_reference_stays_unverified():
    assessment = assess_candidate_answer("Final answer: 42", None)
    assert assessment.status == "unverified"
    assert assessment.coverage == "no_reference"


def test_unsupported_reference_type_is_not_guessed():
    assessment = assess_candidate_answer("Final answer: 42", {"value": 42})
    assert assessment.status == "unsupported"
    assert assessment.coverage == "unsupported_reference"


def test_unparseable_answer_is_reported():
    assessment = assess_candidate_answer("I cannot determine the result.", "42")
    assert assessment.status == "unparseable"


def test_exact_answer_is_verified_without_optional_dependency():
    assessment = assess_candidate_answer(r"\boxed{42}", "42")
    assert assessment.status == "correct"
    assert assessment.method == "normalize-answer:v1"


def test_nonliteral_answer_without_optional_verifier_stays_unverified(monkeypatch):
    from onegpu_posttrain.evaluation import verification

    monkeypatch.setattr(verification, "_verification_identity", lambda: (None, None))
    assessment = assess_candidate_answer("Final answer: 1/2", "0.5")
    assert assessment.status == "unverified"
    assert assessment.verifier_available is False


def test_equivalence_verifier_error_is_unparseable(monkeypatch):
    original_import = builtins.__import__

    def with_math_verify(name, *args, **kwargs):
        if name == "math_verify":
            return ModuleType("math_verify")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", with_math_verify)
    assessment = assess_candidate_answer(
        "Final answer: x", "x+0",
        equivalence_verifier=lambda *_: (_ for _ in ()).throw(ValueError("parse")),
    )
    assert assessment.status == "unparseable"
