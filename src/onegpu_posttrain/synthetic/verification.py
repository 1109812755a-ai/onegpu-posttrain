"""Independent answer checks for synthetic candidates, with explicit coverage."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ..evaluation.answers import answers_equal, extract_answer, normalize_answer
from ..evaluation.verification import AnswerVerificationResult, verify_answer_pairs


class AnswerAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal[
        "correct", "incorrect", "unparseable", "unsupported", "unverified", "timeout", "verifier_error",
    ]
    method: str
    coverage: Literal["reference_and_parser", "reference_only", "no_reference", "unsupported_reference"]
    extracted_answer: str | None = None
    reference_available: bool
    verifier_available: bool
    evidence: str
    verifier: str | None = None
    verifier_version: str | None = None


def assess_candidate_answer(
    solution: str,
    reference_answer: Any | None,
    *,
    answer_parser=extract_answer,
    equivalence_verifier=answers_equal,
) -> AnswerAssessment:
    """Assess a final answer; this does not certify reasoning quality."""
    return assess_candidate_answers_batch(
        [solution], [reference_answer], answer_parser=answer_parser,
        equivalence_verifier=equivalence_verifier,
    )[0]


def _assessment_from_result(
    extracted: str | None,
    reference: str,
    result: AnswerVerificationResult,
) -> AnswerAssessment:
    explanations = {
        "correct": "验证器确认候选答案与可信 reference 等价",
        "incorrect": "验证器完成比较，但未确认候选答案与可信 reference 等价",
        "unparseable": "候选答案或 reference 无法解析",
        "unsupported": "输入类型或答案长度不受当前验证协议支持",
        "unverified": "验证器不可用或缺少可判定证据；不能据此判错",
        "timeout": "验证批次超时；不能据此判错",
        "verifier_error": "验证器或 worker 运行失败；不能据此判错",
    }
    return AnswerAssessment(
        status=result.status, method=result.method, coverage="reference_and_parser",
        extracted_answer=extracted, reference_available=True,
        verifier_available=result.verifier is not None,
        evidence=result.detail or explanations[result.status],
        verifier=result.verifier, verifier_version=result.verifier_version,
    )


def assess_candidate_answers_batch(
    solutions: Sequence[str],
    reference_answers: Sequence[Any | None],
    *,
    answer_parser=extract_answer,
    equivalence_verifier=answers_equal,
) -> list[AnswerAssessment]:
    """Assess a batch with one bounded worker per chunk rather than per answer."""
    if len(solutions) != len(reference_answers):
        raise ValueError("solutions and reference_answers must have equal length")

    prepared: list[tuple[str | None, str | None, AnswerAssessment | None]] = []
    semantic_pairs: list[tuple[str, str]] = []
    semantic_indexes: list[int] = []
    for solution, reference_answer in zip(solutions, reference_answers, strict=True):
        if reference_answer is None:
            prepared.append((None, None, AnswerAssessment(
                status="unverified", method="extract-answer:v1", coverage="no_reference",
                reference_available=False, verifier_available=False,
                evidence="没有可信 reference；不根据模型自报答案判定正确性",
            )))
            continue
        if isinstance(reference_answer, bool) or not isinstance(reference_answer, (str, int, float)):
            prepared.append((None, None, AnswerAssessment(
                status="unsupported", method="extract-answer:v1", coverage="unsupported_reference",
                reference_available=True, verifier_available=False,
                evidence="reference 类型不在当前标量答案验证支持范围",
            )))
            continue
        reference = str(reference_answer)
        try:
            extracted = answer_parser(solution)
        except (ValueError, TypeError, ArithmeticError):
            extracted = None
        except Exception as exc:  # noqa: BLE001 - unexpected parser failures are not answer mismatches.
            prepared.append((None, None, AnswerAssessment(
                status="verifier_error", method="extract-answer:v1", coverage="reference_only",
                reference_available=True, verifier_available=False,
                evidence=f"答案抽取器运行失败：{type(exc).__name__}",
            )))
            continue
        if extracted is None or not str(extracted).strip():
            prepared.append((None, None, AnswerAssessment(
                status="unparseable", method="extract-answer:v1", coverage="reference_only",
                reference_available=True, verifier_available=False,
                evidence="solution 中未能抽取最终答案",
            )))
            continue
        extracted_text = str(extracted)
        if normalize_answer(extracted_text) == normalize_answer(reference):
            prepared.append((extracted_text, reference, AnswerAssessment(
                status="correct", method="normalize-answer:v1", coverage="reference_and_parser",
                extracted_answer=extracted_text, reference_available=True, verifier_available=True,
                evidence="抽取值与可信 reference 规范化后完全相同", verifier="builtin", verifier_version="1",
            )))
            continue
        index = len(prepared)
        prepared.append((extracted_text, reference, None))
        semantic_indexes.append(index)
        semantic_pairs.append((extracted_text, reference))

    if semantic_pairs:
        if equivalence_verifier is answers_equal:
            semantic_results = verify_answer_pairs(semantic_pairs)
        else:
            semantic_results = []
            for prediction, reference in semantic_pairs:
                try:
                    equal = equivalence_verifier(prediction, reference)
                    result = AnswerVerificationResult(
                        status="correct" if equal else "incorrect", method="custom-verifier",
                        verifier=getattr(equivalence_verifier, "__name__", "custom"), verifier_version="custom",
                    )
                except (ValueError, TypeError, ArithmeticError):
                    result = AnswerVerificationResult(
                        status="unparseable", method="custom-verifier", verifier="custom", verifier_version="custom",
                    )
                except Exception as exc:  # noqa: BLE001 - verifier runtime failures remain non-decision outcomes.
                    result = AnswerVerificationResult(
                        status="verifier_error", method="custom-verifier", verifier="custom", verifier_version="custom",
                        detail=f"Custom verifier failed: {type(exc).__name__}",
                    )
                semantic_results.append(result)
        for index, result in zip(semantic_indexes, semantic_results, strict=True):
            extracted, reference, _ = prepared[index]
            prepared[index] = (extracted, reference, _assessment_from_result(extracted, reference or "", result))

    return [item[2] for item in prepared if item[2] is not None]
