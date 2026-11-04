"""Build editor feedback only from the pinned practice set."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..data.io import read_jsonl
from ..evaluation.choice_accuracy import _prompt
from ..schemas import EvalRecord
from .registry import ResolvedInputs


def reject_holdout_content(value: Any, holdout: list[EvalRecord]) -> None:
    """Reject final question IDs or full stems in any outbound prompt/tool result."""
    protected = {item.id for item in holdout}
    protected.update(item.problem_id for item in holdout)
    protected.update(item.prompt for item in holdout if len(item.prompt) >= 12)
    protected.update(item.raw_problem for item in holdout if len(item.raw_problem) >= 12)
    protected.discard(None)
    if isinstance(value, str):
        if any(marker in value for marker in protected):
            raise ValueError("final holdout content cannot enter an iteration request")
    elif isinstance(value, Mapping):
        if any(key in {"final_holdout", "holdout", "final_questions", "final_predictions"}
               for key in value):
            raise ValueError("final holdout fields cannot enter an iteration request")
        for key, item in value.items():
            reject_holdout_content(key, holdout)
            reject_holdout_content(item, holdout)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_holdout_content(item, holdout)


def build_practice_feedback(inputs: ResolvedInputs, predictions: Mapping[str, str]) -> dict[str, Any]:
    """Resolve selected IDs against canonical practice rows; reject arbitrary question text."""
    practice = {rec.id: rec for rec in (
        EvalRecord.model_validate(row) for row in read_jsonl(inputs.practice_path))}
    holdout = [EvalRecord.model_validate(row) for row in read_jsonl(inputs.final_path)]
    if len(practice) != inputs.manifest.practice.record_count or len(holdout) != inputs.manifest.final_holdout.record_count:
        raise ValueError("practice or final dataset count changed")
    if not predictions or len(predictions) > 30 or set(predictions) - set(practice):
        raise ValueError("practice feedback must reference 1 to 30 registered practice questions")
    items = []
    for record_id, output in predictions.items():
        if not isinstance(output, str) or len(output) > 2048:
            raise ValueError("practice prediction exceeds the output limit")
        record = practice[record_id]
        items.append({"id": record.id, "question": _prompt(record),
                      "correct_answer": record.answer, "model_output": output})
    result = {"dataset_id": inputs.manifest.practice.dataset_id, "role": "practice", "items": items}
    reject_holdout_content(result, holdout)
    return result
