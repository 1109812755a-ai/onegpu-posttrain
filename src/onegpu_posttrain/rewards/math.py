from __future__ import annotations

from typing import Any

from ..evaluation.answers import (
    answers_equal,
    extract_answer,
    extract_last_boxed_content,
)


def _completion_text(value: Any) -> str:
    """Normalize the completion shape used by ms-swift reward plugins.

    Current ms-swift custom ORMs receive string completions, but accepting the
    common dict/list variants makes the reward helper safe to unit-test and more
    tolerant of future wrapper changes.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
        return str(value)
    if isinstance(value, list):
        for item in reversed(value):
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                return item["content"]
        return "".join(str(item) for item in value)
    return str(value)



def boxed_format_reward(completion: Any) -> float:
    """Reward the project's frozen final-answer format, ``\\boxed{...}``."""
    return 1.0 if extract_last_boxed_content(_completion_text(completion)) is not None else 0.0


def accuracy_reward(completion: Any, solution: Any) -> float:
    """Outcome reward using the same answer extraction/equivalence path as eval."""
    text = _completion_text(completion)
    pred = extract_answer(text)
    return 1.0 if answers_equal(pred, str(solution)) else 0.0
