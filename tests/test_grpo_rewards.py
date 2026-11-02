from onegpu_posttrain.evaluation.answers import extract_answer
from onegpu_posttrain.rewards.math import (
    accuracy_reward,
    boxed_format_reward,
    extract_last_boxed_content,
)


def test_boxed_format_supports_nested_latex():
    text = r"Reasoning... Therefore \\boxed{\\frac{1}{2}}."
    assert extract_last_boxed_content(text) == r"\\frac{1}{2}"
    assert extract_answer(text) == r"\\frac{1}{2}"
    assert boxed_format_reward(text) == 1.0


def test_boxed_format_missing_is_zero():
    assert boxed_format_reward("Final answer: 4") == 0.0


def test_accuracy_uses_eval_answer_path():
    assert accuracy_reward(r"Work... \\boxed{4}", "4") == 1.0
    assert accuracy_reward(r"Work... \\boxed{5}", "4") == 0.0
