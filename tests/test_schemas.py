from pathlib import Path

import pytest

from onegpu_posttrain.data.normalize import normalize_problem, problem_id
from onegpu_posttrain.data.validate import validate_jsonl
from onegpu_posttrain.reporting.data_report import length_summary

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("filename,stage", [
    ("cpt.jsonl", "cpt"),
    ("sft.jsonl", "sft"),
    ("preference.jsonl", "preference"),
    ("grpo.jsonl", "grpo"),
    ("eval.jsonl", "eval"),
])
def test_example_schema(filename, stage):
    result = validate_jsonl(ROOT / "data" / "examples" / filename, stage)
    assert result.ok, result.errors
    assert result.records >= 1


def test_problem_id_is_normalization_stable():
    assert normalize_problem("  x +  1 = 2 ") == "x + 1 = 2"
    assert problem_id("x + 1 = 2") == problem_id("  x +  1 = 2 ")


def test_length_summary():
    s = length_summary([1, 2, 3, 4, 5])
    assert s["count"] == 5
    assert s["p50"] == 3
