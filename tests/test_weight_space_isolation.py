"""Practice feedback cannot carry protected final question content."""

import json
from types import SimpleNamespace

import pytest

from onegpu_posttrain.weight_space.isolation import (
    build_practice_feedback,
    reject_holdout_content,
)


def _inputs(tmp_path):
    practice = tmp_path / "practice.jsonl"
    final = tmp_path / "final.jsonl"
    practice.write_text(json.dumps({"id": "practice-1", "problem_id": "practice-1",
                                    "eval_prompt_version": "objective-choice-converter-v1",
                                    "prompt": "Which trigram represents heaven?", "raw_problem": "Which trigram represents heaven?",
                                    "answer": "A", "question_type": "single_choice", "options": ["Qian", "Kun"]}) + "\n",
                        encoding="utf-8")
    final.write_text(json.dumps({"id": "final-1", "problem_id": "final-1",
                                 "eval_prompt_version": "objective-choice-converter-v1",
                                 "prompt": "Protected final question about the eight trigrams",
                                 "raw_problem": "Protected final question about the eight trigrams",
                                 "answer": "B", "question_type": "single_choice", "options": ["Qian", "Kun"]}) + "\n",
                     encoding="utf-8")
    return SimpleNamespace(practice_path=practice, final_path=final, manifest=SimpleNamespace(
        practice=SimpleNamespace(dataset_id="practice-dataset", record_count=1),
        final_holdout=SimpleNamespace(record_count=1)))


def test_feedback_uses_registered_practice_only(tmp_path):
    inputs = _inputs(tmp_path)
    payload = build_practice_feedback(inputs, {"practice-1": "A"})
    assert payload["role"] == "practice"
    assert payload["items"][0]["correct_answer"] == "A"
    assert "Protected final question" not in json.dumps(payload)


def test_feedback_rejects_final_id_and_leaked_question(tmp_path):
    inputs = _inputs(tmp_path)
    with pytest.raises(ValueError, match="registered practice"):
        build_practice_feedback(inputs, {"final-1": "B"})
    with pytest.raises(ValueError, match="final holdout content"):
        build_practice_feedback(inputs, {"practice-1": "Protected final question about the eight trigrams"})


def test_tool_result_rejects_final_fields(tmp_path):
    inputs = _inputs(tmp_path)
    from onegpu_posttrain.data.io import read_jsonl
    from onegpu_posttrain.schemas import EvalRecord

    holdout = [EvalRecord.model_validate(row) for row in read_jsonl(inputs.final_path)]
    with pytest.raises(ValueError, match="final holdout fields"):
        reject_holdout_content({"holdout": {"answer": "B"}}, holdout)
