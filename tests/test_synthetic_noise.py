from __future__ import annotations

import copy

import pytest

from onegpu_posttrain.synthetic.noise import inject_sft_answer_noise


def _record(record_id, solution, reference):
    return {
        "id": record_id,
        "raw_problem": "A training seed problem",
        "messages": [
            {"role": "user", "content": "A training seed problem"},
            {"role": "assistant", "content": solution},
        ],
        "prompt_template_id": "fixture-v1",
        "metadata": {"reference_answer": reference, "parent_record_ids": [f"seed-{record_id}"]},
    }


def test_integer_answer_noise_is_deterministic_and_records_parent_and_actual_ratio():
    original = [
        _record("sft-1", r"Compute 2+2. Final answer: 4", "4"),
        _record("sft-2", r"Compute 3+3. Final answer: \boxed{6}", "6"),
    ]
    pristine = copy.deepcopy(original)
    first = inject_sft_answer_noise(original, seed=13, target_ratio=0.5)
    again = inject_sft_answer_noise(original, seed=13, target_ratio=0.5)
    assert first == again
    assert original == pristine
    assert first["report"]["target_count"] == 1
    assert first["report"]["actual_count"] == 1
    changed = [row for row in first["records"] if "answer_noise" in row["metadata"]]
    assert len(changed) == 1
    evidence = changed[0]["metadata"]["answer_noise"]
    assert evidence["parent_record_ids"]
    assert evidence["equivalent_to_reference"] is False
    assert evidence["actual_ratio"] == 0.5
    assert changed[0]["metadata"]["correctness_math_verify"] is False


def test_unsupported_formats_and_missing_reference_are_skipped_without_filling_target():
    result = inject_sft_answer_noise([
        _record("decimal", "Final answer: 0.5", "0"),
        _record("missing", "Final answer: 4", None),
    ], seed=2, target_ratio=1.0)
    assert result["report"]["target_count"] == 2
    assert result["report"]["actual_count"] == 0
    assert result["report"]["skipped"] == {
        "missing_or_unsupported_reference": 1,
        "unsupported_final_answer_format": 1,
    }
    assert all("answer_noise" not in row["metadata"] for row in result["records"])


def test_preference_records_are_never_reordered_or_noised():
    with pytest.raises(ValueError, match="只接受 SFT"):
        inject_sft_answer_noise([{"id": "pair", "chosen": "right", "rejected": "wrong"}],
                                seed=1, target_ratio=1.0)
