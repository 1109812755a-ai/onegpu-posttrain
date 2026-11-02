from onegpu_posttrain.evaluation.choice_accuracy import choice_comparability_key, evaluate_choice_records


class FixtureGenerator:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def generate(self, prompt):
        return next(self.outputs)


def _record(kind, answer, options=None):
    return {"id": f"id-{kind}", "problem_id": f"problem-{kind}", "raw_problem": "q",
            "prompt": "q", "answer": answer, "eval_prompt_version": "test-v1",
            "question_type": kind, "options": options or [],
            "metadata": {"dataset_role": "evaluation_only"}}


def test_choice_accuracy_exact_scores_all_question_types_including_f():
    rows = [
        _record("single_choice", "B", ["a", "b", "c"]),
        _record("multiple_choice", "A,F", list("abcdef")),
        _record("true_false", "true"),
        _record("single_choice", "A", ["a", "b"]),
    ]
    result = evaluate_choice_records(rows, FixtureGenerator(["B. explanation", "答案：F、A", "✅", "无法判断"]))
    assert result["sample_count"] == 4
    assert result["metrics"]["single_choice"] == {
        "correct": 1, "denominator": 2, "parse_failures": 1, "accuracy": 0.5, "status": "measured"}
    assert result["metrics"]["multiple_choice"]["correct"] == 1
    assert result["metrics"]["true_false"]["correct"] == 1
    assert result["metrics"]["overall"]["accuracy"] == 0.75
    assert result["metrics"]["overall"]["parse_failures"] == 1


def test_multiple_choice_requires_exact_set_and_does_not_reward_partial_match():
    result = evaluate_choice_records(
        [_record("multiple_choice", "A,B", list("abc"))], FixtureGenerator(["答案：A"]))
    metric = result["metrics"]["multiple_choice"]
    assert metric["correct"] == 0
    assert metric["denominator"] == 1
    assert metric["accuracy"] == 0


def test_comparability_identity_changes_with_suite_or_coverage():
    key = choice_comparability_key("a" * 64, 20)
    assert key != choice_comparability_key("b" * 64, 20)
    assert key != choice_comparability_key("a" * 64, 19)
