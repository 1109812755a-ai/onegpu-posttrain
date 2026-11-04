"""Editor turns stay bounded and use only verified practice records."""

import json
from types import SimpleNamespace

from onegpu_posttrain.synthetic.providers import ProviderError, ProviderProfile
from onegpu_posttrain.weight_space.editor_loop import (
    run_editor_iteration,
    run_pinned_iteration,
)
from onegpu_posttrain.weight_space.tool_provider import ToolCall, ToolRound


def _inputs(tmp_path):
    practice = tmp_path / "practice.jsonl"
    final = tmp_path / "final.jsonl"
    practice.write_text(json.dumps({"id": "practice-1", "problem_id": "practice-1",
                                    "eval_prompt_version": "objective-choice-converter-v1",
                                    "prompt": "Practice question about heaven", "raw_problem": "Practice question about heaven",
                                    "answer": "A", "question_type": "single_choice", "options": ["Qian", "Kun"]}) + "\n",
                        encoding="utf-8")
    final.write_text(json.dumps({"id": "final-1", "problem_id": "final-1",
                                 "eval_prompt_version": "objective-choice-converter-v1",
                                 "prompt": "Protected final question about trigrams",
                                 "raw_problem": "Protected final question about trigrams",
                                 "answer": "B", "question_type": "single_choice", "options": ["Qian", "Kun"]}) + "\n",
                     encoding="utf-8")
    return SimpleNamespace(practice_path=practice, final_path=final, manifest=SimpleNamespace(
        practice=SimpleNamespace(dataset_id="practice-dataset", record_count=1),
        final_holdout=SimpleNamespace(record_count=1)))


def _profile():
    return ProviderProfile(profile_id="fixture", adapter="fake", model="qwen3.8-max")


def test_editor_loop_sends_practice_and_bounded_tool_result(tmp_path):
    class FakeClient:
        def __init__(self):
            self.calls = 0

        def complete(self, _profile, messages, _holdout, **_kwargs):
            self.calls += 1
            joined = json.dumps([item.model_dump() for item in messages])
            assert "Protected final question" not in joined
            assert "final-1" not in joined
            if self.calls == 1:
                return ToolRound(calls=[ToolCall(id="call-1", name="inspect_tensor",
                                                 arguments={"tensor_name": "w", "offset": 0, "length": 1})],
                                 prompt_tokens=10, completion_tokens=5, finish_reason="tool_calls")
            assert messages[-1].role == "tool"
            assert json.loads(messages[-1].content) == {"values": [0.2]}
            return ToolRound(content="Completed", prompt_tokens=12, completion_tokens=4)

    client = FakeClient()
    result = run_editor_iteration(_inputs(tmp_path), {"practice-1": "A"}, _profile(),
                                  lambda _: {"values": [0.2]}, client=client)
    assert result.status == "completed"
    assert (result.provider_calls, result.tool_calls, result.prompt_tokens, result.completion_tokens) == (2, 1, 22, 9)


def test_editor_loop_stops_at_provider_budget(tmp_path):
    class FakeClient:
        def complete(self, *_args, **_kwargs):
            return ToolRound(calls=[ToolCall(id="call-1", name="inspect_tensor", arguments={})])

    result = run_editor_iteration(_inputs(tmp_path), {"practice-1": "A"}, _profile(),
                                  lambda _: {"values": [0.2]}, client=FakeClient(), max_provider_calls=1)
    assert result.status == "failed" and result.error_category == "budget"


def test_editor_loop_sanitizes_provider_failure(tmp_path):
    class FakeClient:
        def complete(self, *_args, **_kwargs):
            raise ProviderError("rate_limited", "secret contents", retryable=True, uncertain=False)

    result = run_editor_iteration(_inputs(tmp_path), {"practice-1": "A"}, _profile(),
                                  lambda _: {"values": [0.2]}, client=FakeClient())
    assert result.status == "failed" and result.error_category == "rate_limited"
    assert "secret contents" not in str(result)


def test_pinned_iteration_rejects_stale_parent_before_provider_call(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path)
    inputs.model_path = tmp_path
    inputs.manifest.target = SimpleNamespace(checkpoint_sha256="a" * 64)
    monkeypatch.setattr("onegpu_posttrain.weight_space.editor_loop.sha256_directory", lambda _: "b" * 64)

    class FakeClient:
        def complete(self, *_args, **_kwargs):
            raise AssertionError("provider should not be called")

    import pytest

    with pytest.raises(ValueError, match="stale"):
        run_pinned_iteration(inputs, {"practice-1": "A"}, _profile(), client=FakeClient())
