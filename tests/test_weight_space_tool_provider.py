"""Tool-call wire fixtures and protected prompt boundaries."""

import json

import httpx
import pytest

from onegpu_posttrain.schemas import EvalRecord
from onegpu_posttrain.synthetic.providers import ProviderError, ProviderProfile
from onegpu_posttrain.weight_space.provider_profile import PROFILE_ENDPOINT, PROFILE_ID
from onegpu_posttrain.weight_space.tool_provider import BailianToolClient, ToolMessage


def _profile():
    return ProviderProfile(profile_id=PROFILE_ID, adapter="openai_compatible",
                           endpoint=PROFILE_ENDPOINT, model="qwen3.8-max",
                           api_key_env="ONEGPU_SYNTHETIC_API_KEY")


def _holdout():
    return [EvalRecord(id="final-1", problem_id="final-1",
                       prompt="Protected final question about trigrams",
                       raw_problem="Protected final question about trigrams",
                       eval_prompt_version="objective-choice-converter-v1",
                       question_type="single_choice", options=["Qian", "Kun"], answer="B")]


def test_tool_request_serializes_fixed_declarations_and_parses_call(monkeypatch):
    monkeypatch.setenv("ONEGPU_SYNTHETIC_API_KEY", "fixture-secret")

    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "qwen3.8-max"
        assert [tool["function"]["name"] for tool in body["tools"]] == ["inspect_tensor", "propose_edit"]
        assert body["messages"] == [{"role": "user", "content": "practice feedback"}]
        return httpx.Response(200, json={"choices": [{"message": {
            "content": None, "tool_calls": [{"id": "call-1", "type": "function",
                                          "function": {"name": "inspect_tensor",
                                                       "arguments": '{"tensor_name":"model.layers.0.weight","offset":0,"length":2}'}}]},
            "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 12, "completion_tokens": 7}})

    client = BailianToolClient(transport=httpx.MockTransport(handler))
    result = client.complete(_profile(), [ToolMessage(role="user", content="practice feedback")], _holdout())
    assert result.calls[0].arguments["length"] == 2
    assert result.prompt_tokens == 12


def test_tool_request_rejects_holdout_before_network(monkeypatch):
    monkeypatch.setenv("ONEGPU_SYNTHETIC_API_KEY", "fixture-secret")
    client = BailianToolClient(transport=httpx.MockTransport(lambda _: pytest.fail("network called")))
    with pytest.raises(ValueError, match="final holdout"):
        client.complete(_profile(), [ToolMessage(role="user", content="final-1")], _holdout())


def test_unknown_tool_call_is_sanitized(monkeypatch):
    monkeypatch.setenv("ONEGPU_SYNTHETIC_API_KEY", "fixture-secret")
    response = httpx.Response(200, json={"choices": [{"message": {"content": "",
        "tool_calls": [{"id": "call-2", "type": "function",
                        "function": {"name": "arbitrary_python", "arguments": "{}"}}]}}]})
    client = BailianToolClient(transport=httpx.MockTransport(lambda _: response))
    with pytest.raises(ProviderError, match="provider tool response was invalid"):
        client.complete(_profile(), [ToolMessage(role="user", content="practice")], _holdout())


def test_tool_result_message_is_serialized_with_call_id(monkeypatch):
    monkeypatch.setenv("ONEGPU_SYNTHETIC_API_KEY", "fixture-secret")

    def handler(request):
        messages = json.loads(request.content)["messages"]
        assert messages[1]["role"] == "assistant" and messages[1]["tool_calls"][0]["id"] == "call-1"
        assert messages[2] == {"role": "tool", "content": '{"values":[1.0]}', "tool_call_id": "call-1"}
        return httpx.Response(200, json={"choices": [{"message": {"content": "done"}}]})

    messages = [ToolMessage(role="user", content="practice"),
                ToolMessage(role="assistant", content="", tool_calls=[{"id": "call-1"}]),
                ToolMessage(role="tool", content='{"values":[1.0]}', tool_call_id="call-1")]
    result = BailianToolClient(transport=httpx.MockTransport(handler)).complete(_profile(), messages, _holdout())
    assert result.content == "done"
