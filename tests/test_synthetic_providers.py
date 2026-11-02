from __future__ import annotations

import httpx
import pytest
from pydantic import ValidationError

from onegpu_posttrain.synthetic.providers import (
    FakeProvider,
    GenerationRequest,
    OpenAICompatibleProvider,
    OutputProtocol,
    PromptTemplate,
    ProviderError,
    ProviderProfile,
)


def protocol() -> OutputProtocol:
    return OutputProtocol(protocol_id="math-candidate", version="v1", required_fields=[
        "problem_id", "solution", "final_answer",
    ])


def request(**updates) -> GenerationRequest:
    fields = {"prompt": "Solve 2+2", "max_output_tokens": 128, "output_protocol": protocol()}
    fields.update(updates)
    return GenerationRequest(**fields)


def test_fake_provider_is_local_and_deterministic():
    profile = ProviderProfile(profile_id="fake-local", adapter="fake", model="fixture-v1")
    provider = FakeProvider(['{"problem_id":"p1","solution":"2+2=4","final_answer":"4"}'])
    response = provider.generate(profile, request())
    assert response.finish_reason == "stop"
    assert '"final_answer":"4"' in response.text
    assert provider.calls == 1


def test_provider_request_timeout_defaults_to_five_minutes():
    assert request().timeout_seconds == 300


def test_provider_rejects_unsupported_parameters_before_transport():
    profile = ProviderProfile(
        profile_id="fake-no-seed", adapter="fake", model="fixture-v1",
        supported_parameters=["max_output_tokens"],
    )
    provider = FakeProvider()
    with pytest.raises(ValueError, match="temperature"):
        provider.generate(profile, request(temperature=0.2))
    assert provider.calls == 0


def test_openai_profile_rejects_credentials_in_url():
    with pytest.raises(ValidationError, match="api_key_env"):
        ProviderProfile(
            profile_id="bad", adapter="openai_compatible", model="m",
            endpoint="https://user:secret@example.invalid/v1/chat/completions", api_key_env="TEST_KEY",
        )


def test_missing_credential_does_not_send_request(monkeypatch):
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(200, json={"choices": []})

    monkeypatch.delenv("ONEGPU_TEST_SYNTHETIC_KEY", raising=False)
    profile = ProviderProfile(
        profile_id="mock", adapter="openai_compatible", endpoint="https://example.invalid/v1/chat/completions",
        model="mock-model", api_key_env="ONEGPU_TEST_SYNTHETIC_KEY",
    )
    provider = OpenAICompatibleProvider(transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="ONEGPU_TEST_SYNTHETIC_KEY"):
        provider.generate(profile, request())
    assert sent == []


def test_openai_compatible_adapter_sends_declared_request_via_mock_transport(monkeypatch):
    observed = {}

    def handler(incoming):
        observed["auth"] = incoming.headers["Authorization"]
        observed["body"] = incoming.read().decode("utf-8")
        return httpx.Response(
            200,
            headers={"x-request-id": "mock-request-1"},
            json={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 4, "completion_tokens": 2}},
        )

    monkeypatch.setenv("ONEGPU_TEST_SYNTHETIC_KEY", "sentinel-secret")
    profile = ProviderProfile(
        profile_id="mock", adapter="openai_compatible", endpoint="https://example.invalid/v1/chat/completions",
        model="mock-model", api_key_env="ONEGPU_TEST_SYNTHETIC_KEY", supported_parameters=[
            "temperature", "max_output_tokens", "seed",
        ],
    )
    result = OpenAICompatibleProvider(transport=httpx.MockTransport(handler)).generate(
        profile, request(temperature=0.1, seed=7),
    )
    assert observed["auth"] == "Bearer sentinel-secret"
    assert '"seed":7' in observed["body"]
    assert result.provider_request_id == "mock-request-1"
    assert result.prompt_tokens == 4
    assert result.completion_tokens == 2


def test_provider_http_error_does_not_echo_secret_response_body(monkeypatch):
    secret = "sentinel-api-secret"
    monkeypatch.setenv("ONEGPU_TEST_SYNTHETIC_KEY", secret)
    profile = ProviderProfile(
        profile_id="mock", adapter="openai_compatible", endpoint="https://example.invalid/v1/chat/completions",
        model="mock-model", api_key_env="ONEGPU_TEST_SYNTHETIC_KEY",
    )
    provider = OpenAICompatibleProvider(transport=httpx.MockTransport(
        lambda request: httpx.Response(401, headers={"x-dashscope-request-id": "dashscope-req-123"},
                                       text=f"rejected {secret}"),
    ))
    with pytest.raises(RuntimeError) as exc:
        provider.generate(profile, request())
    assert secret not in str(exc.value)
    assert exc.value.status_code == 401
    assert exc.value.provider_request_id == "dashscope-req-123"


def test_provider_timeout_error_reports_configured_deadline_without_exposing_request_body(monkeypatch):
    monkeypatch.setenv("ONEGPU_TEST_SYNTHETIC_KEY", "not-for-logs")
    profile = ProviderProfile(
        profile_id="mock", adapter="openai_compatible", endpoint="https://example.invalid/v1/chat/completions",
        model="mock-model", api_key_env="ONEGPU_TEST_SYNTHETIC_KEY",
    )

    def timeout(_request):
        raise httpx.ReadTimeout("deadline", request=_request)

    provider = OpenAICompatibleProvider(transport=httpx.MockTransport(timeout))
    with pytest.raises(ProviderError) as exc:
        provider.generate(profile, request(timeout_seconds=137))
    assert exc.value.category == "timeout"
    assert exc.value.uncertain is True
    assert exc.value.timeout_seconds == 137


def test_prompt_template_allows_only_declared_variables_and_no_environment_interpolation():
    template = PromptTemplate(
        template_id="sft-math", version="v1", template="Topic: {topic}\nQuestion: {question}",
    )
    assert template.render({"topic": "algebra", "question": "2+2?"}) == "Topic: algebra\nQuestion: 2+2?"
    with pytest.raises(ValidationError, match="环境变量"):
        PromptTemplate(template_id="bad", version="v1", template="Key ${API_KEY}")


def test_core_cli_import_does_not_require_optional_httpx(monkeypatch):
    import builtins
    import importlib

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "httpx" or name.startswith("httpx."):
            raise AssertionError("core import unexpectedly loaded optional httpx")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    importlib.import_module("onegpu_posttrain.cli")
    importlib.import_module("onegpu_posttrain.synthetic.providers")
