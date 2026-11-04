"""Provider contracts for explicit synthetic-data generation.

The OpenAI-compatible HTTP client is imported only inside ``generate`` so the
core project remains usable without the optional ``synthetic`` extra.
"""

from __future__ import annotations

import os
import re
from typing import Any, Literal
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SyntheticModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ProviderProfile(SyntheticModel):
    """Non-secret provider settings; only the environment variable name is stored."""

    profile_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
    adapter: Literal["fake", "openai_compatible"]
    endpoint: str | None = None
    model: str = Field(min_length=1)
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    supported_parameters: list[Literal["temperature", "max_output_tokens", "seed"]] = Field(
        default_factory=lambda: ["temperature", "max_output_tokens"]
    )
    supports_json_object: bool = True
    revision: str | None = None
    input_cost_per_million_tokens: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    output_cost_per_million_tokens: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, endpoint: str | None) -> str | None:
        if endpoint is None:
            return endpoint
        parts = urlsplit(endpoint)
        sensitive_markers = ("apikey", "token", "auth", "password", "secret", "credential")
        has_sensitive_query = any(
            any(marker in key.lower().replace("_", "").replace("-", "") for marker in sensitive_markers)
            or key.lower() == "key"
            for key, _ in parse_qsl(parts.query)
        )
        if (
            parts.scheme not in {"http", "https"}
            or not parts.netloc
            or parts.username is not None
            or parts.password is not None
            or has_sensitive_query
            or parts.fragment
        ):
            raise ValueError("endpoint 必须是无认证信息的 HTTP(S) URL；凭据请使用 api_key_env")
        return endpoint

    @model_validator(mode="after")
    def validate_profile(self) -> ProviderProfile:
        if len(self.supported_parameters) != len(set(self.supported_parameters)):
            raise ValueError("supported_parameters 不能重复")
        if self.adapter == "openai_compatible":
            if self.endpoint is None or self.api_key_env is None:
                raise ValueError("openai_compatible profile 必须提供 endpoint 与 api_key_env")
        elif self.endpoint is not None or self.api_key_env is not None:
            raise ValueError("fake profile 不接受 endpoint 或 api_key_env")
        return self


class OutputProtocol(SyntheticModel):
    protocol_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    response_format: Literal["json_object"] = "json_object"
    required_fields: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def fields_unique(self) -> OutputProtocol:
        if len(self.required_fields) != len(set(self.required_fields)):
            raise ValueError("required_fields 不能重复")
        return self


class PromptTemplate(SyntheticModel):
    template_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    template: str = Field(min_length=1)
    allowed_variables: list[str] = Field(default_factory=lambda: ["question", "topic"])

    @model_validator(mode="after")
    def validate_template(self) -> PromptTemplate:
        if len(self.allowed_variables) != len(set(self.allowed_variables)):
            raise ValueError("allowed_variables 不能重复")
        if re.search(r"\$\{[^}]+\}|\{\{\s*(?:env|secret|environment)[^}]*\}\}", self.template, re.IGNORECASE):
            raise ValueError("提示词模板不能插值环境变量或秘密")
        variables = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", self.template))
        unknown = variables - set(self.allowed_variables)
        if unknown:
            raise ValueError(f"提示词模板包含未声明变量: {', '.join(sorted(unknown))}")
        return self

    def render(self, values: dict[str, str]) -> str:
        unknown = set(values) - set(self.allowed_variables)
        required = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", self.template))
        missing = required - set(values)
        if unknown or missing:
            raise ValueError("提示词变量与模板声明不匹配")
        return self.template.format_map(values)


class GenerationRequest(SyntheticModel):
    prompt: str = Field(min_length=1)
    system_instruction: str | None = Field(default=None, max_length=12000)
    max_output_tokens: int = Field(ge=1)
    temperature: float | None = Field(default=None, ge=0, le=2, allow_inf_nan=False)
    seed: int | None = Field(default=None, ge=0)
    enable_thinking: bool | None = None
    timeout_seconds: float = Field(default=300, gt=0, allow_inf_nan=False)
    output_protocol: OutputProtocol


class GenerationResponse(SyntheticModel):
    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    provider_request_id: str | None = None


class ProviderError(RuntimeError):
    """Sanitized provider failure facts safe for reports and request ledgers."""

    def __init__(self, category: str, message: str, *, retryable: bool, uncertain: bool,
                 status_code: int | None = None, provider_request_id: str | None = None,
                 timeout_seconds: float | None = None):
        super().__init__(message)
        self.category = category
        self.retryable = retryable
        self.uncertain = uncertain
        self.status_code = status_code
        self.provider_request_id = provider_request_id
        self.timeout_seconds = timeout_seconds


class MissingCredentialError(RuntimeError):
    """A required credential is absent; no provider request was attempted."""


def require_provider_credentials(profile: ProviderProfile) -> None:
    if profile.adapter == "openai_compatible" and not os.environ.get(profile.api_key_env or ""):
        raise MissingCredentialError(f"缺少凭据环境变量: {profile.api_key_env}")


def validate_request_support(profile: ProviderProfile, request: GenerationRequest) -> None:
    requested: set[str] = {"max_output_tokens"}
    if request.temperature is not None:
        requested.add("temperature")
    if request.seed is not None:
        requested.add("seed")
    unsupported = requested - set(profile.supported_parameters)
    if unsupported:
        raise ValueError(f"provider profile 不支持请求参数: {', '.join(sorted(unsupported))}")
    if not profile.supports_json_object:
        raise ValueError("provider profile 不支持声明的 JSON object 输出协议")


class FakeProvider:
    """Deterministic local provider for tests and offline walkthroughs."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses or ['{"problem_id":"fixture","solution":"2+2=4","final_answer":"4"}'])
        self.calls = 0

    def generate(self, profile: ProviderProfile, request: GenerationRequest) -> GenerationResponse:
        if profile.adapter != "fake":
            raise ValueError("FakeProvider 只能使用 fake profile")
        validate_request_support(profile, request)
        self.calls += 1
        text = self._responses.pop(0) if self._responses else self._responses_default()
        return GenerationResponse(text=text, finish_reason="stop")

    @staticmethod
    def _responses_default() -> str:
        return '{"problem_id":"fixture","solution":"2+2=4","final_answer":"4"}'


class OpenAICompatibleProvider:
    """Small chat-completions adapter with lazy optional httpx dependency."""

    def __init__(self, *, transport: Any = None):
        self._transport = transport

    def generate(self, profile: ProviderProfile, request: GenerationRequest) -> GenerationResponse:
        if profile.adapter != "openai_compatible":
            raise ValueError("OpenAICompatibleProvider 需要 openai_compatible profile")
        validate_request_support(profile, request)
        require_provider_credentials(profile)
        api_key = os.environ[profile.api_key_env or ""]

        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - exercised without synthetic extra
            raise RuntimeError("请安装 onegpu-posttrain[synthetic] 以使用 OpenAI-compatible provider") from exc

        messages = []
        if request.system_instruction:
            messages.append({"role": "system", "content": request.system_instruction})
        messages.append({"role": "user", "content": request.prompt})
        payload: dict[str, Any] = {
            "model": profile.model,
            "messages": messages,
            "max_tokens": request.max_output_tokens,
            "response_format": {"type": request.output_protocol.response_format},
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.enable_thinking is not None:
            payload["enable_thinking"] = request.enable_thinking

        try:
            with httpx.Client(transport=self._transport, timeout=request.timeout_seconds) as client:
                response = client.post(
                    profile.endpoint,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                )
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            usage = body.get("usage") or {}
            content = choice["message"]["content"]
        except httpx.TimeoutException:
            raise ProviderError("timeout", "provider request timed out", retryable=True, uncertain=True,
                                timeout_seconds=request.timeout_seconds) from None
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            request_id = (exc.response.headers.get("x-request-id")
                          or exc.response.headers.get("x-dashscope-request-id"))
            if request_id and (len(request_id) > 200 or not re.fullmatch(r"[A-Za-z0-9._:-]+", request_id)):
                request_id = None
            raise ProviderError(
                "rate_limited" if status_code == 429 else "http_error",
                f"provider returned HTTP {status_code}",
                retryable=status_code == 429 or status_code >= 500,
                uncertain=status_code >= 500,
                status_code=status_code,
                provider_request_id=request_id,
            ) from None
        except httpx.HTTPError:
            raise ProviderError("transport", "provider transport failed", retryable=True, uncertain=True) from None
        except (ValueError, KeyError, IndexError, TypeError):
            raise ProviderError(
                "invalid_response", "provider response did not match the expected chat-completions shape",
                retryable=False, uncertain=False,
            ) from None
        if not isinstance(content, str) or api_key in content:
            raise ProviderError(
                "invalid_response", "provider response was invalid or contained credential material; response discarded",
                retryable=False, uncertain=False,
            )
        return GenerationResponse(
            text=content,
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            provider_request_id=(response.headers.get("x-request-id")
                                 or response.headers.get("x-dashscope-request-id")),
        )
