"""Bounded OpenAI-compatible tool-call transport for the fixed Bailian editor."""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..schemas import EvalRecord
from ..synthetic.providers import (
    ProviderError,
    ProviderProfile,
    require_provider_credentials,
)
from .isolation import reject_holdout_content
from .provider_profile import PROFILE_ENDPOINT, PROFILE_ID


class ToolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolMessage(ToolModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(max_length=24_000)
    tool_call_id: str | None = Field(default=None, max_length=200)
    tool_calls: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def validate_role(self) -> ToolMessage:
        if (self.role == "tool") != (self.tool_call_id is not None):
            raise ValueError("tool message needs a tool call ID; other roles cannot use one")
        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError("only assistant messages may contain tool calls")
        return self


class ToolCall(ToolModel):
    id: str = Field(min_length=1, max_length=200)
    name: Literal["inspect_tensor", "propose_edit"]
    arguments: dict[str, Any]


class ToolRound(ToolModel):
    content: str = Field(default="", max_length=24_000)
    calls: list[ToolCall] = Field(default_factory=list, max_length=4)
    finish_reason: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    provider_request_id: str | None = None


TOOL_DECLARATIONS: tuple[dict[str, Any], ...] = (
    {"type": "function", "function": {
        "name": "inspect_tensor", "description": "Inspect bounded tensor metadata or values",
        "parameters": {"type": "object", "additionalProperties": False,
                       "properties": {"tensor_name": {"type": "string"},
                                      "offset": {"type": "integer", "minimum": 0},
                                      "length": {"type": "integer", "minimum": 1, "maximum": 32}},
                       "required": ["tensor_name", "offset", "length"]}}},
    {"type": "function", "function": {
        "name": "propose_edit", "description": "Propose a bounded numeric weight edit",
        "parameters": {"type": "object", "additionalProperties": False,
                       "properties": {"tensor_name": {"type": "string"},
                                      "offset": {"type": "integer", "minimum": 0},
                                      "values": {"type": "array", "items": {"type": "number"},
                                                 "minItems": 1, "maxItems": 32},
                                      "parent_sha256": {"type": "string"}},
                       "required": ["tensor_name", "offset", "values", "parent_sha256"]}}},
)


class BailianToolClient:
    def __init__(self, *, transport: Any = None):
        self._transport = transport

    def complete(self, profile: ProviderProfile, messages: list[ToolMessage],
                 holdout: list[EvalRecord], *, max_output_tokens: int = 512,
                 timeout_seconds: float = 45) -> ToolRound:
        if (profile.profile_id != PROFILE_ID or profile.model != "qwen3.8-max"
                or profile.adapter != "openai_compatible" or profile.endpoint != PROFILE_ENDPOINT
                or profile.api_key_env != "ONEGPU_SYNTHETIC_API_KEY"):
            raise ValueError("only the fixed Bailian editor profile can call weight tools")
        if not 1 <= len(messages) <= 12 or not 1 <= max_output_tokens <= 2048 or not 0 < timeout_seconds <= 120:
            raise ValueError("tool request exceeds the configured budget")
        require_provider_credentials(profile)
        api_key = os.environ[profile.api_key_env or ""]
        wire_messages = [message.model_dump(exclude_none=True) for message in messages]
        reject_holdout_content(wire_messages, holdout)
        payload = {"model": profile.model, "messages": wire_messages,
                   "tools": list(TOOL_DECLARATIONS), "tool_choice": "auto",
                   "max_tokens": max_output_tokens, "temperature": 0}
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("install onegpu-posttrain[synthetic] for provider tool calls") from exc
        try:
            with httpx.Client(transport=self._transport, timeout=timeout_seconds) as client:
                response = client.post(profile.endpoint, headers={"Authorization": f"Bearer {api_key}"},
                                       json=payload)
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            message = choice["message"]
            calls = []
            for raw in message.get("tool_calls") or []:
                if raw.get("type") != "function":
                    raise ValueError("unsupported tool call type")
                args = raw["function"]["arguments"]
                if not isinstance(args, str) or len(args) > 4096:
                    raise ValueError("tool arguments exceed limit")
                calls.append(ToolCall(id=raw["id"], name=raw["function"]["name"],
                                      arguments=json.loads(args)))
            usage = body.get("usage") or {}
            content = message.get("content") or ""
            if not isinstance(content, str) or api_key in content or len(calls) > 4:
                raise ValueError("invalid provider response")
            request_id = response.headers.get("x-request-id") or response.headers.get("x-dashscope-request-id")
            if request_id is not None and (len(request_id) > 200 or not request_id.replace("-", "").isalnum()):
                request_id = None
            return ToolRound(content=content, calls=calls, finish_reason=choice.get("finish_reason"),
                             prompt_tokens=usage.get("prompt_tokens"),
                             completion_tokens=usage.get("completion_tokens"),
                             provider_request_id=request_id)
        except httpx.TimeoutException:
            raise ProviderError("timeout", "provider request timed out", retryable=True, uncertain=True,
                                timeout_seconds=timeout_seconds) from None
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise ProviderError("rate_limited" if status == 429 else "http_error",
                                f"provider returned HTTP {status}", retryable=status == 429 or status >= 500,
                                uncertain=status >= 500, status_code=status) from None
        except httpx.HTTPError:
            raise ProviderError("transport", "provider transport failed", retryable=True,
                                uncertain=True) from None
        except (KeyError, ValueError, TypeError, IndexError):
            raise ProviderError("invalid_response", "provider tool response was invalid",
                                retryable=False, uncertain=False) from None
