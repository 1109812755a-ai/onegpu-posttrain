"""Alibaba Bailian model adapter using the project's registered provider profile."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..evaluation.choice_judge import load_choice_judge_profile
from ..synthetic.providers import (
    GenerationRequest,
    OpenAICompatibleProvider,
    OutputProtocol,
    ProviderError,
    ProviderProfile,
    require_provider_credentials,
)
from ..synthetic.secrets import load_local_provider_credential
from .contracts import ModelRequest, ModelResponse, AssistantAnswer


ASSISTANT_PROFILE_ID = "dashscope-qwen3.6-plus-local"
ASSISTANT_PROMPT_VERSION = "workspace-experiment-assistant-zh-v1"
ASSISTANT_OUTPUT_PROTOCOL = OutputProtocol(
    protocol_id="workspace-experiment-assistant-answer",
    version="v1",
    required_fields=["conclusion", "facts", "interpretations", "limitations", "clarification_options"],
)
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,200}$")


class AssistantModelError(RuntimeError):
    """Sanitized model error safe for task records and UI display."""

    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class BailianAssistantModelAdapter:
    """Calls only the fixed local DashScope profile; callers cannot pass endpoints."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        provider: Any | None = None,
        profile: ProviderProfile | None = None,
        timeout_seconds: float = 90,
    ):
        self.workspace = Path(workspace).resolve(strict=True)
        try:
            self.profile = profile or load_choice_judge_profile(self.workspace)
        except (ValueError, OSError):
            self.profile = None
        if self.profile is not None and self.profile.profile_id != ASSISTANT_PROFILE_ID:
            raise ValueError("助手仅允许使用项目登记的百炼 profile")
        self.provider = provider or OpenAICompatibleProvider()
        self.timeout_seconds = timeout_seconds

    def generate(self, request: ModelRequest) -> ModelResponse:
        if self.profile is None:
            raise AssistantModelError("not_configured", "阿里百炼模型配置缺失或不可用。")
        if request.model_profile_id != self.profile.profile_id:
            raise AssistantModelError("model_profile_mismatch", "助手模型配置与请求身份不一致。")
        try:
            load_local_provider_credential(
                self.profile.api_key_env or "", self.workspace / ".env.local"
            )
            require_provider_credentials(self.profile)
        except Exception as exc:
            # Never include file contents or credential values in a persisted error.
            raise AssistantModelError("not_configured", "阿里百炼凭据未配置或不可用。") from exc

        payload = {
            "prompt_version": ASSISTANT_PROMPT_VERSION,
            "question": request.question,
            "recent_turns": [turn.model_dump(mode="json") for turn in request.recent_turns],
            "evidence": [item.model_dump(mode="json") for item in request.evidence],
            "response_contract": {
                "conclusion": "面向用户的简明结论",
                "facts": "由 evidence_ids 引用证据支持的事实列表",
                "interpretations": "明确标注为分析推断并引用证据的列表",
                "limitations": "证据缺失、旧版、部分完成或不可比限制",
                "clarification_options": "对象不明确时的澄清选项；否则为空列表",
            },
        }
        generation = GenerationRequest(
            prompt=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            system_instruction=request.system_instruction,
            max_output_tokens=request.max_output_tokens,
            temperature=0.2,
            enable_thinking=False,
            timeout_seconds=self.timeout_seconds,
            output_protocol=ASSISTANT_OUTPUT_PROTOCOL,
        )
        try:
            response = self.provider.generate(self.profile, generation)
        except ProviderError as exc:
            raise AssistantModelError(
                exc.category,
                "阿里百炼暂时无法完成回答，请检查连接后重试。" if exc.retryable
                else "阿里百炼返回了无效或不完整的响应。",
                retryable=exc.retryable,
            ) from exc
        except Exception as exc:
            raise AssistantModelError("provider_error", "阿里百炼请求失败，请稍后重试。", retryable=True) from exc
        try:
            answer = AssistantAnswer.model_validate_json(response.text)
        except (ValidationError, ValueError) as exc:
            raise AssistantModelError("invalid_answer", "阿里百炼回答未通过助手响应格式校验。") from exc
        request_id = response.provider_request_id
        if not isinstance(request_id, str) or not _SAFE_REQUEST_ID.fullmatch(request_id):
            request_id = None
        return ModelResponse(
            answer=answer,
            model_profile_id=self.profile.profile_id,
            provider_request_id=request_id,
            input_tokens=response.prompt_tokens,
            output_tokens=response.completion_tokens,
        )

    def is_configured(self) -> bool:
        """Return availability without returning or logging credential material."""
        try:
            if self.profile is None:
                return False
            load_local_provider_credential(
                self.profile.api_key_env or "", self.workspace / ".env.local"
            )
            require_provider_credentials(self.profile)
            return True
        except Exception:
            return False
