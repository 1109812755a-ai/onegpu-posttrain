"""Optional, metadata-only Langfuse tracing for workspace assistant tasks."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

from .contracts import ObservationEvent

_ALLOWED_ATTRIBUTES = frozenset({
    "assistant_profile_id", "model_profile_id", "status", "duration_ms",
    "evidence_count", "evidence_ids", "evidence_versions", "error_code",
    "memory_ids", "context_summary_id", "summary_revision", "source_entity_ids",
    "session_id", "input_tokens", "output_tokens",
    "runtime", "tool_calls", "model_calls", "model_attempts", "tool_names",
    "model_duration_ms", "tool_duration_ms", "history_messages_dropped", "provider_called",
})


def _load_langfuse_environment(workspace: Path) -> None:
    """Load only Langfuse settings from the local env file; never execute it."""
    path = workspace / ".env.local"
    if not path.is_file() or path.is_symlink():
        return
    allowed = {
        "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL", "LANGFUSE_HOST",
        "ONEGPU_ASSISTANT_LANGFUSE_ENABLED",
    }
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key not in allowed or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value and "\x00" not in value:
            os.environ[key] = value


class LangfuseObserver:
    """Send only task identifiers and bounded status metadata, never content."""

    def __init__(self, workspace: str | Path, *, client=None):
        self.workspace = Path(workspace).resolve(strict=True)
        _load_langfuse_environment(self.workspace)
        self.base_url = (os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST") or "").rstrip("/")
        self._client = client
        self._state = "disabled"
        self._detail = "未配置 Langfuse"
        if self._client is not None:
            self._state, self._detail = "configured", "已配置，尚未检查连接"
            return
        if os.environ.get("ONEGPU_ASSISTANT_LANGFUSE_ENABLED", "").strip().casefold() not in {"1", "true", "yes"}:
            self._detail = "未启用 Langfuse；设置 ONEGPU_ASSISTANT_LANGFUSE_ENABLED=true 后启用"
            return
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
        if not (public_key and secret_key and self._valid_base_url(self.base_url)):
            return
        try:
            from langfuse import Langfuse
        except ImportError:
            self._detail = "已配置凭据；未安装 Langfuse SDK（可安装 assistant-observability 可选依赖）"
            return
        try:
            self._client = Langfuse(public_key=public_key, secret_key=secret_key, base_url=self.base_url)
            self._state, self._detail = "configured", "已配置，尚未检查连接"
        except Exception:
            self._state, self._detail = "unavailable", "Langfuse 客户端初始化失败"

    @staticmethod
    def _valid_base_url(value: str) -> bool:
        try:
            parsed = urlsplit(value)
            return parsed.scheme in {"http", "https"} and bool(parsed.netloc) and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment
        except ValueError:
            return False

    @property
    def configured(self) -> bool:
        return self._client is not None

    def status(self) -> dict[str, str | bool]:
        return {"configured": self.configured, "state": self._state, "detail": self._detail,
                "base_url": self.base_url if self._valid_base_url(self.base_url) else ""}

    def check_connection(self) -> dict[str, str | bool]:
        if self._client is None:
            return self.status()
        try:
            authenticated = bool(self._client.auth_check())
        except Exception:
            authenticated = False
        self._state = "available" if authenticated else "unavailable"
        self._detail = "连接正常" if authenticated else "无法连接或认证失败；本地问答不受影响"
        return self.status()

    def emit(self, event: ObservationEvent) -> tuple[str | None, str | None]:
        if self._client is None:
            return None, None
        attributes = {
            key: value for key, value in event.attributes.items()
            if key in _ALLOWED_ATTRIBUTES and value is not None
        }
        attributes.update({"task_id": event.task_id, "event_name": event.event_name})
        trace_id = None
        trace_url = None
        try:
            with self._client.start_as_current_observation(
                as_type="span", name="onegpu.workspace-assistant.task",
                metadata=attributes, version="workspace-assistant-v1",
            ) as observation:
                trace_id = observation.trace_id
                trace_url = self._client.get_trace_url(trace_id=trace_id)
            self._client.flush()
            self._state, self._detail = "available", "连接正常"
            return trace_id, trace_url
        except Exception:
            self._state, self._detail = "unavailable", "Langfuse 不可用；本地任务仍已保存"
            return None, None
