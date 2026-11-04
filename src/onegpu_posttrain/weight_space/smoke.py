"""Explicit, minimal Bailian editor connectivity check."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..synthetic.providers import MissingCredentialError, ProviderError
from ..synthetic.secrets import load_local_provider_credential
from .provider_profile import load_editor_profile
from .tool_provider import BailianToolClient, ToolMessage


def live_editor_smoke(workspace: str | Path, *, client: BailianToolClient | None = None) -> dict[str, Any]:
    root = Path(workspace).resolve(strict=True)
    profile = load_editor_profile(root)
    start = time.monotonic()
    try:
        load_local_provider_credential(profile.api_key_env or "", root / ".env.local")
        result = (client or BailianToolClient()).complete(
            profile, [ToolMessage(role="user", content="请只回复 OK。")], [],
            max_output_tokens=16, timeout_seconds=30,
        )
        return {"model": profile.model, "status": "success", "latency_seconds": round(time.monotonic() - start, 3),
                "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
                "tool_calls": len(result.calls), "provider_request_id": result.provider_request_id}
    except MissingCredentialError:
        return {"model": profile.model, "status": "failed", "failure_category": "missing_credential",
                "latency_seconds": round(time.monotonic() - start, 3)}
    except ProviderError as exc:
        return {"model": profile.model, "status": "failed", "failure_category": exc.category,
                "latency_seconds": round(time.monotonic() - start, 3)}
