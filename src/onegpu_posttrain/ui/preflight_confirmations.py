"""Short-lived, one-time confirmation capabilities for stage preflight summaries."""

from __future__ import annotations

import secrets
import re
import threading
import time
from dataclasses import dataclass

from .training_preflight import StagePreflightRequest

TOKEN_TTL_SECONDS = 300
MAX_CONFIRMATIONS = 256


@dataclass
class _Confirmation:
    summary: dict
    request: StagePreflightRequest
    expires_at: float


class StagePreflightConfirmations:
    """In-memory confirmations; the token is invalidated before every consume attempt."""

    def __init__(self, *, ttl_seconds: int = TOKEN_TTL_SECONDS, clock=time.monotonic):
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._items: dict[str, _Confirmation] = {}

    def issue(self, request: StagePreflightRequest, summary: dict, *, run_id: str | None = None) -> dict:
        if summary.get("status") != "ready_for_confirmation":
            raise ValueError("blocked preflight cannot be confirmed")
        self._prune()
        token = secrets.token_urlsafe(32)
        confirmation_id = f"confirm-{secrets.token_hex(12)}"
        from ..lineage import new_run_id

        reserved_run_id = run_id or new_run_id("train", label=f"{request.stage}-{request.profile_id}")
        if not re.fullmatch(r"run-[a-z0-9][a-z0-9-]{0,119}", reserved_run_id):
            raise ValueError("预检预留的 Run ID 无效")
        output_location = f"runs/{request.stage}/{request.profile_id}/{reserved_run_id}"
        bound = {
            **summary,
            "confirmation_id": confirmation_id,
            "run_id": reserved_run_id,
            "confirmation_token": token,
            "expires_in_seconds": self._ttl_seconds,
            "output_location": output_location,
            "bound_config_sha256": _digest(summary, request, output_location),
        }
        with self._lock:
            self._items[token] = _Confirmation(bound, request, self._clock() + self._ttl_seconds)
            if len(self._items) > MAX_CONFIRMATIONS:
                oldest = sorted(self._items, key=lambda key: self._items[key].expires_at)
                for expired in oldest[:len(self._items) - MAX_CONFIRMATIONS]:
                    self._items.pop(expired, None)
        return bound

    def consume(self, token: str, request: StagePreflightRequest, current_summary: dict) -> dict:
        if not isinstance(token, str) or not token or len(token) > 200:
            raise ValueError("confirmation token 无效")
        with self._lock:
            saved = self._items.pop(token, None)
        if saved is None:
            raise ValueError("confirmation token 不存在、已过期或已使用")
        if self._clock() >= saved.expires_at:
            raise ValueError("confirmation token 已过期")
        if request != saved.request:
            raise ValueError("确认请求与预检时的阶段、profile、父模型或数据集不一致")
        output_location = saved.summary["output_location"]
        current_digest = _digest(current_summary, request, output_location)
        if current_summary.get("status") != "ready_for_confirmation":
            raise ValueError("当前输入证据已不再满足预检")
        if current_digest != saved.summary["bound_config_sha256"]:
            raise ValueError("配方、父模型、数据集摘要或预检结果已变化，请重新预检并确认")
        return {key: value for key, value in saved.summary.items() if key != "confirmation_token"}

    def _prune(self) -> None:
        now = self._clock()
        with self._lock:
            self._items = {key: item for key, item in self._items.items() if item.expires_at > now}


def _digest(summary: dict, request: StagePreflightRequest, output_location: str) -> str:
    from ..lineage import sha256_json

    return sha256_json({
        "stage": request.stage,
        "profile_id": request.profile_id,
        "dataset_ids": request.dataset_ids,
        "parent_model_id": request.parent_model_id,
        "parameter_overrides": request.parameter_overrides,
        "effective_config_sha256": (summary.get("training_parameters") or {}).get("config_sha256"),
        "recipe_sha256": (summary.get("approved_recipe") or {}).get("sha256"),
        "datasets": [{"id": item.get("dataset_id"), "manifest_sha256": item.get("manifest_sha256"),
                      "content_sha256": item.get("content_sha256")} for item in summary.get("datasets", [])],
        "parent_model": {"id": (summary.get("parent_model") or {}).get("model_id"),
                         "digest": (summary.get("parent_model") or {}).get("digest")},
        "output_location": output_location,
        "status": summary.get("status"),
    })
