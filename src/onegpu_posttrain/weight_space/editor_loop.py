"""Bounded, auditable editor turns with practice-only context."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..data.io import read_jsonl
from ..lineage import sha256_directory
from ..schemas import EvalRecord
from ..synthetic.providers import MissingCredentialError, ProviderError, ProviderProfile
from .isolation import build_practice_feedback, reject_holdout_content
from .registry import ResolvedInputs
from .tool_provider import BailianToolClient, ToolCall, ToolMessage
from .weight_tools import dispatch_weight_tool


@dataclass(frozen=True)
class IterationResult:
    status: str
    provider_calls: int
    tool_calls: int
    prompt_tokens: int
    completion_tokens: int
    events: list[dict[str, Any]] = field(default_factory=list)
    error_category: str | None = None
    final_text: str = ""


def run_editor_iteration(
    inputs: ResolvedInputs,
    predictions: Mapping[str, str],
    profile: ProviderProfile,
    executor: Callable[[ToolCall], dict[str, Any]],
    *,
    client: BailianToolClient | None = None,
    max_provider_calls: int = 4,
    max_tool_calls: int = 8,
    max_output_tokens_per_call: int = 512,
    max_elapsed_seconds: float = 120,
    parent_sha256: str | None = None,
) -> IterationResult:
    """Never include final questions in editor requests or tool results."""
    if not 1 <= max_provider_calls <= 4 or not 1 <= max_tool_calls <= 8:
        raise ValueError("editor call budget exceeds limit")
    if not 1 <= max_output_tokens_per_call <= 1024 or not 0 < max_elapsed_seconds <= 120:
        raise ValueError("editor token or time budget exceeds limit")
    practice = build_practice_feedback(inputs, predictions)
    holdout = [EvalRecord.model_validate(row) for row in read_jsonl(inputs.final_path)]
    index_path = getattr(inputs, "model_path", None)
    index_file = Path(index_path) / "model.safetensors.index.json" if index_path else None
    index = json.loads(index_file.read_text(encoding="utf-8")) if index_file and index_file.is_file() else {}
    available = sorted(name for name in index.get("weight_map", {})
                       if name.startswith("model.language_model.layers."))[:16]
    context = {"practice_feedback": practice, "parent_checkpoint_sha256": parent_sha256,
               "example_allowed_tensor_names": available,
               "edit_limits": {"max_elements": 32, "max_absolute_delta": 0.1}}
    messages = [
        ToolMessage(role="system", content=(
            "You are proposing one small weight edit for Qwen3.5-0.8B on I Ching practice questions. "
            "Use inspect_tensor to read actual values before propose_edit. Propose only values within 0.1 "
            "of those observed. Keep the same tensor name and parent SHA-256. Use only declared tools. "
            "The final 81-question holdout is unavailable during editing.")),
        ToolMessage(role="user", content=json.dumps(context, ensure_ascii=False, separators=(",", ":"))),
    ]
    client = client or BailianToolClient()
    start = time.monotonic()
    calls = tools = prompt_tokens = completion_tokens = 0
    events: list[dict[str, Any]] = []
    try:
        for _ in range(max_provider_calls):
            remaining = max_elapsed_seconds - (time.monotonic() - start)
            if remaining <= 0:
                raise ProviderError("timeout", "editor iteration time budget exhausted",
                                    retryable=True, uncertain=False)
            response = client.complete(profile, messages, holdout,
                                       max_output_tokens=max_output_tokens_per_call,
                                       timeout_seconds=min(remaining, 45))
            calls += 1
            prompt_tokens += response.prompt_tokens or 0
            completion_tokens += response.completion_tokens or 0
            reject_holdout_content(response.content, holdout)
            events.append({"kind": "provider", "finish_reason": response.finish_reason,
                           "request_id": response.provider_request_id,
                           "prompt_tokens": response.prompt_tokens,
                           "completion_tokens": response.completion_tokens,
                           "tool_count": len(response.calls)})
            if not response.calls:
                return IterationResult("completed", calls, tools, prompt_tokens,
                                       completion_tokens, events, final_text=response.content)
            wire_calls = [{"id": item.id, "type": "function",
                           "function": {"name": item.name,
                                        "arguments": json.dumps(item.arguments, ensure_ascii=False)}}
                          for item in response.calls]
            messages.append(ToolMessage(role="assistant", content=response.content, tool_calls=wire_calls))
            for call in response.calls:
                if tools >= max_tool_calls:
                    raise ProviderError("budget", "editor tool-call budget exhausted",
                                        retryable=False, uncertain=False)
                tools += 1
                try:
                    result = executor(call)
                    if not isinstance(result, dict):
                        raise TypeError("tool result must be an object")
                    reject_holdout_content(result, holdout)
                    events.append({"kind": "tool", "name": call.name, "status": "accepted"})
                except (ValueError, TypeError):
                    result = {"status": "rejected", "reason": "invalid_tool_request"}
                    events.append({"kind": "tool", "name": call.name, "status": "rejected"})
                messages.append(ToolMessage(role="tool", content=json.dumps(result, ensure_ascii=False),
                                            tool_call_id=call.id))
        raise ProviderError("budget", "editor provider-call budget exhausted",
                            retryable=False, uncertain=False)
    except MissingCredentialError:
        return IterationResult("failed", calls, tools, prompt_tokens, completion_tokens,
                               events, error_category="missing_credential")
    except ProviderError as exc:
        return IterationResult("failed", calls, tools, prompt_tokens, completion_tokens,
                               events, error_category=exc.category)


def run_pinned_iteration(inputs: ResolvedInputs, predictions: Mapping[str, str],
                         profile: ProviderProfile, *, client: BailianToolClient | None = None,
                         parent_sha256: str | None = None,
                         parent_path: str | Path | None = None) -> IterationResult:
    """Use only the verified registered checkpoint and fixed server-side tool dispatcher."""
    expected = parent_sha256 or inputs.manifest.target.checkpoint_sha256
    selected_path = Path(parent_path).resolve(strict=True) if parent_path is not None else inputs.model_path
    if sha256_directory(selected_path) != expected:
        raise ValueError("parent checkpoint digest is stale")
    return run_editor_iteration(
        inputs, predictions, profile,
        lambda call: dispatch_weight_tool(call, selected_path, expected), client=client,
        parent_sha256=expected,
    )
