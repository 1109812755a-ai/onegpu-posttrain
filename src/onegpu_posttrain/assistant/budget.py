"""Conservative, dependency-free input budget for the MVP assistant."""

from __future__ import annotations

from dataclasses import dataclass
import json

from .contracts import ModelRequest

MAX_CONTEXT_ESTIMATE = 16_000
RESERVED_OUTPUT_TOKENS = 1_200
MAX_INPUT_ESTIMATE = MAX_CONTEXT_ESTIMATE - RESERVED_OUTPUT_TOKENS
MAX_RECENT_TURNS = 8
MAX_EVIDENCE_ITEMS = 24


@dataclass(frozen=True)
class ContextBudget:
    estimated_input_tokens: int
    output_reserve_tokens: int = RESERVED_OUTPUT_TOKENS
    context_limit_tokens: int = MAX_CONTEXT_ESTIMATE


class ContextBudgetExceeded(ValueError):
    def __init__(self, estimate: int, limit: int):
        self.estimate = estimate
        self.limit = limit
        super().__init__(
            "本轮问题、必要证据和近期上下文超过助手安全预算；请缩小查询范围，或开始新会话。"
        )


def conservative_token_estimate(text: str) -> int:
    """Count CJK code points as two tokens and other code points as one.

    This intentionally overestimates ordinary text and avoids a tokenizer
    dependency. It is a guardrail rather than a promise about provider billing.
    """
    total = 0
    for char in text:
        code = ord(char)
        if (
            0x2E80 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
            or 0x20000 <= code <= 0x3134F
        ):
            total += 2
        else:
            total += 1
    return total


def estimate_request(request: ModelRequest) -> ContextBudget:
    encoded = json.dumps(request.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
    return ContextBudget(estimated_input_tokens=conservative_token_estimate(encoded))


def enforce_request_budget(request: ModelRequest) -> ContextBudget:
    budget = estimate_request(request)
    if budget.estimated_input_tokens > MAX_INPUT_ESTIMATE:
        raise ContextBudgetExceeded(budget.estimated_input_tokens, MAX_INPUT_ESTIMATE)
    return budget
