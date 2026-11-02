"""Append-only, private-free request accounting for targeted generation."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RequestLedger:
    """Persist attempt starts before provider calls so crashes still consume budget."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._latest_by_attempt: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        try:
            for line_no, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                attempt_id = row.get("attempt_id")
                if (not isinstance(attempt_id, str) or not attempt_id
                        or row.get("state") not in {"pending", "completed", "failed", "uncertain", "reviewed_unknown"}):
                    raise ValueError
                self._latest_by_attempt[attempt_id] = row
        except (OSError, json.JSONDecodeError, AttributeError, ValueError, TypeError):
            raise ValueError("request ledger is malformed; refusing to reset or reuse its budget") from None

    @property
    def attempt_count(self) -> int:
        return len(self._latest_by_attempt)

    def remaining_requests(self, max_requests: int) -> int:
        return max(0, max_requests - self.attempt_count)

    def completed_cache(self, request_sha256: str) -> dict[str, Any] | None:
        matches = [
            row for row in self._latest_by_attempt.values()
            if row.get("request_sha256") == request_sha256 and row.get("state") == "completed"
        ]
        return matches[-1] if matches else None

    def unresolved_attempt(self, request_sha256: str) -> dict[str, Any] | None:
        matches = [
            row for row in self._latest_by_attempt.values()
            if row.get("request_sha256") == request_sha256 and row.get("state") in {"pending", "uncertain"}
        ]
        return matches[-1] if matches else None

    def append(self, event: dict[str, Any]) -> None:
        attempt_id = event.get("attempt_id")
        state = event.get("state")
        if not isinstance(attempt_id, str) or state not in {"pending", "completed", "failed", "uncertain", "reviewed_unknown"}:
            raise ValueError("ledger event requires an attempt_id and valid state")
        if state == "pending" and attempt_id in self._latest_by_attempt:
            raise ValueError("request attempt_id cannot be reused")
        row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event}
        encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._latest_by_attempt[attempt_id] = row

    def summary(self) -> dict[str, Any]:
        prompt_values: list[int] = []
        completion_values: list[int] = []
        unknown_usage = 0
        costs: list[float] = []
        unknown_cost = 0
        states: dict[str, int] = defaultdict(int)
        for row in self._latest_by_attempt.values():
            states[row["state"]] += 1
            prompt = row.get("prompt_tokens")
            completion = row.get("completion_tokens")
            if row["state"] == "completed" and isinstance(prompt, int) and isinstance(completion, int):
                prompt_values.append(prompt)
                completion_values.append(completion)
            else:
                unknown_usage += 1
            cost = row.get("estimated_cost_usd")
            if isinstance(cost, (int, float)):
                costs.append(float(cost))
            else:
                unknown_cost += 1
        return {
            "request_attempts": self.attempt_count,
            "request_states": dict(sorted(states.items())),
            "reported_prompt_tokens": sum(prompt_values) if prompt_values else None,
            "reported_completion_tokens": sum(completion_values) if completion_values else None,
            "usage_unknown_attempts": unknown_usage,
            "estimated_cost_usd": sum(costs) if costs and unknown_cost == 0 else None,
            "cost_status": "estimate" if costs and unknown_cost == 0 else "unknown",
            "cost_unknown_attempts": unknown_cost,
        }
