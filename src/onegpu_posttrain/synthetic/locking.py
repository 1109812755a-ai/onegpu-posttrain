"""Exclusive per-plan execution lock; stale locks require operator review."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class PlanExecutionLock:
    def __init__(self, plan_dir: Path):
        self.path = plan_dir / ".execution.lock"
        self.run_id = f"synthetic-run-{uuid4().hex}"
        self._stream = None

    def __enter__(self) -> Any:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._stream = self.path.open("x", encoding="utf-8")
        except FileExistsError:
            raise RuntimeError(
                "another execution owns this plan lock; inspect the lock owner before manual stale-lock recovery"
            ) from None
        payload = {
            "run_id": self.run_id,
            "pid": os.getpid(),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._stream.write(json.dumps(payload, sort_keys=True) + "\n")
            self._stream.flush()
            os.fsync(self._stream.fileno())
        except BaseException:
            self._stream.close()
            self.path.unlink(missing_ok=True)
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._stream is not None:
            self._stream.close()
        self.path.unlink(missing_ok=True)
