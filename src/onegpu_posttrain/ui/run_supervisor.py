"""Reusable local single-GPU run lifecycle, bounded log, and progress protocol."""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RUN_STATUSES = ("queued", "starting", "running", "completed", "partial", "failed", "oom", "interrupted")
ACTIVE_STATUSES = {"queued", "starting", "running"}
TERMINAL_STATUSES = set(RUN_STATUSES) - ACTIVE_STATUSES
MAX_RUN_LOG_BYTES = 1024 * 1024
MAX_RUN_LOG_TAIL_BYTES = 64 * 1024
_RUN_ID = re.compile(r"^run-[a-z0-9][a-z0-9-]{0,119}$")
_PROGRESS = re.compile(
    r"(?:global_step|step)\s*[=:]\s*['\"]?(\d+)(?:\s*/\s*(\d+))?['\"]?"
    r"|(?<![\w.])(\d+)\s*/\s*(\d+)(?![\w.])",
    re.IGNORECASE,
)
_TRANSITIONS = {
    "queued": {"starting", "failed", "interrupted"},
    "starting": {"running", "failed", "oom", "interrupted"},
    "running": {"completed", "partial", "failed", "oom", "interrupted"},
    "completed": set(), "partial": set(), "failed": set(), "oom": set(), "interrupted": set(),
}


class RunLifecycleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(pattern=r"^run-[a-z0-9][a-z0-9-]{0,119}$")
    stage: str = Field(min_length=1, max_length=40)
    profile_id: str = Field(min_length=1, max_length=100)
    status: Literal["queued", "starting", "running", "completed", "partial", "failed", "oom", "interrupted"]
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    output_location: str
    progress: dict[str, int | None] | None = None
    progress_status: Literal["unknown", "measured"] = "unknown"
    log_bytes: int = Field(default=0, ge=0, le=MAX_RUN_LOG_BYTES)
    exit_code: int | None = None
    detail: str | None = None


class LocalRunSupervisor:
    """Persist lifecycle records and bound logs; it does not itself launch processes."""

    def __init__(self, workspace: str | Path, *, max_log_bytes: int = MAX_RUN_LOG_BYTES):
        if max_log_bytes < 128:
            raise ValueError("max_log_bytes must be at least 128")
        self.workspace = Path(workspace).resolve(strict=True)
        self.root = self.workspace / ".onegpu-posttrain" / "supervisor"
        self.records = self.root / "runs"
        self.logs = self.root / "logs"
        self.max_log_bytes = min(max_log_bytes, MAX_RUN_LOG_BYTES)
        self._lock = threading.RLock()
        self._active_run_id: str | None = None
        self._prepare_roots()
        self._recover_stale()

    def create(self, *, run_id: str, stage: str, profile_id: str, output_location: str) -> dict[str, Any]:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("无效 run_id")
        output = Path(output_location)
        if output.is_absolute() or ".." in output.parts or not output.parts:
            raise ValueError("output_location 必须是工作区内相对路径")
        target = (self.workspace / output).resolve(strict=False)
        if not target.is_relative_to(self.workspace):
            raise ValueError("output_location 越出工作区")
        with self._lock:
            active = self.active_run()
            if active:
                raise RuntimeError(f"已有本地 GPU Run 活动：{active['run_id']} ({active['status']})")
            path = self._record_path(run_id)
            if path.exists():
                raise FileExistsError("run_id 已存在；历史 Run 不覆盖")
            record = RunLifecycleRecord(
                run_id=run_id, stage=stage, profile_id=profile_id, status="queued",
                created_at=_utc_now(), output_location=output.as_posix(),
            )
            self._write_record(record)
            self._active_run_id = run_id
            (self.logs / f"{run_id}.log").write_bytes(b"")
            return record.model_dump(mode="json")

    def transition(self, run_id: str, status: str, *, detail: str | None = None,
                   exit_code: int | None = None) -> dict[str, Any]:
        if status not in RUN_STATUSES:
            raise ValueError(f"unsupported Run status: {status}")
        with self._lock:
            record = self._read_record(run_id)
            if status not in _TRANSITIONS[record.status]:
                raise ValueError(f"非法 Run 状态转换：{record.status} -> {status}")
            now = _utc_now()
            if status == "starting":
                record.started_at = record.started_at or now
            if status in TERMINAL_STATUSES:
                record.completed_at = now
                record.exit_code = exit_code
                self._active_run_id = None
            if detail is not None:
                record.detail = detail[:1000]
            record.status = status  # validated transition above
            self._write_record(record)
            return record.model_dump(mode="json")

    def append_log(self, run_id: str, text: str) -> dict[str, Any]:
        if not isinstance(text, str):
            raise TypeError("log chunk must be text")
        encoded = text.encode("utf-8", errors="replace")
        with self._lock:
            record = self._read_record(run_id)
            path = self._log_path(run_id)
            existing = path.read_bytes() if path.is_file() else b""
            combined = existing + encoded
            if len(combined) > self.max_log_bytes:
                tail_size = min(MAX_RUN_LOG_TAIL_BYTES, self.max_log_bytes - 64)
                combined = "[日志已达到上限，保留尾部内容]\n".encode("utf-8") + combined[-tail_size:]
            self._atomic_write(path, combined)
            record.log_bytes = len(combined)
            self._write_record(record)
            return {"run_id": run_id, "log_bytes": record.log_bytes}

    def update_progress(self, run_id: str, text: str) -> dict[str, Any]:
        parsed = parse_training_progress(text)
        with self._lock:
            record = self._read_record(run_id)
            record.progress = parsed
            record.progress_status = "measured" if parsed is not None else "unknown"
            self._write_record(record)
            return record.model_dump(mode="json")

    def read_log(self, run_id: str, *, tail_bytes: int = MAX_RUN_LOG_TAIL_BYTES) -> str:
        if not 1 <= tail_bytes <= MAX_RUN_LOG_TAIL_BYTES:
            raise ValueError("tail_bytes exceeds the safe bound")
        data = self._log_path(run_id).read_bytes()
        return data[-tail_bytes:].decode("utf-8", errors="replace")

    def get(self, run_id: str) -> dict[str, Any] | None:
        try:
            return self._read_record(run_id).model_dump(mode="json")
        except (OSError, ValueError, TypeError):
            return None

    def active_run(self) -> dict[str, Any] | None:
        if self._active_run_id is None:
            return None
        record = self.get(self._active_run_id)
        if record is None or record["status"] not in ACTIVE_STATUSES:
            self._active_run_id = None
            return None
        return record

    def interrupt_active(self, reason: str = "UI 服务关闭时中断活动 Run") -> dict[str, Any] | None:
        active = self.active_run()
        return self.transition(active["run_id"], "interrupted", detail=reason) if active else None

    def _recover_stale(self) -> None:
        for path in self.records.glob("*.json"):
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 128 * 1024:
                continue
            try:
                record = RunLifecycleRecord.model_validate_json(path.read_text(encoding="utf-8"))
                if record.status in ACTIVE_STATUSES:
                    record.status = "interrupted"
                    record.completed_at = _utc_now()
                    record.detail = "supervisor 重启后无法确认旧进程状态；不会自动恢复训练"
                    self._write_record(record)
            except (OSError, ValueError, TypeError):
                continue

    def _prepare_roots(self) -> None:
        parent = self.workspace / ".onegpu-posttrain"
        if parent.is_symlink() or self.root.is_symlink():
            raise ValueError("supervisor 状态目录不能经过符号链接")
        self.records.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        for path in (self.root, self.records, self.logs):
            if path.is_symlink() or not path.resolve().is_relative_to(self.workspace):
                raise ValueError("supervisor 状态目录越出工作区或包含符号链接")

    def _record_path(self, run_id: str) -> Path:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("无效 run_id")
        return self.records / f"{run_id}.json"

    def _log_path(self, run_id: str) -> Path:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("无效 run_id")
        path = self.logs / f"{run_id}.log"
        if path.is_symlink():
            raise ValueError("Run 日志不能是符号链接")
        return path

    def _read_record(self, run_id: str) -> RunLifecycleRecord:
        path = self._record_path(run_id)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 128 * 1024:
            raise ValueError("Run lifecycle record 不存在或无效")
        record = RunLifecycleRecord.model_validate_json(path.read_text(encoding="utf-8"))
        if record.run_id != run_id:
            raise ValueError("Run lifecycle identity mismatch")
        return record

    def _write_record(self, record: RunLifecycleRecord) -> None:
        self._atomic_write(self._record_path(record.run_id),
                           (record.model_dump_json(indent=2) + "\n").encode("utf-8"))

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        temp = path.with_name(f"{path.name}.{secrets.token_hex(6)}.tmp")
        try:
            with temp.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def parse_training_progress(text: str) -> dict[str, int | None] | None:
    """Return measured progress only for explicit current/total step evidence."""
    if not isinstance(text, str):
        return None
    matches = list(_PROGRESS.finditer(text[-16_384:]))
    if not matches:
        return None
    match = matches[-1]
    step_group = match.group(1) or match.group(3)
    total_group = match.group(2) or match.group(4)
    step, total = int(step_group), int(total_group) if total_group is not None else None
    if step < 0 or (total is not None and (total <= 0 or step > total)):
        return None
    return {"step": step, "total": total}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
