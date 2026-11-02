"""Atomic, workspace-scoped admission lease for local GPU work."""

from __future__ import annotations

import json
import os
import shutil
import socket
import threading
import time
import uuid
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_INITIALIZING_GRACE_SECONDS = 15


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) maps to a console signal on Windows and can interrupt
        # the current process; query process state through the native API instead.
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # access denied means it exists
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class GpuLeaseBusy(RuntimeError):
    def __init__(self, owner: dict[str, Any]):
        self.owner = owner
        super().__init__("本地 GPU 正被其他任务占用" if not owner.get("stale") else "发现未恢复的陈旧 GPU 租约")


class GpuLeaseRecoveryRequired(RuntimeError):
    pass


@dataclass
class GpuLeaseHandle:
    manager: "WorkspaceGpuLease"
    token: str
    task_id: str

    def heartbeat(self) -> bool:
        return self.manager.heartbeat(self.token)

    def set_owner_pid(self, pid: int) -> bool:
        return self.manager.set_owner_pid(self.token, pid)

    def release(self) -> bool:
        return self.manager.release(self.token)

    def __enter__(self) -> "GpuLeaseHandle":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


class WorkspaceGpuLease:
    """One atomic lease per workspace; stale owners require explicit recovery."""

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve(strict=True)
        state_root = self.workspace / ".onegpu-posttrain"
        if state_root.is_symlink():
            raise ValueError("GPU 租约状态目录不能包含符号链接或越出工作区")
        self.lock_dir = state_root / "gpu-lease"
        self._local_lock = threading.RLock()

    def inspect(self) -> dict[str, Any]:
        path = self.lock_dir
        if not path.exists():
            return {"status": "available", "owner": None}
        if path.is_symlink() or not path.is_dir() or not path.resolve().is_relative_to(self.workspace):
            return {"status": "invalid", "owner": None, "reason": "GPU 租约记录不安全，需人工检查"}
        owner_path = path / "owner.json"
        if owner_path.is_symlink() or not owner_path.is_file():
            age = max(0.0, time.time() - path.stat().st_mtime)
            return {"status": "stale" if age >= _INITIALIZING_GRACE_SECONDS else "initializing",
                    "owner": None, "stale": age >= _INITIALIZING_GRACE_SECONDS,
                    "reason": "租约初始化中断；需显式恢复"}
        try:
            if owner_path.stat().st_size > 16 * 1024:
                raise ValueError("oversized")
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            if not isinstance(owner, dict) or owner.get("schema_version") != "1.0":
                raise ValueError("invalid")
        except (OSError, UnicodeError, ValueError):
            return {"status": "invalid", "owner": None, "reason": "租约记录无效，需人工检查"}
        pid = owner.get("owner_pid")
        host = owner.get("owner_host")
        stale = host == socket.gethostname() and isinstance(pid, int) and not _pid_alive(pid)
        return {"status": "stale" if stale else "busy", "stale": stale,
                "owner": {key: owner.get(key) for key in (
                    "task_type", "task_id", "run_id", "owner_pid", "owner_host", "acquired_at", "heartbeat_at",
                )} | {"recovery_token": owner.get("token")}}

    def acquire(self, *, task_type: str, task_id: str, run_id: str | None = None,
                owner_pid: int | None = None) -> GpuLeaseHandle:
        for name, value, limit in (("task_type", task_type, 40), ("task_id", task_id, 160)):
            if not isinstance(value, str) or not value or len(value) > limit or any(ord(c) < 32 for c in value):
                raise ValueError(f"无效 GPU 租约 {name}")
        if run_id is not None and (not isinstance(run_id, str) or len(run_id) > 160):
            raise ValueError("无效 GPU 租约 run_id")
        token = uuid.uuid4().hex
        with self._local_lock:
            state_root = self.lock_dir.parent
            if state_root.is_symlink():
                raise ValueError("GPU 租约状态目录不能包含符号链接")
            state_root.mkdir(parents=True, exist_ok=True)
            if not state_root.resolve().is_relative_to(self.workspace):
                raise ValueError("GPU 租约状态目录越出工作区")
            try:
                self.lock_dir.mkdir()
            except FileExistsError as exc:
                state = self.inspect()
                owner = state.get("owner") or {"status": state["status"], "reason": state.get("reason")}
                owner["stale"] = state.get("stale", False)
                raise GpuLeaseBusy(owner) from exc
            record = {"schema_version": "1.0", "token": token, "task_type": task_type,
                      "task_id": task_id, "run_id": run_id, "owner_pid": owner_pid or os.getpid(),
                      "owner_host": socket.gethostname(), "acquired_at": _utc_now(), "heartbeat_at": _utc_now()}
            try:
                self._write_owner(record)
            except BaseException:
                # A failed initializer leaves a deliberately detectable stale directory.
                raise
        return GpuLeaseHandle(self, token, task_id)

    def heartbeat(self, token: str) -> bool:
        with self._local_lock:
            owner = self._read_owner()
            if not owner or owner.get("token") != token:
                return False
            owner["heartbeat_at"] = _utc_now()
            self._write_owner(owner)
            return True

    def set_owner_pid(self, token: str, pid: int) -> bool:
        if not isinstance(pid, int) or pid <= 0:
            raise ValueError("owner_pid must be a positive integer")
        with self._local_lock:
            owner = self._read_owner()
            if not owner or owner.get("token") != token:
                return False
            owner["owner_pid"] = pid
            owner["heartbeat_at"] = _utc_now()
            self._write_owner(owner)
            return True

    def release(self, token: str) -> bool:
        with self._local_lock:
            owner = self._read_owner()
            if not owner or owner.get("token") != token:
                return False
            retired = self.lock_dir.with_name(f"gpu-lease-released-{uuid.uuid4().hex}")
            try:
                os.rename(self.lock_dir, retired)
            except OSError:
                return False
            shutil.rmtree(retired, ignore_errors=True)
            return True

    def recover_stale(self, recovery_token: str, *, confirm: bool = False) -> bool:
        if confirm is not True:
            raise GpuLeaseRecoveryRequired("必须显式确认后才能恢复陈旧 GPU 租约")
        state = self.inspect()
        owner = state.get("owner")
        if state["status"] != "stale":
            raise GpuLeaseRecoveryRequired("当前租约不是可恢复的陈旧租约")
        expected_token = owner.get("recovery_token") if isinstance(owner, dict) else "orphaned-initializer"
        if not recovery_token or expected_token != recovery_token:
            raise GpuLeaseRecoveryRequired("陈旧租约恢复凭据不匹配")
        retired = self.lock_dir.with_name(f"gpu-lease-recovered-{uuid.uuid4().hex}")
        try:
            os.rename(self.lock_dir, retired)
        except OSError as exc:
            raise GpuLeaseRecoveryRequired("租约状态已变化，请重新检查") from exc
        shutil.rmtree(retired, ignore_errors=True)
        return True

    def _read_owner(self) -> dict[str, Any] | None:
        owner_path = self.lock_dir / "owner.json"
        try:
            if self.lock_dir.is_symlink() or owner_path.is_symlink() or not owner_path.is_file():
                return None
            value = json.loads(owner_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, UnicodeError, ValueError):
            return None

    def _write_owner(self, value: dict[str, Any]) -> None:
        temp = self.lock_dir / f"owner.{uuid.uuid4().hex}.tmp"
        target = self.lock_dir / "owner.json"
        payload = (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        try:
            with temp.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, target)
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
