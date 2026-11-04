"""Versioned local persistence for assistant sessions, tasks and profiles."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from .contracts import (
    AgentTaskRecord, AssistantProfile, ContextSummary, ConversationSession,
    SessionMessage, WorkspaceMemory,
)

MAX_SESSION_BYTES = 5 * 1024 * 1024
MAX_TASK_BYTES = 512 * 1024
MAX_PROFILE_BYTES = 64 * 1024
MAX_MEMORY_BYTES = 64 * 1024
MAX_SUMMARY_BYTES = 256 * 1024
MAX_EVALUATION_BYTES = 2 * 1024 * 1024
MAX_STORED_SESSIONS = 1000
MAX_STORED_TASKS = 5000
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class WorkspaceAssistantStore:
    """Store assistant data below `.onegpu-posttrain/assistant`, outside artifacts."""

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve(strict=True)
        self.root = self.workspace / ".onegpu-posttrain" / "assistant"
        self.sessions = self.root / "sessions"
        self.tasks = self.root / "tasks"
        self.profiles = self.root / "profiles"
        self.memories = self.root / "memories"
        self.summaries = self.root / "summaries"
        self.evaluations = self.root / "evaluations"
        self._ensure_layout()

    def _ensure_layout(self) -> None:
        for path in (self.workspace / ".onegpu-posttrain", self.root, self.sessions, self.tasks,
                     self.profiles, self.evaluations, self.memories, self.summaries):
            if path.is_symlink():
                raise ValueError("助手存储目录不能是符号链接")
            path.mkdir(parents=True, exist_ok=True)
            if not path.resolve(strict=True).is_relative_to(self.workspace):
                raise ValueError("助手存储目录越出当前工作区")

    @staticmethod
    def _path(directory: Path, entity_id: str) -> Path:
        if not _SAFE_ID.fullmatch(entity_id):
            raise ValueError("无效助手记录 ID")
        path = directory / f"{entity_id}.json"
        if path.is_symlink():
            raise ValueError("助手记录不能是符号链接")
        return path

    @staticmethod
    def _read(path: Path, model, max_bytes: int):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
            raise ValueError("助手记录缺失或超过安全读取上限")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return model.model_validate(raw)
        except (OSError, UnicodeError, ValueError, ValidationError, TypeError):
            raise ValueError("助手记录格式无效或版本不受支持") from None

    @staticmethod
    def _write(path: Path, value, max_bytes: int) -> None:
        encoded = value.model_dump_json(indent=2).encode("utf-8")
        if len(encoded) > max_bytes:
            raise ValueError("助手记录超过安全写入上限")
        temporary = path.with_name(f".{path.stem}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            if path.is_symlink():
                raise ValueError("助手记录不能是符号链接")
            os.replace(temporary, path)
        finally:
            if temporary.exists() and not temporary.is_symlink():
                temporary.unlink()

    def create(self, task: AgentTaskRecord) -> None:
        path = self._path(self.tasks, task.task_id)
        if path.exists():
            raise ValueError("任务记录已存在")
        if len(list(self.tasks.glob("*.json"))) >= MAX_STORED_TASKS:
            raise ValueError("任务记录数量达到上限")
        self._write(path, task, MAX_TASK_BYTES)

    def update(self, task: AgentTaskRecord) -> None:
        path = self._path(self.tasks, task.task_id)
        if not path.is_file():
            raise KeyError(task.task_id)
        self._read(path, AgentTaskRecord, MAX_TASK_BYTES)
        self._write(path, task, MAX_TASK_BYTES)

    def get(self, task_id: str) -> AgentTaskRecord | None:
        path = self._path(self.tasks, task_id)
        return self._read(path, AgentTaskRecord, MAX_TASK_BYTES) if path.exists() else None

    def list_tasks(self, *, session_id: str | None = None, limit: int = 100) -> list[AgentTaskRecord]:
        limit = max(1, min(int(limit), 200))
        paths = [path for path in self.tasks.glob("*.json")
                 if not path.is_symlink() and _SAFE_ID.fullmatch(path.stem)]
        paths.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        result: list[AgentTaskRecord] = []
        for path in paths:
            try:
                record = self._read(path, AgentTaskRecord, MAX_TASK_BYTES)
            except ValueError:
                continue
            if session_id is None or record.session_id == session_id:
                result.append(record)
                if len(result) >= limit:
                    break
        return result

    def save_evaluation_report(self, report) -> None:
        evaluation_id = getattr(report, "evaluation_id", None)
        path = self._path(self.evaluations, evaluation_id)
        if path.exists():
            raise ValueError("助手评估报告 ID 已存在")
        if len(list(self.evaluations.glob("*.json"))) >= 500:
            raise ValueError("助手评估报告数量达到上限")
        self._write(path, report, MAX_EVALUATION_BYTES)

    def list_evaluation_reports(self, limit: int = 20) -> list[dict]:
        limit = max(1, min(int(limit), 100))
        paths = [path for path in self.evaluations.glob("*.json")
                 if not path.is_symlink() and _SAFE_ID.fullmatch(path.stem)]
        paths.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        reports = []
        for path in paths[:limit]:
            try:
                report = json.loads(self._read_raw(path, MAX_EVALUATION_BYTES))
                if report.get("evaluation_id") == path.stem:
                    reports.append(report)
            except (ValueError, TypeError, AttributeError):
                continue
        return reports

    @staticmethod
    def _read_raw(path: Path, max_bytes: int) -> str:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
            raise ValueError("评估报告缺失或超过安全读取上限")
        return path.read_text(encoding="utf-8")

    def create_session(self, title: str, assistant_profile_id: str) -> ConversationSession:
        if len(list(self.sessions.glob("*.json"))) >= MAX_STORED_SESSIONS:
            raise ValueError("会话数量达到上限")
        now = _now()
        session = ConversationSession(
            session_id=f"session-{uuid4().hex[:24]}",
            title=title.strip()[:160] or "新对话",
            assistant_profile_id=assistant_profile_id,
            created_at=now,
            updated_at=now,
        )
        self._write(self._path(self.sessions, session.session_id), session, MAX_SESSION_BYTES)
        return session

    def get_session(self, session_id: str) -> ConversationSession | None:
        path = self._path(self.sessions, session_id)
        return self._read(path, ConversationSession, MAX_SESSION_BYTES) if path.exists() else None

    def list_sessions(self, limit: int = 100) -> list[ConversationSession]:
        limit = max(1, min(int(limit), 200))
        paths = [path for path in self.sessions.glob("*.json")
                 if not path.is_symlink() and _SAFE_ID.fullmatch(path.stem)]
        paths.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        return [self._read(path, ConversationSession, MAX_SESSION_BYTES) for path in paths[:limit]]

    def save_session(self, session: ConversationSession) -> ConversationSession:
        path = self._path(self.sessions, session.session_id)
        if not path.is_file():
            raise KeyError(session.session_id)
        self._read(path, ConversationSession, MAX_SESSION_BYTES)
        updated = session.model_copy(update={"updated_at": _now(), "revision": session.revision + 1})
        self._write(path, updated, MAX_SESSION_BYTES)
        return updated

    def append_message(self, session_id: str, message: SessionMessage) -> ConversationSession:
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        if len(session.messages) >= 200:
            raise ValueError("单个会话消息数达到上限")
        return self.save_session(session.model_copy(update={"messages": [*session.messages, message]}))

    def grant_data_egress_consent(self, session_id: str, *, version: str) -> ConversationSession:
        """Record explicit user confirmation for sending minimal evidence to Bailian."""
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,40}", version):
            raise ValueError("无效的数据外发说明版本")
        return self.save_session(session.model_copy(update={
            "data_egress_consent_at": _now(),
            "data_egress_consent_version": version,
        }))

    def delete_session(self, session_id: str) -> bool:
        path = self._path(self.sessions, session_id)
        if not path.exists():
            return False
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(self.root.resolve()):
            raise ValueError("会话记录路径无效")
        path.unlink()
        return True

    def save_profile(self, profile: AssistantProfile) -> None:
        if profile.built_in:
            raise ValueError("内置助手配置由程序管理")
        path = self._path(self.profiles, profile.profile_id)
        if path.exists():
            current = self._read(path, AssistantProfile, MAX_PROFILE_BYTES)
            profile = profile.model_copy(update={"created_at": current.created_at})
        self._write(path, profile, MAX_PROFILE_BYTES)

    def get_profile(self, profile_id: str) -> AssistantProfile | None:
        path = self._path(self.profiles, profile_id)
        return self._read(path, AssistantProfile, MAX_PROFILE_BYTES) if path.exists() else None

    def list_profiles(self) -> list[AssistantProfile]:
        paths = [path for path in self.profiles.glob("*.json")
                 if not path.is_symlink() and _SAFE_ID.fullmatch(path.stem)]
        return [self._read(path, AssistantProfile, MAX_PROFILE_BYTES) for path in paths]

    def delete_profile(self, profile_id: str) -> bool:
        path = self._path(self.profiles, profile_id)
        if not path.exists():
            return False
        profile = self._read(path, AssistantProfile, MAX_PROFILE_BYTES)
        if profile.built_in:
            raise ValueError("不能删除内置助手")
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(self.root.resolve()):
            raise ValueError("助手配置路径无效")
        path.unlink()
        return True

    def save_memory(self, memory: WorkspaceMemory) -> WorkspaceMemory:
        path = self._path(self.memories, memory.memory_id)
        if path.exists():
            current = self._read(path, WorkspaceMemory, MAX_MEMORY_BYTES)
            memory = memory.model_copy(update={"created_at": current.created_at})
        elif len(list(self.memories.glob("*.json"))) >= 500:
            raise ValueError("工作区记忆数量达到上限")
        self._write(path, memory, MAX_MEMORY_BYTES)
        return memory

    def list_memories(self, *, include_disabled: bool = True) -> list[WorkspaceMemory]:
        paths = [path for path in self.memories.glob("*.json")
                 if not path.is_symlink() and _SAFE_ID.fullmatch(path.stem)]
        paths.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        result = []
        for path in paths:
            try:
                memory = self._read(path, WorkspaceMemory, MAX_MEMORY_BYTES)
            except ValueError:
                continue
            if include_disabled or memory.status == "active":
                result.append(memory)
        return result

    def delete_memory(self, memory_id: str) -> bool:
        path = self._path(self.memories, memory_id)
        if not path.exists():
            return False
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(self.root.resolve()):
            raise ValueError("记忆记录路径无效")
        path.unlink()
        return True

    def save_summary(self, summary: ContextSummary) -> ContextSummary:
        path = self._path(self.summaries, summary.session_id)
        if path.exists():
            current = self._read(path, ContextSummary, MAX_SUMMARY_BYTES)
            summary = summary.model_copy(update={"revision": current.revision + 1})
        self._write(path, summary, MAX_SUMMARY_BYTES)
        return summary

    def get_summary(self, session_id: str) -> ContextSummary | None:
        path = self._path(self.summaries, session_id)
        return self._read(path, ContextSummary, MAX_SUMMARY_BYTES) if path.exists() else None
