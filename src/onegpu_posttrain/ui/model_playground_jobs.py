"""In-memory job orchestration for private model-playground requests."""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from .model_playground_catalog import ModelPlaygroundCatalog
from .model_playground_session import MAX_PROMPT_CHARS, ModelPlaygroundSessionManager

MAX_IN_MEMORY_JOBS = 100


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelPlaygroundJobs:
    """Keep prompts and responses only in worker memory; never write chat data to disk."""

    def __init__(self, workspace, catalog: ModelPlaygroundCatalog, *, session_manager: Any | None = None):
        self.manager = session_manager or ModelPlaygroundSessionManager(catalog, workspace)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-playground")
        self._lock = threading.RLock()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._jobs: dict[str, dict[str, Any]] = {}
        self._comparison_jobs: set[str] = set()
        self._futures: dict[str, Future] = {}
        self._closed = False

    def start_session(self, model_id: str) -> dict[str, Any]:
        if not isinstance(model_id, str) or len(model_id) > 160:
            raise ValueError("model_id 无效")
        model = self.manager.catalog.resolve(model_id)
        if not model or not model.get("available"):
            raise ValueError("模型不存在或当前不可体验")
        session_id = uuid.uuid4().hex
        with self._lock:
            self._ensure_open()
            if self._has_active_comparison():
                raise RuntimeError("已有模型对照任务；请等待其完成后再开始会话")
            if any(row["status"] in {"loading", "ready", "generating"} for row in self._sessions.values()):
                raise RuntimeError("已有模型体验会话；请先结束当前会话")
            self._prune()
            state = {"session_id": session_id, "model_id": model_id, "status": "loading",
                     "created_at": _now(), "detail": None, "model": None}
            self._sessions[session_id] = state
            self._futures[f"session:{session_id}"] = self._executor.submit(self._load_session, session_id, model_id)
            return self._public_session(state)

    def _load_session(self, session_id: str, model_id: str) -> None:
        try:
            metadata = self.manager.open(model_id, session_id=session_id)
            with self._lock:
                state = self._sessions.get(session_id)
                if state and state["status"] != "closed":
                    state.update(status="ready", model=metadata)
                    return
            # The UI may have left while synchronous local weights were loading.
            self.manager.close(session_id)
        except Exception as exc:
            with self._lock:
                state = self._sessions.get(session_id)
                if state and state["status"] != "closed":
                    detail = str(exc)[:240] if isinstance(exc, (ValueError, RuntimeError)) else f"模型加载失败：{type(exc).__name__}"
                    state.update(status="failed", detail=detail)

    def session_status(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            state = self._sessions.get(session_id)
            return self._public_session(state) if state else None

    def submit_message(self, session_id: str, prompt: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"消息不能为空或超过 {MAX_PROMPT_CHARS} 个字符")
        effective_parameters = ModelPlaygroundSessionManager._validate_options(parameters or {})
        with self._lock:
            self._ensure_open()
            session = self._sessions.get(session_id)
            if not session or session["status"] != "ready":
                raise ValueError("模型会话尚未就绪或已结束")
            session["status"] = "generating"
            job_id = uuid.uuid4().hex
            job = {"job_id": job_id, "session_id": session_id, "status": "queued",
                   "created_at": _now(), "parameters": effective_parameters, "result": None,
                   "detail": None, "cancel_requested": False}
            self._jobs[job_id] = job
            self._prune()
            # Prompt is captured by the worker closure, not copied to job/status metadata.
            future = self._executor.submit(self._generate, job_id, session_id, prompt, effective_parameters)
            self._futures[job_id] = future
            return self._public_job(job)

    def _generate(self, job_id: str, session_id: str, prompt: str, parameters: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            if job["cancel_requested"]:
                job.update(status="cancelled", detail="生成已取消")
                session = self._sessions.get(session_id)
                if session:
                    session["status"] = "ready"
                return
            job["status"] = "running"
        try:
            result = self.manager.chat(session_id, prompt, parameters)
            with self._lock:
                job = self._jobs.get(job_id)
                if job:
                    job.update(status="completed", result=result, completed_at=_now())
        except Exception as exc:
            with self._lock:
                job = self._jobs.get(job_id)
                if job:
                    cancelled = job["cancel_requested"] or "取消" in str(exc)
                    job.update(status="cancelled" if cancelled else "failed",
                               detail="生成已取消" if cancelled else str(exc)[:240], completed_at=_now())
        finally:
            with self._lock:
                session = self._sessions.get(session_id)
                if session and session["status"] == "generating":
                    manager_session = self.manager.active_session
                    session["status"] = "ready" if manager_session and manager_session["session_id"] == session_id else "closed"

    def submit_comparison(self, model_ids: list[str], prompt: str,
                          parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"消息不能为空或超过 {MAX_PROMPT_CHARS} 个字符")
        effective = ModelPlaygroundSessionManager._validate_options(parameters or {})
        # Validate the selection before enqueueing and before any model is loaded.
        if not isinstance(model_ids, list) or not 2 <= len(model_ids) <= 3:
            raise ValueError("对照体验需要选择 2 到 3 个模型")
        if any(not isinstance(value, str) or not value or len(value) > 160 for value in model_ids):
            raise ValueError("model_ids 无效")
        if len(set(model_ids)) != len(model_ids):
            raise ValueError("对照模型不能重复")
        for model_id in model_ids:
            model = self.manager.catalog.resolve(model_id) if isinstance(model_id, str) else None
            if not model or not model.get("available"):
                raise ValueError("对照列表包含不存在或当前不可体验的模型")
        with self._lock:
            self._ensure_open()
            if self._has_active_comparison():
                raise RuntimeError("已有模型对照任务；请等待其完成后再提交")
            if any(row["status"] in {"loading", "ready", "generating"} for row in self._sessions.values()):
                raise RuntimeError("已有模型体验会话；请先结束当前会话")
            job_id = uuid.uuid4().hex
            job = {"job_id": job_id, "session_id": None, "status": "queued", "created_at": _now(),
                   "parameters": effective, "result": None, "detail": None, "cancel_requested": False}
            self._jobs[job_id] = job
            self._comparison_jobs.add(job_id)
            self._prune()
            # The prompt and selected ids are worker-only ephemeral inputs, not job metadata.
            self._futures[job_id] = self._executor.submit(self._compare, job_id, model_ids[:], prompt, effective)
            return self._public_job(job)

    def _compare(self, job_id: str, model_ids: list[str], prompt: str, parameters: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job["status"] = "running"
        try:
            result = self.manager.compare(model_ids, prompt, parameters)
            with self._lock:
                job = self._jobs.get(job_id)
                if job:
                    job.update(status="completed" if result["status"] == "completed" else result["status"],
                               result=result, completed_at=_now())
        except Exception as exc:
            with self._lock:
                job = self._jobs.get(job_id)
                if job:
                    job.update(status="failed", detail=str(exc)[:240], completed_at=_now())

    def job_status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return self._public_job(job) if job else None

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            if job["status"] in {"completed", "failed", "cancelled"}:
                return self._public_job(job)
            job["cancel_requested"] = True
            session_id = job["session_id"]
            is_comparison = job_id in self._comparison_jobs
            if is_comparison:
                # A running comparison is intentionally not interrupted between model
                # swaps; queued work can still be cancelled before it starts.
                future = self._futures.get(job_id)
                if future and future.cancel():
                    job.update(status="cancelled", detail="对照体验已取消")
                    return self._public_job(job)
                return self._public_job(job)
        self.manager.cancel(session_id)
        return self.job_status(job_id)

    def close_session(self, session_id: str) -> bool:
        with self._lock:
            state = self._sessions.get(session_id)
            if not state or state["status"] in {"closed", "failed"}:
                return False
            if state["status"] == "loading":
                future = self._futures.get(f"session:{session_id}")
                if future:
                    future.cancel()
                state.update(status="closed", model=None, detail="会话已结束")
                return True
        self.manager.close(session_id)
        with self._lock:
            state = self._sessions.get(session_id)
            if state:
                state.update(status="closed", model=None)
        return True

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            for session in self._sessions.values():
                if session["status"] == "generating":
                    self.manager.cancel(session["session_id"])
        self.manager.close()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("模型体验服务已关闭")

    def _has_active_comparison(self) -> bool:
        return any(self._jobs.get(job_id, {}).get("status") in {"queued", "running"}
                   for job_id in self._comparison_jobs)

    def _prune(self) -> None:
        if len(self._jobs) + len(self._sessions) < MAX_IN_MEMORY_JOBS:
            return
        for key, state in list(self._jobs.items()):
            if state["status"] in {"completed", "failed", "cancelled"}:
                self._jobs.pop(key, None)
                self._futures.pop(key, None)
                self._comparison_jobs.discard(key)
                if len(self._jobs) + len(self._sessions) < MAX_IN_MEMORY_JOBS:
                    break
        for key, state in list(self._sessions.items()):
            if state["status"] in {"closed", "failed"}:
                self._sessions.pop(key, None)
                self._futures.pop(f"session:{key}", None)
                if len(self._jobs) + len(self._sessions) < MAX_IN_MEMORY_JOBS:
                    break

    @staticmethod
    def _public_session(state: dict[str, Any]) -> dict[str, Any]:
        return {key: state.get(key) for key in ("session_id", "model_id", "status", "created_at", "detail", "model")}

    @staticmethod
    def _public_job(state: dict[str, Any]) -> dict[str, Any]:
        return {key: state.get(key) for key in ("job_id", "session_id", "status", "created_at", "completed_at",
                                                "parameters", "result", "detail")}
