"""Persist provider attempt state across independent executions of a bundle."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from .cache import write_response_cache
from .ledger import RequestLedger
from .providers import MissingCredentialError, ProviderError


class UncertainProviderRequest(ValueError):
    """A prior attempt has an unknown remote outcome and requires review."""


@contextmanager
def _exclusive_request_lock(bundle_dir: Path):
    """Serialize attempt/review updates across processes; locks release on crash."""
    lock_path = bundle_dir / "generation-requests.lock"
    if bundle_dir.is_symlink() or lock_path.is_symlink():
        raise ValueError("生成请求锁路径不可用")
    with lock_path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise UncertainProviderRequest("同一方案的请求正在执行或审查，请稍后重试") from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def ensure_no_uncertain_requests(bundle_dir: Path) -> None:
    """Check an existing ledger without creating one during preflight."""
    ledger_path = bundle_dir / "generation-requests.jsonl"
    if bundle_dir.is_symlink() or ledger_path.is_symlink():
        raise ValueError("生成请求账本路径不可用")
    if not ledger_path.exists():
        return
    ledger = RequestLedger(ledger_path)
    if any(row.get("state") in {"pending", "uncertain"}
           for row in ledger._latest_by_attempt.values()):
        raise UncertainProviderRequest("上次生成请求结果不确定；请先核查 Provider 记录，不能自动重试")


def unresolved_attempts(bundle_dir: Path) -> list[dict]:
    ledger_path = bundle_dir / "generation-requests.jsonl"
    if bundle_dir.is_symlink() or ledger_path.is_symlink():
        raise ValueError("生成请求账本路径不可用")
    if not ledger_path.exists():
        return []
    ledger = RequestLedger(ledger_path)
    return [{"attempt_id": row["attempt_id"], "execution_id": row.get("execution_id"),
             "state": row["state"], "timestamp_utc": row.get("timestamp_utc")}
            for row in ledger._latest_by_attempt.values() if row.get("state") in {"pending", "uncertain"}]


def acknowledge_uncertain_attempt(bundle_dir: Path, attempt_id: str) -> None:
    """Record acknowledgement of billing risk without claiming the request failed."""
    with _exclusive_request_lock(bundle_dir):
        if not any(row["attempt_id"] == attempt_id for row in unresolved_attempts(bundle_dir)):
            raise ValueError("未知请求不存在或已处理")
        ledger = RequestLedger(bundle_dir / "generation-requests.jsonl")
        previous = ledger._latest_by_attempt[attempt_id]
        ledger.append({"attempt_id": attempt_id, "execution_id": previous.get("execution_id"),
                       "request_sha256": previous.get("request_sha256"), "state": "reviewed_unknown",
                       "note": "用户确认已核查并了解可能重复计费；原请求结果仍未知"})


class RequestGuard:
    """Fail closed on unknown prior outcomes; never persist prompts or credentials."""

    def __init__(self, bundle_dir: Path, execution_id: str, provider):
        ledger_path = bundle_dir / "generation-requests.jsonl"
        if bundle_dir.is_symlink() or ledger_path.is_symlink():
            raise ValueError("生成请求账本路径不可用")
        ensure_no_uncertain_requests(bundle_dir)
        self.ledger = RequestLedger(ledger_path)
        self.bundle_dir = bundle_dir
        self.execution_id = execution_id
        self.provider = provider

    def generate(self, profile, request):
        attempt_id = uuid4().hex
        request_sha = hashlib.sha256(json.dumps(request.model_dump(mode="json"), sort_keys=True,
                                                  ensure_ascii=False).encode("utf-8")).hexdigest()
        event = {"attempt_id": attempt_id, "execution_id": self.execution_id,
                 "request_sha256": request_sha, "timeout_seconds": request.timeout_seconds}
        with _exclusive_request_lock(self.bundle_dir):
            ensure_no_uncertain_requests(self.bundle_dir)
            self.ledger = RequestLedger(self.bundle_dir / "generation-requests.jsonl")
            self.ledger.append({**event, "state": "pending"})
            try:
                response = self.provider.generate(profile, request)
            except MissingCredentialError:
                self.ledger.append({**event, "state": "failed", "error_category": "missing_credential",
                                    "exception_type": "MissingCredentialError"})
                raise ProviderError("missing_credential", "本地 Provider 凭据不可用；请求未发送",
                                    retryable=False, uncertain=False,
                                    timeout_seconds=request.timeout_seconds) from None
            except ProviderError as exc:
                self.ledger.append({**event, "state": "uncertain" if exc.uncertain else "failed",
                                    "error_category": exc.category, "status_code": exc.status_code,
                                    "provider_request_id": exc.provider_request_id,
                                    "timeout_seconds": exc.timeout_seconds or request.timeout_seconds})
                raise
            except Exception as exc:
                self.ledger.append({**event, "state": "uncertain", "error_category": "unexpected_provider_error",
                                    "exception_type": type(exc).__name__})
                raise ProviderError("unexpected_provider_error", "provider outcome unknown", retryable=False,
                                    uncertain=True, timeout_seconds=request.timeout_seconds) from None
            # Candidate slots can have identical prompts; use the unique attempt as
            # part of the cache key so independent responses are never conflated.
            cache_key = hashlib.sha256(f"{self.execution_id}:{attempt_id}:{request_sha}".encode()).hexdigest()
            cache_path, response_digest = write_response_cache(self.bundle_dir, cache_key, response)
            self.ledger.append({**event, "state": "completed", "cache_key": cache_key,
                                "response_cache_path": cache_path, "response_sha256": response_digest,
                                "prompt_tokens": response.prompt_tokens,
                                "completion_tokens": response.completion_tokens,
                                "provider_request_id": response.provider_request_id})
            return response

    def provider_diagnostics(self) -> list[dict]:
        """Return safe identifiers and error facts, never prompts or credentials."""
        allowed = ("execution_id", "state", "error_category", "exception_type", "status_code",
                   "provider_request_id", "timeout_seconds", "timestamp_utc")
        rows = [row for row in self.ledger._latest_by_attempt.values()
                if row.get("state") in {"failed", "uncertain"}]
        return [{key: row[key] for key in allowed if row.get(key) is not None} for row in rows[-50:]]
