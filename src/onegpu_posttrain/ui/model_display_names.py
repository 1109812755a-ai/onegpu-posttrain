"""Workspace-local presentation labels for immutable model identities."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

MAX_DISPLAY_NAME_CHARS = 80
MAX_DISPLAY_NAME_RECORD_BYTES = 1024 * 1024
MAX_DISPLAY_NAME_RECORDS = 2000
_MODEL_ID = re.compile(r"^model-[a-zA-Z0-9][a-zA-Z0-9._-]{0,159}$")
_STORE_LOCK = threading.RLock()


class DisplayNameStoreError(RuntimeError):
    """Raised when the local display-name sidecar cannot be safely updated."""


def validate_display_name(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("显示名必须是文本")
    label = value.strip()
    if not label:
        raise ValueError("显示名不能为空")
    if len(label) > MAX_DISPLAY_NAME_CHARS:
        raise ValueError(f"显示名不能超过 {MAX_DISPLAY_NAME_CHARS} 个字符")
    if any(unicodedata.category(char) == "Cc" for char in label):
        raise ValueError("显示名不能包含控制字符或换行")
    return label


class ModelDisplayNameStore:
    """Persist aliases separately from model/Run artifacts, with atomic replacement."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()
        self.directory = self.workspace / ".onegpu-posttrain"
        self.path = self.directory / "model-display-names.json"

    def _check_paths(self, *, create: bool = False) -> None:
        if self.directory.is_symlink():
            raise DisplayNameStoreError("模型显示名目录不能是符号链接")
        if create:
            self.directory.mkdir(parents=True, exist_ok=True)
        try:
            if self.directory.exists() and not self.directory.resolve().is_relative_to(self.workspace):
                raise DisplayNameStoreError("模型显示名目录越出工作区")
        except (OSError, RuntimeError, ValueError) as exc:
            if isinstance(exc, DisplayNameStoreError):
                raise
            raise DisplayNameStoreError("无法安全访问模型显示名目录") from exc
        if self.path.is_symlink():
            raise DisplayNameStoreError("模型显示名文件不能是符号链接")

    def _read_raw(self) -> tuple[dict[str, str], bool]:
        self._check_paths()
        if not self.path.exists():
            return {}, False
        try:
            if not self.path.is_file():
                return {}, True
            if self.path.stat().st_size > MAX_DISPLAY_NAME_RECORD_BYTES:
                raise DisplayNameStoreError("模型显示名记录超过读取上限；原文件未更改")
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if (not isinstance(payload, dict) or payload.get("schema") != "model-display-names-v1"
                    or not isinstance(payload.get("display_names"), dict)):
                return {}, True
            names: dict[str, str] = {}
            rows = payload["display_names"]
            if len(rows) > MAX_DISPLAY_NAME_RECORDS:
                return {}, True
            for model_id, value in rows.items():
                if not isinstance(model_id, str) or not _MODEL_ID.fullmatch(model_id):
                    return {}, True
                names[model_id] = validate_display_name(value)
            return names, False
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
            return {}, True

    def get_all(self, registered_model_ids: Iterable[str] | None = None) -> dict[str, str]:
        """Return known labels; stale-but-valid keys are ignored when IDs are supplied."""
        with _STORE_LOCK:
            names, _ = self._read_raw()
            if registered_model_ids is None:
                return names
            known = set(registered_model_ids)
            return {model_id: name for model_id, name in names.items() if model_id in known}

    def set(self, model_id: str, display_name: object, *, registered_model_ids: Iterable[str]) -> str:
        self._validate_registered(model_id, registered_model_ids)
        name = validate_display_name(display_name)
        self._update(model_id, name)
        return name

    def clear(self, model_id: str, *, registered_model_ids: Iterable[str]) -> None:
        self._validate_registered(model_id, registered_model_ids)
        self._update(model_id, None)

    @staticmethod
    def _validate_registered(model_id: str, registered_model_ids: Iterable[str]) -> None:
        if not isinstance(model_id, str) or not _MODEL_ID.fullmatch(model_id):
            raise ValueError("model_id 无效")
        if model_id not in set(registered_model_ids):
            raise KeyError("模型尚未登记，不能设置显示名")

    def _update(self, model_id: str, display_name: str | None) -> None:
        with _STORE_LOCK:
            names, malformed = self._read_raw()
            if display_name is None:
                names.pop(model_id, None)
            else:
                names[model_id] = display_name
            if len(names) > MAX_DISPLAY_NAME_RECORDS:
                raise DisplayNameStoreError(f"工作区模型显示名数量超过 {MAX_DISPLAY_NAME_RECORDS}")
            self._check_paths(create=True)
            backup_path: Path | None = None
            if malformed and self.path.exists():
                backup_path = self.path.with_name(
                    f"{self.path.stem}.corrupt-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}{self.path.suffix}"
                )
                try:
                    shutil.copy2(self.path, backup_path)
                except OSError as exc:
                    raise DisplayNameStoreError("损坏的显示名记录无法备份；原文件未更改") from exc
            temp_name: str | None = None
            try:
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=self.directory,
                                                 prefix=".model-display-names-", suffix=".tmp", delete=False) as stream:
                    temp_name = stream.name
                    json.dump({"schema": "model-display-names-v1", "display_names": names},
                              stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    stream.flush()
                    os.fsync(stream.fileno())
                if Path(temp_name).stat().st_size > MAX_DISPLAY_NAME_RECORD_BYTES:
                    raise DisplayNameStoreError("模型显示名记录超过存储上限")
                os.replace(temp_name, self.path)
            except (OSError, ValueError) as exc:
                if isinstance(exc, DisplayNameStoreError):
                    raise
                raise DisplayNameStoreError("无法原子更新模型显示名；现有记录保持不变") from exc
            finally:
                if temp_name and os.path.exists(temp_name):
                    try:
                        os.unlink(temp_name)
                    except OSError:
                        pass

