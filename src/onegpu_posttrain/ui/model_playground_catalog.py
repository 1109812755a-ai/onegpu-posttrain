"""Verified, server-side catalog for locally loadable model-playground artifacts."""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..lineage import sha256_directory, sha256_file
from .run_workspace import (MAX_RECORD_BYTES, MAX_RUN_RECORDS, RunWorkspace,
                            _has_symlink_component, _read_json, _walk_records)

MAX_MODEL_REGISTRIES = 2000
_MODEL_ID = re.compile(r"^model-[a-zA-Z0-9][a-zA-Z0-9._-]{0,159}$")


def _read_registry(path: Path, workspace: Path) -> dict[str, Any] | None:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RECORD_BYTES:
            return None
        if not path.resolve().is_relative_to(workspace):
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, ValueError, RuntimeError):
        return None


def _has_unsafe_symlink(path: Path, trusted_root: Path, *, allow_file_links: bool = False) -> bool:
    """Reject path escapes; HF cache snapshots may link files only within the cache."""
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            try:
                target = current.resolve(strict=True)
                if not target.is_relative_to(trusted_root):
                    return True
                if current.is_dir() or not allow_file_links:
                    return True
            except (OSError, RuntimeError, ValueError):
                return True
    if path.is_dir():
        for current, dirs, files in os.walk(path, followlinks=False):
            base = Path(current)
            if base.is_symlink():
                return True
            for name in dirs + files:
                child = base / name
                if not child.is_symlink():
                    continue
                try:
                    target = child.resolve(strict=True)
                    if not target.is_relative_to(trusted_root) or child.is_dir() or not allow_file_links:
                        return True
                except (OSError, RuntimeError, ValueError):
                    return True
    return False


def _has_transformers_files(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    weights = ("*.safetensors", "*.safetensors.index.json", "pytorch_model*.bin", "pytorch_model*.bin.index.json")
    return any(next(path.glob(pattern), None) is not None for pattern in weights)


def _timestamp_key(value: Any) -> tuple[int, float]:
    """Sort valid completion timestamps newest-first; put missing/invalid values last."""
    if not isinstance(value, str) or not value.strip():
        return (1, 0.0)
    try:
        normalized = value.strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return (0, -parsed.timestamp())
    except (OverflowError, OSError, TypeError, ValueError):
        return (1, 0.0)


class ModelPlaygroundCatalog:
    """Resolve model IDs only from workspace registries and successful Run lineage."""

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.runs_root = self.workspace / "runs"
        self.run_workspace = RunWorkspace(self.workspace)
        self._snapshot_lock = threading.Lock()
        self._snapshot_status = "idle"
        self._snapshot_models: list[dict[str, Any]] = []
        self._snapshot_error: str | None = None

    def snapshot(self) -> dict[str, Any]:
        """Return immediately while the expensive on-disk digest audit runs once in background."""
        with self._snapshot_lock:
            if self._snapshot_status == "idle":
                self._snapshot_status = "loading"
                threading.Thread(target=self._refresh_snapshot, name="model-catalog-audit", daemon=True).start()
            return {"status": self._snapshot_status,
                    "models": [dict(model) for model in self._snapshot_models],
                    "detail": self._snapshot_error}

    def _refresh_snapshot(self) -> None:
        try:
            models = self.list_models()
            status, detail = "ready", None
        except Exception as exc:
            models, status = [], "failed"
            detail = f"本地模型目录核验失败：{type(exc).__name__}"
        with self._snapshot_lock:
            self._snapshot_models = models
            self._snapshot_status = status
            self._snapshot_error = detail

    def _verified_sources(self) -> dict[str, dict[str, Any]]:
        """Build training-to-merged lineage once; avoid run_detail's per-Run full rescan."""
        rows, _ = self.run_workspace._candidates()
        training_ids = {row.get("run_id") for row in rows
                        if row.get("run_kind") == "training" and row.get("status") == "completed"}
        training_completed_at = {row.get("run_id"): row.get("completed_at") for row in rows
                                 if row.get("run_kind") == "training" and row.get("status") == "completed"}
        if not training_ids:
            return {}
        paths, _ = _walk_records(self.runs_root, {"lineage.json"}, MAX_RUN_RECORDS)
        lineages: dict[str, dict[str, Any]] = {}
        for path in paths:
            if _has_symlink_component(path, self.workspace):
                continue
            record = _read_json(path, self.workspace)
            if record and isinstance(record.get("run_id"), str):
                lineages.setdefault(record["run_id"], record)

        training_outputs: dict[str, list[tuple[str, str]]] = {}
        for run_id in training_ids:
            lineage = lineages.get(run_id)
            if (not lineage or lineage.get("run_type") != "train"
                    or lineage.get("status") != "completed" or lineage.get("lineage_status") != "complete"):
                continue
            for output in lineage.get("outputs", []):
                if not isinstance(output, dict) or output.get("kind") != "model":
                    continue
                model_id = output.get("id")
                digest = output.get("digest")
                digest_value = digest.get("value") if isinstance(digest, dict) else None
                if isinstance(model_id, str) and isinstance(digest_value, str):
                    training_outputs.setdefault(model_id, []).append((run_id, digest_value))

        sources: dict[str, dict[str, Any]] = {}
        try:
            model_paths = list(self.runs_root.glob("**/model_artifacts/*.json"))[:2000]
        except OSError:
            return {}
        for path in model_paths:
            if _has_symlink_component(path, self.workspace):
                continue
            record = _read_registry(path, self.workspace)
            if not record or record.get("kind") != "merged":
                continue
            model_id = record.get("model_id")
            digest = record.get("digest")
            digest_value = digest.get("value") if isinstance(digest, dict) else None
            if not isinstance(model_id, str) or not isinstance(digest_value, str):
                continue
            produced_by = record.get("produced_by_run_id")
            producer = lineages.get(produced_by) if isinstance(produced_by, str) else None
            if (not producer or producer.get("run_type") != "merge"
                    or producer.get("status") != "completed"):
                continue
            output = next((item for item in producer.get("outputs", [])
                           if isinstance(item, dict) and item.get("kind") == "model"
                           and item.get("id") == model_id), None)
            output_digest = output.get("digest") if isinstance(output, dict) else None
            if not isinstance(output_digest, dict) or output_digest.get("value") != digest_value:
                continue
            explicit_inputs = {item.get("id"): item for item in producer.get("inputs", [])
                               if isinstance(item, dict) and item.get("kind") == "model"}
            parents = set(record.get("parent_model_ids", []))
            if isinstance(record.get("adapter_id"), str):
                parents.add(record["adapter_id"])
            for parent_id in parents:
                for source_run_id, source_digest in training_outputs.get(parent_id, []):
                    merge_input = explicit_inputs.get(parent_id)
                    merge_digest = merge_input.get("digest") if isinstance(merge_input, dict) else None
                    if isinstance(merge_digest, dict) and merge_digest.get("value") == source_digest:
                        sources.setdefault(model_id, {
                            "model_id": model_id, "source_run_id": source_run_id,
                            "produced_by_run_id": produced_by, "path_available": True,
                            "training_completed_at": training_completed_at.get(source_run_id),
                        })
                        break
                if model_id in sources:
                    break
        return sources

    def _allowed_local_path(self, value: str, kind: str) -> Path | None:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        try:
            resolved = candidate.resolve(strict=True)
            if not resolved.is_dir():
                return None
            if kind in {"merged", "weight_space_candidate"}:
                if not resolved.is_relative_to(self.workspace) or _has_unsafe_symlink(candidate, self.workspace):
                    return None
            else:
                # Base models may be registered in the workspace or in the local HF cache.
                hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).resolve()
                allowed_roots = (self.workspace, hf_home, hf_home / "hub")
                trusted_root = next((root for root in allowed_roots if resolved.is_relative_to(root)), None)
                if trusted_root is None:
                    return None
                allow_file_links = trusted_root != self.workspace
                if _has_unsafe_symlink(candidate, trusted_root, allow_file_links=allow_file_links):
                    return None
            return resolved
        except (OSError, RuntimeError, ValueError):
            return None

    def list_models(self) -> list[dict[str, Any]]:
        if self.runs_root.is_symlink() or not self.runs_root.is_dir():
            return []
        sources = self._verified_sources()
        try:
            registries = list(self.runs_root.glob("**/model_artifacts/*.json"))[:MAX_MODEL_REGISTRIES]
        except OSError:
            return []
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for registry in registries:
            if _has_symlink_component(registry, self.workspace):
                continue
            record = _read_registry(registry, self.workspace)
            if not record:
                continue
            model_id = record.get("model_id")
            kind = record.get("kind")
            if not isinstance(model_id, str) or not _MODEL_ID.fullmatch(model_id) or model_id in seen:
                continue
            if kind not in {"base", "merged", "weight_space_candidate"}:
                continue
            seen.add(model_id)
            source = sources.get(model_id) if kind == "merged" else None
            reason = None
            digest = record.get("digest") if isinstance(record.get("digest"), dict) else {}
            expected = digest.get("value")
            if digest.get("status") != "available" or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                reason = "模型摘要登记不可用"
            path_value = record.get("path")
            path = self._allowed_local_path(path_value, kind) if isinstance(path_value, str) and path_value else None
            if not reason and path is None:
                reason = "模型路径缺失、越界或不可访问"
            if not reason and path and not _has_transformers_files(path):
                reason = "缺少 Transformers 配置或权重文件"
            if not reason and kind == "merged" and not source:
                reason = "缺少成功训练 Run 的已验证模型血缘"
            elif not reason and kind == "merged" and source.get("produced_by_run_id") != record.get("produced_by_run_id"):
                reason = "模型登记与已验证 Run 血缘不一致"
            if not reason and kind == "weight_space_candidate":
                edit_path = record.get("edit_artifact_path")
                if (not isinstance(edit_path, str) or not edit_path
                        or not isinstance(record.get("parent_model_ids"), list)
                        or len(record["parent_model_ids"]) != 1):
                    reason = "候选模型缺少父模型或编辑产物血缘"
                else:
                    edit_file = Path(edit_path)
                    if not edit_file.is_absolute():
                        edit_file = self.workspace / edit_file
                    try:
                        edit_resolved = edit_file.resolve(strict=True)
                        if (not edit_resolved.is_relative_to(self.workspace) or edit_file.is_symlink()
                                or edit_resolved.parent != path.parent):
                            raise ValueError("candidate edit path mismatch")
                        edit_record = _read_json(edit_resolved, self.workspace)
                        if (not edit_record or edit_record.get("candidate_sha256") != expected
                                or not isinstance(edit_record.get("parent_sha256"), str)):
                            raise ValueError("candidate edit digest mismatch")
                    except (OSError, ValueError, RuntimeError):
                        reason = "候选模型编辑产物无效"
            result.append({
                "model_id": model_id,
                "kind": kind,
                "stage": record.get("stage") if isinstance(record.get("stage"), str) else "unknown",
                "digest": expected if isinstance(expected, str) else None,
                "source_run_id": source.get("source_run_id") if source else None,
                "produced_by_run_id": record.get("produced_by_run_id"),
                "training_completed_at": source.get("training_completed_at") if source else None,
                "available": reason is None,
                "unavailable_reason": reason,
                # Kept server-side for the inference resolver; never serialize this field from the API.
                "_resolved_path": str(path) if path and reason is None else None,
            })
        return sorted(result, key=lambda model: (
            not model["available"], *_timestamp_key(model.get("training_completed_at")),
            model["stage"], model["model_id"],
        ))

    def resolve(self, model_id: str) -> dict[str, Any] | None:
        snapshot = self.snapshot()
        if snapshot["status"] != "ready":
            raise RuntimeError("本地模型目录仍在核验；请稍后重试")
        model = next((model for model in snapshot["models"]
                      if model["model_id"] == model_id and model["available"]), None)
        if model is None:
            return None
        path = Path(model["_resolved_path"])
        expected = model.get("digest")
        try:
            actual = sha256_directory(path) if path.is_dir() else sha256_file(path)
        except (OSError, ValueError, RuntimeError) as exc:
            raise RuntimeError("无法核验所选模型的内容摘要") from exc
        if not isinstance(expected, str) or actual != expected:
            raise RuntimeError("模型内容摘要与登记值不匹配，已阻止加载")
        return model
