"""Bounded, read-only adapter for recorded local training and evaluation runs."""

from __future__ import annotations

import json
import os
from itertools import islice
from pathlib import Path
from typing import Any

MAX_RUN_RECORDS = 5000
MAX_SCAN_DIRS = 20000
MAX_RECORD_BYTES = 512 * 1024
MAX_LOG_BYTES = 64 * 1024
MAX_OUTPUTS = 80
_PRUNED_DIRS = {"checkpoint", "checkpoints", "merged", "adapters", "model", "models", "__pycache__"}


def _read_json(path: Path, root: Path, limit: int = MAX_RECORD_BYTES) -> dict[str, Any] | None:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
            return None
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            return None
        value = json.loads(resolved.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, ValueError):
        return None


def _has_symlink_component(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _walk_records(root: Path, filenames: set[str], remaining: int) -> tuple[list[Path], bool]:
    found: list[Path] = []
    if root.is_symlink() or not root.is_dir():
        return found, False
    truncated = False
    visited = 0
    for current, dirs, files in os.walk(root, followlinks=False):
        visited += 1
        if len(found) >= remaining or visited > MAX_SCAN_DIRS:
            truncated = True
            break
        current_path = Path(current)
        try:
            if current_path.is_symlink() or not current_path.resolve().is_relative_to(root.resolve()):
                dirs[:] = []
                continue
        except OSError:
            dirs[:] = []
            continue
        dirs[:] = [name for name in dirs if name.lower() not in _PRUNED_DIRS
                   and not (current_path / name).is_symlink()]
        for name in filenames.intersection(files):
            path = current_path / name
            if not path.is_symlink():
                found.append(path)
                if len(found) >= remaining:
                    truncated = True
                    return found, truncated
    return found, truncated


class RunWorkspace:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.runs_root = self.workspace / "runs"
        self.launcher_root = self.workspace / ".onegpu-posttrain" / "launcher" / "runs"

    def _candidates(self) -> tuple[list[dict[str, Any]], bool]:
        records, truncated = _walk_records(self.runs_root, {"run.json", "evaluation_run.json"}, MAX_RUN_RECORDS)
        by_id: dict[str, dict[str, Any]] = {}

        def add(path: Path, record: dict[str, Any], source: str) -> None:
            run_id = record.get("run_id")
            if not isinstance(run_id, str) or not run_id or len(run_id) > 160:
                return
            relative = path.parent.relative_to(self.workspace).as_posix()
            observed_status = record.get("status")
            if not isinstance(observed_status, str):
                observed_status = (record.get("train") or {}).get("status") if isinstance(record.get("train"), dict) else None
            if not isinstance(observed_status, str):
                merge = record.get("merge")
                execution = merge.get("execution") if isinstance(merge, dict) else None
                observed_status = execution.get("status") if isinstance(execution, dict) else None
            normalized_status = {"PASS": "completed", "SUCCESS": "completed", "COMPLETED": "completed",
                                 "FAILED": "failed", "OOM": "oom", "PARTIAL": "partial", "INTERRUPTED": "interrupted"}.get(
                                     observed_status.upper() if isinstance(observed_status, str) else "", observed_status or "unknown")
            candidate = {
                "run_id": run_id,
                "run_kind": "evaluation" if source == "evaluation" or record.get("run_kind") == "evaluation" else "training",
                "stage": record.get("stage") if isinstance(record.get("stage"), str) else ("evaluate" if source == "evaluation" else None),
                "status": normalized_status,
                "started_at": record.get("started_at") if isinstance(record.get("started_at"), str) else None,
                "completed_at": record.get("completed_at") if isinstance(record.get("completed_at"), str) else None,
                "run_label": record.get("run_label") if isinstance(record.get("run_label"), str) else None,
                "experiment_id": record.get("experiment_id") if isinstance(record.get("experiment_id"), str) else None,
                "parent_run_id": record.get("parent_run_id") if isinstance(record.get("parent_run_id"), str) else None,
                "recipe_path": record.get("recipe_path") if isinstance(record.get("recipe_path"), str) else None,
                "source": source,
                "source_ref": relative,
                "source_refs": [relative],
                "outputs": [],
                "warnings": [],
            }
            old = by_id.get(run_id)
            if old is None:
                by_id[run_id] = candidate
                return
            if relative not in old["source_refs"]:
                old["source_refs"].append(relative)
            if old["status"] != candidate["status"]:
                old["warnings"].append("同一 run_id 的运行记录状态不一致；保留各自来源供详情核对")
                old["status_conflict"] = True
            # Launcher state is the supervisor's authoritative latest lifecycle record;
            # otherwise preserve the first stable filesystem record and expose conflicts.
            if source == "launcher_state":
                source_refs = old["source_refs"] + [relative]
                old.update({key: value for key, value in candidate.items() if value is not None})
                old["source_refs"] = sorted(set(source_refs))
            elif not old.get("stage") and candidate.get("stage"):
                old["stage"] = candidate["stage"]

        for path in records:
            source = "evaluation" if path.name == "evaluation_run.json" else "cli"
            record = _read_json(path, self.workspace)
            if record:
                add(path, record, source)
        state_files: list[Path] = []
        if not self.launcher_root.is_symlink() and self.launcher_root.is_dir():
            try:
                state_files = list(islice(self.launcher_root.glob("*.json"), MAX_RUN_RECORDS + 1))
                if len(state_files) > MAX_RUN_RECORDS:
                    truncated = True
                    state_files = state_files[:MAX_RUN_RECORDS]
            except OSError:
                state_files = []
        for path in state_files:
            record = _read_json(path, self.workspace, 128 * 1024)
            if record:
                add(path, record, "launcher_state")
        return sorted(by_id.values(), key=lambda item: (item.get("started_at") or "", item["run_id"]), reverse=True), truncated

    def list_runs(self, *, stage: str | None = None, status: str | None = None,
                  q: str | None = None, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        limit = max(1, min(int(limit), 200))
        offset = max(0, min(int(offset), 5000))
        rows, truncated = self._candidates()
        if stage:
            rows = [row for row in rows if row.get("stage") == stage]
        if status:
            rows = [row for row in rows if row.get("status") == status]
        if q:
            needle = q.casefold()[:120]
            rows = [row for row in rows if needle in " ".join(str(row.get(key) or "") for key in
                    ("run_id", "run_label", "experiment_id", "stage", "source_ref")).casefold()]
        total = len(rows)
        return {"runs": rows[offset:offset + limit], "total": total, "limit": limit,
                "offset": offset, "partial_scan": truncated}

    def run_detail(self, run_id: str) -> dict[str, Any] | None:
        summary = next((item for item in self._candidates()[0] if item["run_id"] == run_id), None)
        if summary is None:
            return None
        detail = dict(summary)
        detail.update(config=None, metrics=[], logs=None, lineage=None, evaluations=[], provenance=[], produced_artifacts=[], handoff_models=[])
        refs = summary.get("source_refs", [])
        bases: list[Path] = []
        for ref in refs:
            source_base = self.workspace / ref
            if ref.replace("\\", "/").startswith(".onegpu-posttrain/launcher/runs/") or ref.replace("\\", "/") == ".onegpu-posttrain/launcher/runs":
                state_path = source_base / f"{run_id}.json"
                state = _read_json(state_path, self.workspace, 128 * 1024)
                if state and state.get("run_id") == run_id:
                    detail["launcher_record"] = state
                    output_ref = state.get("output")
                    if isinstance(output_ref, str) and output_ref:
                        output_base = self.workspace / output_ref
                        if not output_base.is_symlink() and output_base.resolve().is_relative_to(self.workspace):
                            bases.append(output_base)
            else:
                bases.append(source_base)
        for base in bases:
            if not base.resolve().is_relative_to(self.workspace) or base.is_symlink():
                continue
            for filename in ("run.json", "evaluation_run.json"):
                record = _read_json(base / filename, self.workspace)
                if record and record.get("run_id") == run_id:
                    detail["normalized_record"] = record
                    break
            for filename in ("lineage.json", "evaluation_lineage.json"):
                value = _read_json(base / filename, self.workspace)
                if value:
                    detail["lineage"] = value
                    detail["provenance"] = value.get("inputs", []) if isinstance(value.get("inputs"), list) else []
                    detail["produced_artifacts"] = value.get("outputs", []) if isinstance(value.get("outputs"), list) else []
                    lineage_metrics = value.get("metrics")
                    if isinstance(lineage_metrics, dict):
                        detail["metrics"] = [{"name": key, "metric": key.rsplit(".", 1)[-1],
                                               "value": item.get("value"), "denominator": item.get("denominator")}
                                              for key, item in lineage_metrics.items()
                                              if isinstance(item, dict) and isinstance(item.get("value"), (int, float))]
            for filename in ("resolved_config.json", "eval.json"):
                value = _read_json(base / filename, self.workspace, 2 * 1024 * 1024)
                if value:
                    if filename == "resolved_config.json":
                        detail["config"] = value
                    else:
                        detail["metrics"] = value.get("metrics", []) if isinstance(value.get("metrics"), list) else []
                        metric_statuses = value.get("metric_statuses")
                        if isinstance(metric_statuses, dict):
                            status_rows = []
                            for name, item in metric_statuses.items():
                                if not isinstance(name, str) or not isinstance(item, dict):
                                    continue
                                metric_value = item.get("value")
                                status_rows.append({
                                    "name": name, "metric": "value",
                                    "value": metric_value if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool) else None,
                                    "denominator": item.get("denominator") if isinstance(item.get("denominator"), int) else None,
                                    "denominator_unit": item.get("denominator_unit"),
                                    "status": item.get("status") if isinstance(item.get("status"), str) else "unknown",
                                    "reason": item.get("reason") if isinstance(item.get("reason"), str) else None,
                                })
                            if status_rows:
                                detail["metrics"] = status_rows
                        if not detail["metrics"] and isinstance(value.get("evalscope_reports"), dict):
                            for name, report in value["evalscope_reports"].items():
                                if isinstance(report, dict) and isinstance(report.get("score"), (int, float)):
                                    detail["metrics"].append({"name": name, "metric": "score", "value": report["score"],
                                                              "denominator": report.get("samples")})
            for filename in ("launcher.log", "logs/train/stdout.log", "logs/evalscope/stdout.log"):
                log_path = base / filename
                if log_path.is_symlink() or not log_path.is_file() or not log_path.resolve().is_relative_to(self.workspace):
                    continue
                try:
                    with log_path.open("rb") as handle:
                        handle.seek(max(0, log_path.stat().st_size - MAX_LOG_BYTES))
                        data = handle.read(MAX_LOG_BYTES)
                    detail["logs"] = {"path": log_path.relative_to(self.workspace).as_posix(),
                                      "text": data.decode("utf-8", errors="replace"),
                                      "truncated": log_path.stat().st_size > MAX_LOG_BYTES}
                    break
                except OSError:
                    continue
            outputs = []
            for filename in ("run.json", "lineage.json", "evaluation_run.json", "evaluation_lineage.json", "eval.json", "resolved_config.json", "launcher.log"):
                path = base / filename
                if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(self.workspace):
                    outputs.append(path.relative_to(self.workspace).as_posix())
            detail["outputs"] = sorted(set(detail["outputs"] + outputs))[:MAX_OUTPUTS]
        record = detail.get("normalized_record") or detail.get("launcher_record") or {}
        if not detail.get("stage") and isinstance(record.get("stage"), str):
            detail["stage"] = record["stage"]
        if not detail.get("run_label") and isinstance(record.get("run_label"), str):
            detail["run_label"] = record["run_label"]
        if summary.get("run_kind") == "evaluation":
            parent_id = (summary.get("parent_run_id") or record.get("parent_run_id")
                         or (detail.get("lineage") or {}).get("training_run_id"))
            detail["parent_run_id"] = parent_id if isinstance(parent_id, str) else None
            if detail["parent_run_id"]:
                detail["parent_available"] = any(row["run_id"] == detail["parent_run_id"] for row in self._candidates()[0])
        else:
            detail["evaluations"] = [row for row in self._candidates()[0]
                                      if row.get("run_kind") == "evaluation" and
                                      row.get("parent_run_id") == run_id]
        detail.pop("launcher_record", None)
        detail.pop("normalized_record", None)
        detail["handoff_models"] = self._verified_handoff_models(run_id, detail.get("lineage"))
        if summary.get("status_conflict"):
            detail["warnings"] = summary["warnings"]
        detail.pop("launcher_record", None)
        detail.pop("normalized_record", None)
        return detail

    def _verified_handoff_models(self, source_run_id: str, lineage: Any) -> list[dict[str, Any]]:
        """Find model outputs directly or through an explicitly recorded successful merge Run."""
        if not isinstance(lineage, dict) or lineage.get("lineage_status") != "complete":
            return []
        source_outputs = {item.get("id"): item for item in lineage.get("outputs", [])
                          if isinstance(item, dict) and item.get("kind") == "model" and isinstance(item.get("id"), str)}
        if not source_outputs or self.runs_root.is_symlink() or not self.runs_root.is_dir():
            return []

        model_paths = list(self.runs_root.glob("**/model_artifacts/*.json"))[:2000]
        lineage_paths, _ = _walk_records(self.runs_root, {"lineage.json"}, MAX_RUN_RECORDS)
        runs_by_id: dict[str, dict[str, Any]] = {}
        for path in lineage_paths:
            record = _read_json(path, self.workspace)
            if record and isinstance(record.get("run_id"), str):
                runs_by_id.setdefault(record["run_id"], record)

        grouped: dict[str, dict[str, Any]] = {}
        counts: dict[str, int] = {}
        for path in model_paths:
            if _has_symlink_component(path, self.workspace):
                continue
            record = _read_json(path, self.workspace)
            if not record or not isinstance(record.get("model_id"), str):
                continue
            model_id = record["model_id"]
            counts[model_id] = counts.get(model_id, 0) + 1
            if model_id in grouped:
                continue
            digest = record.get("digest") if isinstance(record.get("digest"), dict) else {}
            if digest.get("status") != "available" or not isinstance(digest.get("value"), str):
                continue
            model_path = record.get("path")
            if not isinstance(model_path, str) or not model_path:
                continue
            artifact_path = Path(model_path)
            if not artifact_path.is_absolute():
                artifact_path = self.workspace / artifact_path
            if _has_symlink_component(artifact_path, self.workspace):
                continue
            try:
                artifact_path = artifact_path.resolve(strict=True)
                if not artifact_path.is_relative_to(self.workspace) or not artifact_path.exists():
                    continue
            except (OSError, RuntimeError):
                continue

            produced_by = record.get("produced_by_run_id")
            source_relation = False
            if produced_by == source_run_id and model_id in source_outputs:
                output_digest = (source_outputs[model_id].get("digest") or {}).get("value")
                source_relation = output_digest == digest.get("value")
            elif isinstance(produced_by, str):
                producer = runs_by_id.get(produced_by)
                if producer and producer.get("run_type") == "merge" and producer.get("status") == "completed":
                    explicit_inputs = {item.get("id"): item for item in producer.get("inputs", [])
                                       if isinstance(item, dict) and item.get("kind") == "model"}
                    explicit_outputs = {item.get("id"): item for item in producer.get("outputs", [])
                                        if isinstance(item, dict) and item.get("kind") == "model"}
                    registered_parents = set(record.get("parent_model_ids", []))
                    adapter_id = record.get("adapter_id")
                    if isinstance(adapter_id, str) and adapter_id:
                        registered_parents.add(adapter_id)
                    for input_id, source_ref in source_outputs.items():
                        merge_input = explicit_inputs.get(input_id)
                        source_digest = (source_ref.get("digest") or {}).get("value")
                        merge_digest = ((merge_input or {}).get("digest") or {}).get("value")
                        merge_output = explicit_outputs.get(model_id)
                        output_digest = ((merge_output or {}).get("digest") or {}).get("value")
                        if (input_id in registered_parents and merge_input and source_digest
                                and merge_digest == source_digest and output_digest == digest.get("value")):
                            source_relation = True
                            break
            if not source_relation:
                continue
            grouped[model_id] = {
                "model_id": model_id, "stage": record.get("stage"), "kind": record.get("kind"),
                "produced_by_run_id": produced_by, "source_run_id": source_run_id,
                "digest": digest, "path_available": True, "registry_count": 1,
            }
        for model_id, candidate in grouped.items():
            candidate["registry_count"] = counts.get(model_id, 0)
        return sorted(grouped.values(), key=lambda item: (str(item.get("stage")), str(item.get("model_id"))))

    def read_artifact(self, run_id: str, filename: str) -> tuple[bytes, str] | None:
        allowed = {"run.json", "lineage.json", "evaluation_run.json", "evaluation_lineage.json",
                   "eval.json", "resolved_config.json", "launcher.log"}
        if filename not in allowed:
            return None
        detail = self.run_detail(run_id)
        if detail is None:
            return None
        matches = [self.workspace / relative for relative in detail["outputs"]
                   if Path(relative).name == filename]
        if len(matches) != 1:
            return None
        path = matches[0]
        limit = 2 * 1024 * 1024 if filename.endswith(".json") else MAX_LOG_BYTES
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > limit or not path.resolve().is_relative_to(self.workspace):
                return None
            mime = "application/json; charset=utf-8" if filename.endswith(".json") else "text/plain; charset=utf-8"
            return path.read_bytes(), mime
        except OSError:
            return None
