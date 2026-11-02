"""Bounded, read-only access to Mini unified evaluation matrices."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

MAX_MATRIX_FILES = 100
MAX_MATRIX_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 10 * 1024 * 1024
MAX_MODELS = 100
MAX_METRICS = 50
MAX_LINEAGE_FILES = 2000
MAX_LINEAGE_BYTES = 256 * 1024
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,200}$")
_METRIC_UNITS = {"gsm8k": "ratio", "math_500": "ratio", "internal_acc": "ratio", "domain_ppl": "ppl"}


def _identity(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:20]


def discover_matrix_files(workspace: Path) -> list[tuple[str, Path]]:
    """Discover only direct Mini experiment matrix locations; never follow links."""
    workspace = workspace.resolve()
    runs_root = workspace / "runs"
    mini_root = workspace / "runs" / "mini"
    if (runs_root.is_symlink() or mini_root.is_symlink() or not mini_root.is_dir()
            or not mini_root.resolve().is_relative_to(workspace)):
        return []

    found: list[tuple[str, Path]] = []
    total_bytes = 0
    try:
        experiments = sorted(mini_root.iterdir(), key=lambda item: item.name)
    except OSError:
        return []
    for experiment in experiments:
        if len(found) >= MAX_MATRIX_FILES:
            break
        if experiment.is_symlink() or not experiment.is_dir():
            continue
        try:
            matrix_dirs = sorted(
                (item for item in experiment.iterdir() if item.name.startswith("evaluation-matrix")),
                key=lambda item: item.name,
            )
        except OSError:
            continue
        for matrix_dir in matrix_dirs:
            if len(found) >= MAX_MATRIX_FILES:
                break
            path = matrix_dir / "evaluation_matrix.json"
            try:
                if (matrix_dir.is_symlink() or not matrix_dir.is_dir() or path.is_symlink() or not path.is_file()
                        or not matrix_dir.resolve().is_relative_to(mini_root.resolve())
                        or not path.resolve().is_relative_to(matrix_dir.resolve())):
                    continue
                size = path.stat().st_size
                if size > MAX_MATRIX_BYTES or total_bytes + size > MAX_TOTAL_BYTES:
                    found.append((_identity(path.relative_to(workspace).as_posix()), path))
                    continue
                total_bytes += size
                found.append((_identity(path.relative_to(workspace).as_posix()), path))
            except (OSError, ValueError, RecursionError):
                continue
    return found


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _text(value: Any, limit: int = 300) -> str | None:
    return value[:limit] if isinstance(value, str) and value else None


def _identifier(value: Any) -> str | None:
    return value if isinstance(value, str) and _ID_RE.fullmatch(value) else None


def _string_list(value: Any, limit: int = 100) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item[:200] for item in value[:limit] if isinstance(item, str) and item]


def _resolve_lineage(workspace: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Index bounded run lineage metadata by recorded IDs only (no path guessing)."""
    root = workspace.resolve() / "runs"
    runs: dict[str, dict[str, Any]] = {}
    models: dict[str, dict[str, Any]] = {}
    evaluations: dict[str, dict[str, Any]] = {}
    if root.is_symlink() or not root.is_dir() or not root.resolve().is_relative_to(workspace.resolve()):
        return runs, models, evaluations
    scanned = 0
    for current, dirs, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        try:
            if current_path.is_symlink() or not current_path.resolve().is_relative_to(root.resolve()):
                dirs[:] = []
                continue
            if len(current_path.relative_to(root).parts) > 8:
                dirs[:] = []
                continue
        except (OSError, ValueError):
            dirs[:] = []
            continue
        dirs[:] = sorted(name for name in dirs if not (current_path / name).is_symlink()
                         and (current_path / name).resolve().is_relative_to(root.resolve()))
        for name in files:
            if name not in {"lineage.json", "evaluation_run.json", "evaluation_lineage.json"}:
                continue
            scanned += 1
            if scanned > MAX_LINEAGE_FILES:
                return runs, models, evaluations
            path = current_path / name
            try:
                if path.is_symlink() or path.stat().st_size > MAX_LINEAGE_BYTES:
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    continue
                if name == "evaluation_lineage.json":
                    eval_id = _identifier(payload.get("eval_id"))
                    if eval_id:
                        evaluations[eval_id] = {
                            "run_id": _identifier(payload.get("produced_by_run_id")),
                            "model_id": _identifier(payload.get("model_id")),
                            "dataset_ids": [_identifier(item) for item in _string_list(payload.get("dataset_ids")) if _identifier(item)],
                            "protocol_version": _text(payload.get("protocol_version"), 100),
                            "config_sha256": _text(payload.get("config_sha256"), 128),
                            "status": _text(payload.get("status"), 50),
                        }
                    continue
                run_id = _identifier(payload.get("run_id"))
                if not run_id or payload.get("lineage_schema_version") != "1.0":
                    continue
                inputs = payload.get("inputs") if isinstance(payload.get("inputs"), list) else []
                outputs = payload.get("outputs") if isinstance(payload.get("outputs"), list) else []
                parent_ids = [_identifier(item.get("id")) for item in inputs
                              if isinstance(item, dict) and item.get("kind") == "model" and item.get("role") in {"parent", "base"}]
                dataset_ids = [_identifier(item.get("id")) for item in inputs
                               if isinstance(item, dict) and item.get("kind") == "dataset"]
                record = {"run_id": run_id, "run_type": _text(payload.get("run_type"), 50),
                          "status": _text(payload.get("status"), 50),
                          "parent_model_ids": [item for item in parent_ids if item],
                          "dataset_ids": [item for item in dataset_ids if item]}
                runs[run_id] = record
                for output in outputs:
                    if isinstance(output, dict) and output.get("kind") == "model":
                        model_id = _identifier(output.get("id"))
                        if model_id:
                            models[model_id] = record
            except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError, RecursionError):
                continue
    return runs, models, evaluations


def read_matrix(path: Path, workspace: Path) -> dict[str, Any]:
    """Read and allowlist a matrix payload; never expose local paths or raw errors."""
    workspace = workspace.resolve()
    if path.is_symlink() or not path.resolve().is_relative_to(workspace):
        raise ValueError("matrix is unavailable")
    if path.stat().st_size > MAX_MATRIX_BYTES:
        raise ValueError("matrix exceeds the read limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("models"), list):
        raise ValueError("unsupported matrix format")

    metrics = [name for name in _string_list(value.get("expected_metrics"), MAX_METRICS)]
    models = []
    for item in value["models"][:MAX_MODELS]:
        if not isinstance(item, dict):
            continue
        raw_metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        raw_samples = item.get("samples") if isinstance(item.get("samples"), dict) else {}
        raw_delta = item.get("delta_vs_baseline") if isinstance(item.get("delta_vs_baseline"), dict) else {}
        samples = {}
        for key in metrics:
            sample = raw_samples.get(key)
            if not isinstance(sample, dict):
                # The matrix may use an alternate denominator name for internal holdout.
                sample = raw_samples.get("internal_holdout") if key == "internal_acc" else None
            sample = sample if isinstance(sample, dict) else {}
            samples[key] = {"requested": _number(sample.get("requested")), "observed": _number(sample.get("observed"))}
        raw_lineage = item.get("lineage") if isinstance(item.get("lineage"), dict) else {}
        status = _text(item.get("status"), 50) or "unknown"
        metric_values = {}
        for key in metrics:
            metric_value = _number(raw_metrics.get(key))
            sample = samples[key]
            metric_status = "unavailable" if metric_value is None else ("partial" if status != "completed" else "available")
            metric_values[key] = {
                "value": metric_value,
                "unit": _METRIC_UNITS.get(key, "unknown"),
                "status": metric_status,
                "requested": sample["requested"],
                "observed": sample["observed"],
                "delta": _number(raw_delta.get(key)),
            }
        raw_provenance = item.get("provenance") if isinstance(item.get("provenance"), dict) else {}
        raw_issues = raw_provenance.get("issues") if isinstance(raw_provenance.get("issues"), list) else []
        provenance_issues = []
        for issue in raw_issues[:20]:
            if isinstance(issue, dict):
                code = _identifier(issue.get("code"))
                message = _text(issue.get("message"), 300)
                if code and message:
                    provenance_issues.append({"code": code, "message": message})
        models.append({
            "model_id": _text(item.get("model_id"), 200),
            "status": status,
            "provenance": {"status": _text(raw_provenance.get("status"), 50) or "unknown", "issues": provenance_issues},
            "metrics": metric_values,
            "samples": samples,
            "lineage": {
                "eval_id": _identifier(raw_lineage.get("eval_id")),
                "evaluation_run_id": _identifier(item.get("evaluation_run_id") or raw_lineage.get("produced_by_run_id")),
                "model_artifact_id": _identifier(raw_lineage.get("model_id")),
                "dataset_ids": [_identifier(ref) for ref in _string_list(raw_lineage.get("dataset_ids")) if _identifier(ref)],
                "protocol_version": _text(raw_lineage.get("protocol_version"), 100),
                "config_sha256": _text(raw_lineage.get("config_sha256"), 128),
            },
            "errors_count": len(item.get("errors", [])) if isinstance(item.get("errors"), list) else 0,
        })

    comparison = value.get("comparability") if isinstance(value.get("comparability"), dict) else {}
    consistency = comparison.get("protocol_consistency") if isinstance(comparison.get("protocol_consistency"), dict) else {}
    requested_limits = value.get("requested_limits") if isinstance(value.get("requested_limits"), dict) else {}
    return {
        "protocol": _text(value.get("protocol"), 100),
        "baseline_model_id": _text(value.get("baseline_model_id"), 200),
        "expected_metrics": metrics,
        "requested_limits": {str(key)[:100]: _number(count) for key, count in list(requested_limits.items())[:MAX_METRICS]},
        "comparability": {
            "status": _text(comparison.get("status"), 50) or "unknown",
            "protocol_version_consistent": consistency.get("protocol_version_consistent") if isinstance(consistency.get("protocol_version_consistent"), bool) else None,
            "config_hash_consistent": consistency.get("config_hash_consistent") if isinstance(consistency.get("config_hash_consistent"), bool) else None,
            "dataset_ids_consistent": consistency.get("dataset_ids_consistent") if isinstance(consistency.get("dataset_ids_consistent"), bool) else None,
            "expected_protocol_version": _text(consistency.get("expected_protocol_version"), 100),
            "protocol_versions": _string_list(consistency.get("protocol_versions"), 100),
            "config_hashes": _string_list(consistency.get("config_hashes"), 100),
            "dataset_id_sets": [[ref for ref in _string_list(item, 100) if _identifier(ref)]
                                 for item in consistency.get("dataset_id_sets", [])[:100]
                                 if isinstance(item, list)] if isinstance(consistency.get("dataset_id_sets"), list) else [],
            "limitations": _string_list(comparison.get("limitations")),
        },
        "models": models,
        "truncated_models": len(value["models"]) > MAX_MODELS,
    }


def matrix_index(workspace: Path) -> dict[str, Any]:
    entries = []
    for matrix_id, path in discover_matrix_files(workspace):
        try:
            if path.stat().st_size > MAX_MATRIX_BYTES:
                entries.append({"matrix_id": matrix_id, "name": path.parent.parent.name, "status": "invalid", "issue": "矩阵文件超过读取上限。"})
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or not isinstance(value.get("models"), list):
                raise ValueError
            suffix = path.parent.name.removeprefix("evaluation-matrix")
            display_name = path.parent.parent.name if not suffix else f"{path.parent.parent.name} · 矩阵{suffix}"
            entries.append({"matrix_id": matrix_id, "name": display_name, "status": "available",
                            "protocol": _text(value.get("protocol"), 100),
                            "baseline_model_id": _text(value.get("baseline_model_id"), 200),
                            "model_count": min(len(value["models"]), MAX_MODELS),
                            "comparability_status": _text((value.get("comparability") or {}).get("status"), 50)})
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError, AttributeError, RecursionError):
            entries.append({"matrix_id": matrix_id, "name": path.parent.parent.name, "status": "invalid", "issue": "矩阵格式无法读取或不受支持。"})
    return {"matrices": entries, "read_only": True}


def matrix_detail(workspace: Path, matrix_id: str) -> dict[str, Any] | None:
    run_index, model_index, evaluation_index = _resolve_lineage(workspace)
    for candidate_id, path in discover_matrix_files(workspace):
        if candidate_id != matrix_id:
            continue
        try:
            result = read_matrix(path, workspace)
            for model in result["models"]:
                lineage = model["lineage"]
                training = model_index.get(lineage["model_artifact_id"] or "")
                evaluation = evaluation_index.get(lineage["eval_id"] or "")
                evaluation_run_id = lineage["evaluation_run_id"]
                evaluation_run = run_index.get(evaluation_run_id or "")
                if evaluation and evaluation.get("run_id") != evaluation_run_id:
                    evaluation = None
                lineage["training_resolution"] = ({**training, "resolved": True} if training else {"resolved": False})
                lineage["evaluation_resolution"] = ({
                    "resolved": bool(evaluation and evaluation_run),
                    "run_id": evaluation_run_id if evaluation_run else None,
                    "run_status": evaluation_run.get("status") if evaluation_run else None,
                    "eval_id": lineage["eval_id"] if evaluation else None,
                    "model_id": evaluation.get("model_id") if evaluation else None,
                    "dataset_ids": evaluation.get("dataset_ids", []) if evaluation else [],
                    "protocol_version": evaluation.get("protocol_version") if evaluation else None,
                    "config_sha256": evaluation.get("config_sha256") if evaluation else None,
                } if evaluation else {"resolved": False})
            return {"matrix_id": candidate_id, "name": path.parent.parent.name, **result}
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError, AttributeError, RecursionError):
            return None
    return None
