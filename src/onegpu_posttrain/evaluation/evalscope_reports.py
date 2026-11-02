from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _score(payload: dict[str, Any]) -> float | None:
    score = payload.get("score")
    if isinstance(score, (int, float)):
        return float(score)
    # Defensive fallback for report schema variations.
    for key in ("macro_score", "micro_score", "overall_score"):
        val = payload.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    # EvalScope schema v2 stores the aggregate score inside the primary metric.
    metrics = payload.get("metrics")
    if isinstance(metrics, list):
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            for key in ("score", "macro_score", "micro_score", "overall_score"):
                val = metric.get(key)
                if isinstance(val, (int, float)):
                    return float(val)
    return None


def _sample_count(payload: dict[str, Any]) -> int | None:
    for key in ("num_samples", "n_samples", "samples"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    perf = payload.get("perf_metrics")
    summary = perf.get("summary") if isinstance(perf, dict) else None
    if isinstance(summary, dict) and isinstance(summary.get("n_samples"), int):
        return summary["n_samples"]
    metrics = payload.get("metrics")
    if isinstance(metrics, list):
        for metric in metrics:
            if isinstance(metric, dict) and isinstance(metric.get("num"), int):
                return metric["num"]
    return None


def collect_evalscope_reports(work_dir: str | Path) -> dict[str, Any]:
    work_dir = Path(work_dir)
    root = work_dir / "reports"
    out: dict[str, Any] = {}
    if not work_dir.exists():
        return out
    paths = set(root.rglob("*.json")) if root.is_dir() else set()
    for report_dir in work_dir.rglob("reports"):
        if report_dir.is_dir() and not report_dir.is_symlink():
            paths.update(report_dir.rglob("*.json"))
    for path in sorted(paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        dataset = str(payload.get("dataset_name") or payload.get("name") or path.stem)
        out[dataset] = {
            "score": _score(payload),
            "samples": _sample_count(payload),
            "model_name": payload.get("model_name"),
            "report_path": str(path),
        }
    return out
