from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


def _checkpoint_step(path: Path) -> int:
    m = re.search(r"checkpoint-(\d+)$", path.parent.name)
    if m:
        return int(m.group(1))
    m = re.search(r"checkpoint-(\d+)$", path.name)
    return int(m.group(1)) if m else -1


def discover_trainer_state(train_dir: str | Path) -> Path:
    root = Path(train_dir)
    candidates = list(root.rglob("trainer_state.json"))
    if not candidates:
        raise FileNotFoundError(f"No trainer_state.json found under {root}")
    return max(candidates, key=lambda p: (_checkpoint_step(p), p.stat().st_mtime_ns))


def _numeric(history: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in history:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _latest(history: list[dict[str, Any]], key: str) -> float | None:
    values = _numeric(history, key)
    return values[-1] if values else None


def _avg(history: list[dict[str, Any]], key: str) -> float | None:
    values = _numeric(history, key)
    return mean(values) if values else None


def _component_keys(history: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for row in history:
        for key in row:
            if re.match(r"^(?:reward|rewards)/.+/(?:mean|std)$", key):
                keys.add(key)
    return keys


def collect_grpo_training_metrics(
    train_dir: str | Path,
    *,
    beta: float,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Extract stable GRPO metrics from Hugging Face/ms-swift trainer state.

    The exact log history may contain additional implementation-specific fields;
    the project only promotes a small, documented set into the public artifact.
    """
    state_path = discover_trainer_state(train_dir)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    history = [x for x in (state.get("log_history") or []) if isinstance(x, dict)]

    metric_keys = [
        "reward",
        "reward_std",
        "frac_reward_zero_std",
        "kl",
        "completions/mean_length",
        "completions/min_length",
        "completions/max_length",
        "completions/clipped_ratio",
        "loss",
        "grad_norm",
        "learning_rate",
    ]
    latest = {key: _latest(history, key) for key in metric_keys}
    averages = {key: _avg(history, key) for key in metric_keys}

    components: dict[str, dict[str, float | None]] = {}
    for key in sorted(_component_keys(history)):
        # ms-swift has used both `reward/<name>/...` and
        # `rewards/<name>/...` spellings across versions/code paths.
        m = re.match(r"^(?:reward|rewards)/(.+)/(mean|std)$", key)
        if not m:
            continue
        name, stat = m.groups()
        components.setdefault(name, {})[f"latest_{stat}"] = _latest(history, key)
        components.setdefault(name, {})[f"mean_{stat}"] = _avg(history, key)

    steps = [float(x["step"]) for x in history if isinstance(x.get("step"), (int, float))]
    result: dict[str, Any] = {
        "protocol": "grpo-training-metrics-v1",
        "status": "measured" if latest.get("reward") is not None else "not_measured",
        "trainer_state": str(state_path),
        "log_points": len(history),
        "optimizer_step": int(max(steps)) if steps else None,
        "beta": float(beta),
        "kl_applicable": bool(float(beta) != 0.0),
        "latest": latest,
        "mean": averages,
        "reward_components": components,
    }
    result["metric_statuses"] = {
        key: {
            "status": ("not_applicable" if key == "kl" and beta == 0 else
                       "measured" if latest.get(key) is not None else "not_measured"),
            "denominator": len(history),
            "reason": ("reference-free low-memory profile" if key == "kl" and beta == 0 else
                       None if latest.get(key) is not None else "trainer_state 没有该 rollout/training metric"),
        }
        for key in metric_keys
    }
    for name, values in components.items():
        result["metric_statuses"][f"reward_components.{name}"] = {
            "status": "measured" if values.get("latest_mean") is not None else "not_measured",
            "denominator": len(history),
            "reason": None if values.get("latest_mean") is not None else "trainer_state 没有该 reward component",
        }
    if beta == 0:
        # Explicit N/A semantics for the no-reference low-memory profile.
        result["latest"]["kl"] = None
        result["mean"]["kl"] = None

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
