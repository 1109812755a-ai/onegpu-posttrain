from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .stage_compare import _metrics


def _load(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _run_summary(run: dict[str, Any] | None) -> dict[str, Any]:
    if not run:
        return {"status": "NOT_RUN"}
    train = run.get("train") or {}
    metrics = run.get("grpo_metrics") or {}
    latest = metrics.get("latest") or {}
    comps = metrics.get("reward_components") or {}

    def comp_latest(name: str) -> Any:
        data = comps.get(name) or {}
        return data.get("latest_mean")

    return {
        "status": train.get("status", "UNKNOWN"),
        "wall_seconds": train.get("wall_seconds"),
        "peak_gpu_memory_mib_sampled": train.get("peak_gpu_memory_mib_sampled"),
        "reward": latest.get("reward"),
        "reward_std": latest.get("reward_std"),
        "frac_reward_zero_std": latest.get("frac_reward_zero_std"),
        "kl": latest.get("kl"),
        "completion_mean_length": latest.get("completions/mean_length"),
        "completion_clipped_ratio": latest.get("completions/clipped_ratio"),
        "accuracy_reward": comp_latest("onegpu_accuracy"),
        "format_reward": comp_latest("onegpu_boxed_format"),
    }


def write_grpo_comparison(
    *,
    sft_eval: str | Path,
    standard_run: str | Path,
    standard_eval: str | Path | None,
    lowmem_run: str | Path,
    lowmem_eval: str | Path | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    sft = _load(sft_eval) or {}
    std_run = _load(standard_run)
    low_run = _load(lowmem_run)
    std_eval = _load(standard_eval)
    low_eval = _load(lowmem_eval)

    result = {
        "protocol": "grpo-profile-comparison-v1",
        "sft": {"task": _metrics(sft)},
        "standard": {
            "profile": "beta=0.04, reference enabled",
            "training": _run_summary(std_run),
            "task": _metrics(std_eval) if std_eval else None,
        },
        "lowmem": {
            "profile": "beta=0, reference disabled",
            "training": _run_summary(low_run),
            "task": _metrics(low_eval) if low_eval else None,
        },
        "notes": [
            "Both GRPO profiles start from the same merged M2-SFT parent and canonical GRPO data.",
            "KL is N/A by design for beta=0 low-memory GRPO.",
            "Smoke results validate feasibility and instrumentation, not general algorithm superiority.",
        ],
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "grpo_comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def fmt(v: Any) -> str:
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    sftm = result["sft"]["task"]
    stdt = result["standard"]["training"]
    lowt = result["lowmem"]["training"]
    stdm = result["standard"]["task"] or {}
    lowm = result["lowmem"]["task"] or {}
    rows = [
        ("status", "BASELINE", stdt.get("status"), lowt.get("status")),
        ("gsm8k", sftm.get("gsm8k"), stdm.get("gsm8k"), lowm.get("gsm8k")),
        ("math_500", sftm.get("math_500"), stdm.get("math_500"), lowm.get("math_500")),
        ("internal_acc", sftm.get("internal_acc"), stdm.get("internal_acc"), lowm.get("internal_acc")),
        ("reward", None, stdt.get("reward"), lowt.get("reward")),
        ("accuracy_reward", None, stdt.get("accuracy_reward"), lowt.get("accuracy_reward")),
        ("format_reward", None, stdt.get("format_reward"), lowt.get("format_reward")),
        ("reward_std", None, stdt.get("reward_std"), lowt.get("reward_std")),
        ("frac_reward_zero_std", None, stdt.get("frac_reward_zero_std"), lowt.get("frac_reward_zero_std")),
        ("kl", None, stdt.get("kl"), None),
        ("completion_mean_length", None, stdt.get("completion_mean_length"), lowt.get("completion_mean_length")),
        ("peak_vram_mib_sampled", None, stdt.get("peak_gpu_memory_mib_sampled"), lowt.get("peak_gpu_memory_mib_sampled")),
        ("wall_seconds", None, stdt.get("wall_seconds"), lowt.get("wall_seconds")),
    ]
    md_rows = [f"| `{n}` | {fmt(a)} | {fmt(b)} | {fmt(c)} |" for n, a, b, c in rows]
    md = "\n".join([
        "# GRPO Smoke Profile Comparison",
        "",
        "> Standard uses beta=0.04 with a reference model; lowmem uses beta=0 and no reference model.",
        "",
        "| Metric | M2-SFT | GRPO-standard | GRPO-lowmem |",
        "|---|---:|---:|---:|",
        *md_rows,
        "",
        "`kl` is intentionally N/A for lowmem. A standard-profile OOM is a valid 8GB feasibility result.",
        "",
    ])
    (out / "grpo_comparison.md").write_text(md, encoding="utf-8")
    return result
