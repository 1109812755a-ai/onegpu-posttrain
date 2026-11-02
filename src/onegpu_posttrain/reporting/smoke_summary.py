from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .stage_compare import _metrics


def _json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _row(stage: str, run_dir: Path, *, preference: bool = False, grpo: bool = False) -> dict[str, Any]:
    eval_data = _json(run_dir / "evaluation" / "eval.json")
    run = _json(run_dir / "run.json") or {}
    task = _metrics(eval_data) if eval_data else {}
    row: dict[str, Any] = {
        "stage": stage,
        "status": ((run.get("train") or run.get("execution") or {}).get("status") if run else None) or ("EVAL_ONLY" if eval_data else "NOT_RUN"),
        **task,
    }
    if preference:
        pref = _json(run_dir / "preference_evaluation" / "preference_metrics.json") or {}
        row["policy_pair_accuracy"] = pref.get("policy_pair_accuracy")
        row["policy_pair_margin"] = pref.get("policy_pair_margin")
    if grpo:
        gm = _json(run_dir / "grpo_metrics.json") or run.get("grpo_metrics") or {}
        latest = gm.get("latest") or {}
        row["reward"] = latest.get("reward")
        row["kl"] = latest.get("kl")
    return row


def write_smoke_summary(run_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    root = Path(run_root)
    specs = [
        ("base", root / "base-smoke-s42", False, False),
        ("peft-cpt", root / "cpt-peft-smoke-s42", False, False),
        ("sft", root / "sft-smoke-s42", True, False),
        ("dpo", root / "dpo-smoke-s42", True, False),
        ("simpo-original", root / "simpo-original-smoke-s42", True, False),
        ("grpo-standard", root / "grpo-standard-smoke-s42", False, True),
        ("grpo-lowmem", root / "grpo-lowmem-smoke-s42", False, True),
    ]
    rows = [_row(name, d, preference=pref, grpo=grpo) for name, d, pref, grpo in specs]
    result = {"protocol": "smoke-pipeline-summary-v1", "run_root": str(root), "stages": rows}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def fmt(v: Any) -> str:
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    lines = [
        "# OneGPU-PostTrain v0.1 Smoke Pipeline Summary",
        "",
        "| Stage | Status | GSM8K | MATH-500 | Internal | Domain PPL | Pair Acc | Reward | KL |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['stage']} | {fmt(r.get('status'))} | {fmt(r.get('gsm8k'))} | {fmt(r.get('math_500'))} | "
            f"{fmt(r.get('internal_acc'))} | {fmt(r.get('domain_ppl'))} | {fmt(r.get('policy_pair_accuracy'))} | "
            f"{fmt(r.get('reward'))} | {fmt(r.get('kl'))} |"
        )
    lines += ["", "Smoke validates the end-to-end experiment pipeline. Do not interpret short smoke runs as final capability rankings.", ""]
    (out / "smoke_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return result
