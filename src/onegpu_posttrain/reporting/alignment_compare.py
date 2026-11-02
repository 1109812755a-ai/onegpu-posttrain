from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..lineage import sha256_json
from .stage_compare import _metrics


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pref(path: str | Path) -> dict[str, Any]:
    data = _load(path)
    statuses = data.get("metric_statuses") if isinstance(data.get("metric_statuses"), dict) else {}
    result = {}
    for name in ("policy_pair_accuracy", "policy_pair_margin", "chosen_mean_logprob", "rejected_mean_logprob"):
        value = data.get(name)
        evidence = statuses.get(f"preference.{name.removeprefix('policy_pair_')}", {})
        status = evidence.get("status") if isinstance(evidence, dict) else None
        if status is None:
            status = "measured" if isinstance(value, (int, float)) else "not_measured"
        result[name] = value if isinstance(value, (int, float)) else None
        result[f"{name}_evidence"] = {
            "status": status,
            "denominator": evidence.get("denominator") if isinstance(evidence, dict) else data.get("pairs"),
            "reason": evidence.get("reason") if isinstance(evidence, dict) else (
                None if isinstance(value, (int, float)) else "metric missing or evaluator failed"
            ),
        }
    return result


def _run_context(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    lineage_path = root if root.name == "lineage.json" else root / "lineage.json"
    config_path = lineage_path.with_name("resolved_config.json")
    lineage = _load(lineage_path)
    resolved = _load(config_path)
    expected = lineage.get("resolved_config_sha256")
    actual = sha256_json(resolved)
    if not expected or expected != actual:
        raise ValueError(f"resolved config hash mismatch: {lineage_path}")
    inputs = lineage.get("inputs") if isinstance(lineage.get("inputs"), list) else []
    parent = next((item for item in inputs if isinstance(item, dict) and item.get("kind") == "model"), None)
    dataset = next((item for item in inputs if isinstance(item, dict) and item.get("kind") == "dataset"), None)
    training = resolved.get("training") if isinstance(resolved.get("training"), dict) else {}
    shared_keys = (
        "seed", "max_steps", "max_length", "max_seq_length", "per_device_train_batch_size",
        "gradient_accumulation_steps", "num_train_epochs", "learning_rate",
    )
    controls = {key: training[key] for key in shared_keys if key in training}
    return {
        "run_id": lineage.get("run_id"), "status": lineage.get("status"),
        "algorithm": resolved.get("algorithm"), "config_sha256": expected,
        "parent": parent, "dataset": dataset, "controls": controls,
    }


def _paired_comparability(dpo_run: str | Path | None, simpo_run: str | Path | None) -> dict[str, Any]:
    if dpo_run is None or simpo_run is None:
        return {"status": "observational_only", "matched_ab": False,
                "reasons": ["缺少 DPO/SimPO 两侧 training lineage 与 resolved config；不能证明受控匹配"]}
    try:
        dpo, simpo = _run_context(dpo_run), _run_context(simpo_run)
    except (OSError, ValueError, TypeError) as exc:
        return {"status": "observational_only", "matched_ab": False,
                "reasons": [f"训练配置/血缘证据无效：{str(exc)[:240]}"]}
    reasons = []
    if dpo["run_id"] == simpo["run_id"]:
        reasons.append("DPO 与 SimPO 必须是两个独立 run")
    if dpo["status"] != "completed" or simpo["status"] != "completed":
        reasons.append("两侧训练都必须完成")
    for key in ("parent", "dataset"):
        left, right = dpo[key], simpo[key]
        if not left or not right or (left.get("id"), left.get("digest")) != (right.get("id"), right.get("digest")):
            reasons.append(f"DPO 与 SimPO 的 {key} 身份或摘要不同/缺失")
    common_controls = set(dpo["controls"]) & set(simpo["controls"])
    if not common_controls or any(dpo["controls"][key] != simpo["controls"][key] for key in common_controls):
        reasons.append("训练 seed/预算/超参数不一致或缺少共同控制字段")
    if dpo["algorithm"] != "dpo" or simpo["algorithm"] != "simpo":
        reasons.append("两侧算法身份与 DPO/SimPO 不匹配")
    return {
        "status": "matched_ab" if not reasons else "observational_only",
        "matched_ab": not reasons,
        "reasons": reasons,
        "runs": {"dpo": dpo, "simpo": simpo},
    }


def write_alignment_comparison(
    *,
    sft_eval: str | Path,
    dpo_eval: str | Path,
    simpo_eval: str | Path,
    sft_preference: str | Path,
    dpo_preference: str | Path,
    simpo_preference: str | Path,
    output_dir: str | Path,
    dpo_run: str | Path | None = None,
    simpo_run: str | Path | None = None,
) -> dict[str, Any]:
    evals = {key: _load(value) for key, value in {
        "sft": sft_eval, "dpo": dpo_eval, "simpo": simpo_eval,
    }.items()}
    preference = {key: _pref(value) for key, value in {
        "sft": sft_preference, "dpo": dpo_preference, "simpo": simpo_preference,
    }.items()}
    stages = {
        key: {"task": _metrics(payload), "preference": preference[key]}
        for key, payload in evals.items()
    }
    evaluation_lineages = {}
    for key, path in {"sft": sft_eval, "dpo": dpo_eval, "simpo": simpo_eval}.items():
        sidecar = Path(path).with_name("evaluation_lineage.json")
        evaluation_lineages[key] = _load(sidecar) if sidecar.is_file() else None
    matched = _paired_comparability(dpo_run, simpo_run)
    task_comparable = True
    task_reasons = []
    dpo_eval_lineage, simpo_eval_lineage = evaluation_lineages["dpo"], evaluation_lineages["simpo"]
    if not dpo_eval_lineage or not simpo_eval_lineage:
        task_comparable = False
        task_reasons.append("缺少 DPO/SimPO evaluation lineage")
    else:
        for key in ("protocol_version", "dataset_ids", "config_sha256"):
            if dpo_eval_lineage.get(key) != simpo_eval_lineage.get(key):
                task_comparable = False
                task_reasons.append(f"DPO 与 SimPO 评测 {key} 不一致")
    result = {
        "protocol": "alignment-comparison-v1",
        "stages": stages,
        "comparability": {
            "dpo_vs_simpo": matched,
            "task_evaluation_dpo_vs_simpo": {
                "status": "comparable" if task_comparable else "observational_only",
                "reasons": task_reasons,
            },
        },
        "guardrails": {"status": "not_configured", "reason": "本比较未配置任务回退阈值；仅展示实测指标，不自动判定通过/失败"},
        "training_cost": {"status": "not_measured", "reason": "本次输入未提供可核验的运行时长/峰值显存成本证据"},
        "notes": [
            "只有传入并校验两侧训练 lineage/config 后才可能标记 matched A/B；否则仅作观察性比较。",
            "policy_pair_* metrics are computed by one common held-out evaluator, not trainer-specific reward scales.",
            "Smoke results validate the pipeline; they are not algorithm leaderboard claims.",
            "缺失指标用 not_measured 表示，绝不以 0 替代。",
        ],
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "alignment_comparison.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    def fmt(v: Any) -> str:
        return "N/A" if v is None else f"{float(v):.6g}" if isinstance(v, (int, float)) else str(v)

    metrics = [
        ("gsm8k", lambda x: x["task"].get("gsm8k")),
        ("math_500", lambda x: x["task"].get("math_500")),
        ("internal_acc", lambda x: x["task"].get("internal_acc")),
        ("policy_pair_accuracy", lambda x: x["preference"].get("policy_pair_accuracy")),
        ("policy_pair_margin", lambda x: x["preference"].get("policy_pair_margin")),
    ]
    rows = [
        f"| `{name}` | {fmt(getter(stages['sft']))} | {fmt(getter(stages['dpo']))} | {fmt(getter(stages['simpo']))} |"
        for name, getter in metrics
    ]
    md = "\n".join([
        "# Alignment Smoke Comparison",
        "",
        "> Same M2 parent and training preference data. Public preference metrics use the same held-out evaluator for all stages.",
        "",
        f"> DPO vs SimPO: `{matched['status']}`. Evaluation match: `{result['comparability']['task_evaluation_dpo_vs_simpo']['status']}`. Missing values are not measured, not zero.",
        "",
        "| Metric | SFT (M2) | DPO (M3) | SimPO-original (M3) |",
        "|---|---:|---:|---:|",
        *rows,
        "",
        "Smoke comparisons validate the controlled experiment pipeline; do not treat these small-step results as general algorithm rankings.",
        "",
    ])
    (out / "alignment_comparison.md").write_text(md, encoding="utf-8")
    return result
