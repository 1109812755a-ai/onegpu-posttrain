"""Evidence-aware CPT before/after effect and resource report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _metrics(payload: dict[str, Any]) -> dict[str, float | None]:
    domain = payload.get("domain_perplexity") or {}
    internal = payload.get("internal_math") or {}
    reports = payload.get("evalscope_reports") or {}
    gsm = next((v.get("score") for k, v in reports.items() if "gsm8k" in str(k).lower() and isinstance(v, dict)), None)
    math = next((v.get("score") for k, v in reports.items() if "math_500" in str(k).lower() and isinstance(v, dict)), None)
    def number(value):
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    return {"domain_ppl": number(domain.get("perplexity")), "gsm8k": number(gsm),
            "math_500": number(math), "internal_accuracy": number(internal.get("accuracy"))}


def build_cpt_effect_report(training_run_dir: str | Path, base_eval_path: str | Path,
                            cpt_eval_path: str | Path) -> dict[str, Any]:
    run_dir = Path(training_run_dir)
    run = _read(run_dir / "run.json")
    training_lineage = _read(run_dir / "lineage.json")
    base = _read(base_eval_path)
    candidate = _read(cpt_eval_path)
    base_lineage_path = Path(base_eval_path).with_name("evaluation_lineage.json")
    candidate_lineage_path = Path(cpt_eval_path).with_name("evaluation_lineage.json")
    base_lineage = _read(base_lineage_path) if base_lineage_path.is_file() else None
    candidate_lineage = _read(candidate_lineage_path) if candidate_lineage_path.is_file() else None

    warnings: list[str] = []
    same_protocol = bool(base_lineage and candidate_lineage
                         and base_lineage.get("protocol_version") == candidate_lineage.get("protocol_version"))
    same_holdout = bool(base_lineage and candidate_lineage
                        and set(base_lineage.get("dataset_ids", [])) == set(candidate_lineage.get("dataset_ids", [])))
    if not base_lineage or not candidate_lineage:
        warnings.append("评测 lineage 缺失；不能确认评测协议和数据是否一致。")
    elif not same_protocol:
        warnings.append("评测协议不一致；指标只能并列展示，不能计算可比 delta。")
    if base_lineage and candidate_lineage and not same_holdout:
        warnings.append("评测数据集集合不一致；delta 不具可比性。")
    expected_run_ids = {training_lineage.get("run_id")}
    merge = run.get("merge") if isinstance(run.get("merge"), dict) else {}
    if merge.get("run_id"):
        expected_run_ids.add(merge["run_id"])
    candidate_model_traces_to_run = bool(
        candidate_lineage and candidate_lineage.get("model_produced_by_run_id") in expected_run_ids
    )
    if candidate_lineage and not candidate_model_traces_to_run:
        warnings.append("CPT 后评测模型未追溯到本次训练/合并运行。")
    base_model_ids = {item.get("id") for item in training_lineage.get("inputs", [])
                      if item.get("kind") == "model" and item.get("role") == "parent"}
    base_model_matches = bool(base_lineage and base_lineage.get("model_id") in base_model_ids)
    if base_lineage and not base_model_matches:
        warnings.append("Base 前测模型身份与 CPT run 的 parent model 不匹配。")
    train_status = (run.get("train") or {}).get("status") if isinstance(run.get("train"), dict) else None
    if train_status != "PASS":
        warnings.append("CPT 训练没有 PASS 状态；不把该候选解释为训练后模型效果。")
    evals_completed = bool(base_lineage and candidate_lineage
                           and base_lineage.get("status") == "completed"
                           and candidate_lineage.get("status") == "completed")
    if base_lineage and candidate_lineage:
        if not evals_completed:
            warnings.append("至少一侧评测未完整完成；缺失项保持 not_measured。")
    comparable = (same_protocol and same_holdout and evals_completed and train_status == "PASS"
                  and candidate_model_traces_to_run and base_model_matches)
    bm, cm = _metrics(base), _metrics(candidate)
    effects = {}
    for key in bm:
        delta = cm[key] - bm[key] if comparable and bm[key] is not None and cm[key] is not None else None
        effects[key] = {"before": bm[key], "after": cm[key], "delta": delta,
                        "status": "measured" if delta is not None else "not_measured",
                        "direction": "lower_is_better" if key == "domain_ppl" else "higher_is_better"}
    train = run.get("train") if isinstance(run.get("train"), dict) else {}
    return {
        "schema_version": "1.0", "report_type": "cpt_before_after_effect",
        "training_run_id": training_lineage.get("run_id"),
        "dataset_id": next((item.get("id") for item in training_lineage.get("inputs", [])
                            if item.get("kind") == "dataset"), None),
        "base_eval_id": base_lineage.get("eval_id") if base_lineage else base.get("eval_id"),
        "cpt_eval_id": candidate_lineage.get("eval_id") if candidate_lineage else candidate.get("eval_id"),
        "holdout_dataset_ids": candidate_lineage.get("dataset_ids") if candidate_lineage else None,
        "comparison_status": "matched_observation" if comparable and not warnings else "observational_or_incomplete",
        "causal_claim": False,
        "metrics": effects,
        "resources": {
            "training_wall_seconds": train.get("wall_seconds"),
            "peak_gpu_memory_mib_sampled": train.get("peak_gpu_memory_mib_sampled"),
            "merge_wall_seconds": (merge.get("execution") or {}).get("wall_seconds") if merge else None,
        },
        "warnings": warnings,
    }


def write_cpt_effect_report(training_run_dir: str | Path, base_eval_path: str | Path,
                            cpt_eval_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    report = build_cpt_effect_report(training_run_dir, base_eval_path, cpt_eval_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report
