"""Evidence-bounded effect summaries for targeted stage evaluation runs."""

from __future__ import annotations

from typing import Any

from ..data.targeted import MetricTarget
from ..data.targeted_stage_plan import TargetedStagePlan
from ..lineage import EvaluationLineage, RunRecord, sha256_json


def _measured_metric(evaluation: EvaluationLineage, target: MetricTarget) -> dict[str, Any] | None:
    candidates = (target.name, f"evalscope.{target.name}")
    for name in candidates:
        value = evaluation.metrics.get(name)
        if not isinstance(value, dict):
            continue
        score, denominator = value.get("value"), value.get("denominator")
        if (isinstance(score, (int, float)) and not isinstance(score, bool)
                and isinstance(denominator, int) and not isinstance(denominator, bool)
                and denominator >= target.minimum_samples):
            return {"metric": name, "value": float(score), "denominator": denominator,
                    "status": "measured"}
    return None


def build_stage_effect_report(
    *, plan: TargetedStagePlan, training_run: RunRecord, model_id: str,
    after: EvaluationLineage, before: EvaluationLineage | None,
) -> dict[str, Any]:
    """Build a GRPO effect report without inventing missing baselines or metrics."""
    if plan.target.stage != "grpo":
        raise ValueError("stage effect report currently supports GRPO plans only")
    if training_run.run_type != "train" or training_run.status != "completed":
        raise ValueError("effect report requires the actual GRPO training Run")
    if (training_run.targeted_plan_id, training_run.targeted_plan_version) != (plan.plan_id, plan.version):
        raise ValueError("training Run does not match the targeted plan identity")
    if after.status != "completed":
        raise ValueError("effect report requires a completed post-training evaluation")
    if after.model_id != model_id or after.model_produced_by_run_id != training_run.run_id:
        raise ValueError("post-training evaluation is not linked to the supplied model and training Run")
    if (after.targeted_plan_id, after.targeted_plan_version) != (plan.plan_id, plan.version):
        raise ValueError("post-training evaluation does not match the targeted plan identity")

    training_datasets = [
        {"dataset_id": item.id,
         "content_sha256": item.digest.value if item.digest and item.digest.status == "available" else None}
        for item in training_run.inputs if item.kind == "dataset" and item.role == "train"
    ]
    report: dict[str, Any] = {
        "schema_version": "1.0", "status": "not_comparable",
        "targeted_plan_id": plan.plan_id, "targeted_plan_version": plan.version,
        "training_run_id": training_run.run_id, "model_id": model_id,
        "training_datasets": training_datasets,
        "before_eval_id": before.eval_id if before else None,
        "after_eval_id": after.eval_id,
        "target_metric": None, "guardrail_metrics": {}, "comparison_limitations": [],
    }
    contract = plan.diagnostic_evaluation
    after_matches_contract = (contract.dataset_id in after.dataset_ids
                              and after.protocol_version == contract.protocol_version
                              and after.config_sha256 == contract.config_sha256)
    before_matches_contract = bool(before and before.status == "completed"
                                   and before.model_id == plan.starting_model.model_id
                                   and contract.dataset_id in before.dataset_ids
                                   and before.protocol_version == contract.protocol_version
                                   and before.config_sha256 == contract.config_sha256)
    same_protocol = bool(before_matches_contract and after_matches_contract
                         and before.protocol_version == after.protocol_version
                         and before.config_sha256 == after.config_sha256
                         and before.dataset_ids == after.dataset_ids)
    if not after_matches_contract:
        report["comparison_limitations"].append("训练后评测未匹配计划指定的诊断集、协议或配置摘要")
    if not before_matches_contract:
        report["comparison_limitations"].append("未找到与计划诊断集、协议和配置摘要完全匹配的起点模型评测")
    elif after_matches_contract and not same_protocol:
        report["comparison_limitations"].append("前后评测的数据集、协议或配置摘要不同")

    target = plan.target.target_metric
    if target is None:
        report["target_state"] = "not_configured"
        report["comparison_limitations"].append("计划未配置目标指标和阈值；仅报告原始测量，不判定达标")
    elif same_protocol:
        before_metric = _measured_metric(before, target)
        after_metric = _measured_metric(after, target)
        if before_metric and after_metric:
            delta = after_metric["value"] - before_metric["value"]
            passed = (after_metric["value"] >= target.threshold if target.direction == "higher"
                      else after_metric["value"] <= target.threshold)
            report["target_metric"] = {
                "name": target.name, "direction": target.direction,
                "threshold": target.threshold, "threshold_source": target.threshold_source,
                "minimum_samples": target.minimum_samples,
                "before": before_metric, "after": after_metric, "delta": delta,
                "threshold_status": "pass" if passed else "fail",
            }
            report["target_state"] = report["target_metric"]["threshold_status"]
        else:
            report["target_state"] = "not_measured"
            report["comparison_limitations"].append("目标指标缺失或实际评测分母低于计划最小样本数")

        failures = []
        for guardrail in plan.target.guardrail_metrics:
            before_value = _measured_metric(before, guardrail)
            after_value = _measured_metric(after, guardrail)
            result = {"name": guardrail.name, "direction": guardrail.direction,
                      "threshold": guardrail.threshold, "threshold_source": guardrail.threshold_source,
                      "minimum_samples": guardrail.minimum_samples,
                      "before": before_value, "after": after_value}
            if before_value and after_value:
                passed = (after_value["value"] >= guardrail.threshold if guardrail.direction == "higher"
                          else after_value["value"] <= guardrail.threshold)
                result["threshold_status"] = "pass" if passed else "fail"
                if not passed:
                    failures.append(guardrail.name)
            else:
                result["threshold_status"] = "not_measured"
            report["guardrail_metrics"][guardrail.name] = result
        if report.get("target_state") in {"pass", "fail"}:
            report["status"] = "guardrail_regression" if failures else (
                "target_pass" if report["target_state"] == "pass" else "target_fail")
    else:
        report["target_state"] = "not_comparable" if target else report["target_state"]

    report["evidence"] = {
        "before_eval_id": before.eval_id if before else None,
        "before_model_id": before.model_id if before else None,
        "after_eval_id": after.eval_id, "after_model_id": after.model_id,
        "protocol_version": after.protocol_version,
        "evaluation_dataset_ids": after.dataset_ids,
        "evaluation_config_sha256": after.config_sha256,
    }
    report["report_id"] = f"targeted-stage-effect-{sha256_json(report)[:20]}"
    return report
