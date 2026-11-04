"""Deterministic evaluation comparability checks (no LLM judgment involved)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .contracts import AssistantContract, EvidenceRef

ComparisonStatus = Literal["comparable", "incompatible", "unknown", "partial", "failed"]


class MetricDelta(AssistantContract):
    metric_id: str
    left_value: float
    right_value: float
    delta: float
    denominator: int


class EvaluationComparison(AssistantContract):
    status: ComparisonStatus
    left_evidence_id: str
    right_evidence_id: str
    left_model_id: str | None = None
    right_model_id: str | None = None
    same_model: bool | None = None
    protocol_version: str | None = None
    dataset_ids: list[str] = Field(default_factory=list)
    metrics: list[MetricDelta] = Field(default_factory=list, max_length=30)
    reasons: list[str] = Field(default_factory=list, max_length=20)


def _facts(evidence: EvidenceRef) -> dict[str, str | int | float | bool | None]:
    return {fact.key: fact.value for fact in evidence.facts}


def compare_evaluations(left: EvidenceRef, right: EvidenceRef) -> EvaluationComparison:
    """Compare only when status, model, protocol, dataset and metric coverage are known.

    Different model identities are expected in a model comparison and are
    reported. Protocol/dataset identity and per-metric comparability identities
    must agree before a numeric delta is produced.
    """
    a, b = _facts(left), _facts(right)
    left_model = a.get("model_id") if isinstance(a.get("model_id"), str) else None
    right_model = b.get("model_id") if isinstance(b.get("model_id"), str) else None
    left_protocol = a.get("protocol_version") if isinstance(a.get("protocol_version"), str) else None
    right_protocol = b.get("protocol_version") if isinstance(b.get("protocol_version"), str) else None
    left_datasets = sorted(str(a.get("dataset_ids", "")).split(", ")) if a.get("dataset_ids") else []
    right_datasets = sorted(str(b.get("dataset_ids", "")).split(", ")) if b.get("dataset_ids") else []
    base = dict(
        left_evidence_id=left.evidence_id,
        right_evidence_id=right.evidence_id,
        left_model_id=left_model,
        right_model_id=right_model,
        same_model=(left_model == right_model) if left_model and right_model else None,
        protocol_version=left_protocol if left_protocol == right_protocol else None,
        dataset_ids=left_datasets if left_datasets == right_datasets else [],
    )
    identity_notes: list[str] = []
    if left_protocol and right_protocol and left_protocol != right_protocol:
        identity_notes.append("评测协议不一致。")
    if left_datasets and right_datasets and left_datasets != right_datasets:
        identity_notes.append("评测数据集身份不一致。")
    if left.status == "failed" or right.status == "failed":
        return EvaluationComparison(status="failed", **base,
                                    reasons=["至少一项评测失败；不生成分数差值。", *identity_notes])
    if left.status == "partial" or right.status == "partial":
        return EvaluationComparison(status="partial", **base,
                                    reasons=["至少一项评测部分完成；不将未完成部分视为零分。", *identity_notes])
    if left.status != "available" or right.status != "available":
        return EvaluationComparison(status="unknown", **base, reasons=["评测证据缺失、过旧或状态未知。"])
    if not left_model or not right_model or not left_protocol or not right_protocol or not left_datasets or not right_datasets:
        return EvaluationComparison(status="unknown", **base, reasons=["模型、协议或评测数据集身份不完整。"])
    if left_protocol != right_protocol:
        return EvaluationComparison(status="incompatible", **base, reasons=["评测协议不一致。"])
    if left_datasets != right_datasets:
        return EvaluationComparison(status="incompatible", **base, reasons=["评测数据集身份不一致。"])

    left_metrics = {key.removeprefix("metric."): value for key, value in a.items()
                    if key.startswith("metric.") and isinstance(value, (int, float)) and not isinstance(value, bool)}
    right_metrics = {key.removeprefix("metric."): value for key, value in b.items()
                     if key.startswith("metric.") and isinstance(value, (int, float)) and not isinstance(value, bool)}
    common = sorted(set(left_metrics) & set(right_metrics))
    deltas: list[MetricDelta] = []
    blocked: list[str] = []
    for metric_id in common:
        protocol_a = a.get(f"protocol.{metric_id}", left_protocol)
        protocol_b = b.get(f"protocol.{metric_id}", right_protocol)
        key_a = a.get(f"comparability.{metric_id}")
        key_b = b.get(f"comparability.{metric_id}")
        coverage_a = a.get(f"coverage.{metric_id}")
        coverage_b = b.get(f"coverage.{metric_id}")
        denominator_a = next((fact.denominator for fact in left.facts if fact.key == f"metric.{metric_id}"), None)
        denominator_b = next((fact.denominator for fact in right.facts if fact.key == f"metric.{metric_id}"), None)
        if protocol_a != protocol_b or not key_a or key_a != key_b:
            blocked.append(f"{metric_id} 的协议或可比性身份不一致。")
            continue
        if (not isinstance(denominator_a, int) or not isinstance(denominator_b, int)
                or denominator_a != denominator_b
                or (coverage_a is not None and coverage_b is not None and coverage_a != coverage_b)):
            blocked.append(f"{metric_id} 的分母或覆盖范围不一致。")
            continue
        deltas.append(MetricDelta(
            metric_id=metric_id,
            left_value=float(left_metrics[metric_id]),
            right_value=float(right_metrics[metric_id]),
            delta=float(right_metrics[metric_id] - left_metrics[metric_id]),
            denominator=denominator_a,
        ))
    if not deltas:
        reason = blocked or ["没有同时具备可比性身份和覆盖分母的共同指标。"]
        return EvaluationComparison(status="unknown", **base, reasons=reason[:20])
    if blocked:
        return EvaluationComparison(status="partial", **base, metrics=deltas, reasons=blocked[:20])
    return EvaluationComparison(
        status="comparable", **base, metrics=deltas,
        reasons=["模型身份已分别标出；差值仅针对协议、评测数据和指标覆盖一致的项目。"],
    )
