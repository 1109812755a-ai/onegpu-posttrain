"""Explicit source × stage support matrix for the dataset workbench."""

from __future__ import annotations

from .registry import SourceStageCapability

_CAPABILITIES = (
    SourceStageCapability(source_id="upload", source_kind="local_upload", supported_stages=["cpt", "sft", "preference", "grpo"],
                          input_format="canonical-jsonl-v1", status="available"),
    SourceStageCapability(source_id="finemath", source_kind="open_source", supported_stages=["cpt"],
                          input_format="finemath-builder-v2", requires_network=True, status="available"),
    SourceStageCapability(source_id="deepmath", source_kind="open_source", supported_stages=["sft"],
                          input_format="deepmath-mini-sft-v1", requires_network=True, status="available"),
    SourceStageCapability(source_id="openr1", source_kind="open_source", supported_stages=["sft", "preference", "grpo"],
                          input_format="openr1-build-v1", requires_network=True, status="available"),
    SourceStageCapability(source_id="llm-generation", source_kind="model_generated", supported_stages=["sft", "preference"],
                          input_format="canonical-jsonl-v1", requires_credentials=True, status="available"),
    SourceStageCapability(source_id="llm-generation-cpt", source_kind="model_generated", supported_stages=[],
                          input_format="unsupported", requires_credentials=True, status="unsupported",
                          reason="CPT 语料生成需要独立的领域覆盖、去重与污染控制设计。"),
    SourceStageCapability(source_id="llm-generation-grpo", source_kind="model_generated", supported_stages=[],
                          input_format="unsupported", requires_credentials=True, status="unsupported",
                          reason="GRPO 数据生成需要可信、可执行且版本化的 reward/reference。"),
)


def source_stage_capabilities() -> list[SourceStageCapability]:
    """Return validated capability declarations without triggering I/O."""
    return [item.model_copy(deep=True) for item in _CAPABILITIES]


def check_source_stage(source_id: str, stage: str) -> tuple[bool, str | None]:
    """Check a requested combination using the same capability contract as the UI."""
    entry = next((item for item in _CAPABILITIES if item.source_id == source_id), None)
    if entry is None:
        return False, "未知数据来源。"
    if entry.status != "available" or stage not in entry.supported_stages:
        return False, entry.reason or f"来源 {source_id} 当前不支持 {stage} 阶段。"
    return True, None
