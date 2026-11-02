"""Persistent read model for targeted stage workflow evidence."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..lineage import sha256_json


Stage = Literal["cpt", "sft", "preference", "grpo"]
StepKind = Literal["generation", "data_import", "quality", "registration", "training", "merge", "evaluation"]
StepStatus = Literal["not_started", "running", "partial", "succeeded", "failed", "blocked"]


class WorkflowEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    operation_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
    kind: StepKind
    status: StepStatus
    run_id: str | None = None
    execution_id: str | None = None
    dataset_id: str | None = None
    model_id: str | None = None
    evaluation_id: str | None = None
    input_dataset_ids: tuple[str, ...] = ()
    input_model_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    reason: str | None = None
    observed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @model_validator(mode="after")
    def validate_actual_evidence(self) -> WorkflowEvidence:
        if self.status in {"running", "partial", "succeeded", "failed"} and not (self.run_id or self.execution_id):
            raise ValueError("实际运行状态必须引用 run_id 或 execution_id")
        if self.status == "succeeded":
            required_output = {
                "registration": self.dataset_id,
                "training": self.model_id,
                "merge": self.model_id,
                "evaluation": self.evaluation_id,
            }.get(self.kind)
            if self.kind in {"registration", "training", "merge", "evaluation"} and not required_output:
                raise ValueError(f"{self.kind} 成功状态必须引用真实输出身份")
        if self.status in {"failed", "blocked"} and not self.reason:
            raise ValueError("失败或阻塞状态必须说明原因")
        if len(self.artifact_ids) != len(set(self.artifact_ids)):
            raise ValueError("artifact_ids 不能重复")
        if len(self.input_dataset_ids) != len(set(self.input_dataset_ids)):
            raise ValueError("input_dataset_ids 不能重复")
        if len(self.input_model_ids) != len(set(self.input_model_ids)):
            raise ValueError("input_model_ids 不能重复")
        return self


class TargetedWorkflowState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal["1.0"] = "1.0"
    plan_id: str = Field(pattern=r"^plan-targeted-[0-9a-f]{20}$")
    plan_version: int = Field(ge=1)
    stage: Stage
    starting_model_id: str = Field(min_length=1)
    initial_dataset_ids: tuple[str, ...] = ()
    evidence: tuple[WorkflowEvidence, ...] = ()

    @model_validator(mode="after")
    def validate_unique_operations(self) -> TargetedWorkflowState:
        operation_ids = [item.operation_id for item in self.evidence]
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("同一工作流中的 operation_id 必须唯一")
        datasets = set(self.initial_dataset_ids)
        models = {self.starting_model_id}
        for item in self.evidence:
            if item.kind == "training" and item.status in {"running", "partial", "succeeded", "failed"}:
                missing_datasets = set(item.input_dataset_ids) - datasets
                missing_models = set(item.input_model_ids) - models
                if missing_datasets:
                    raise ValueError(f"训练输入数据集尚未登记或缺失: {sorted(missing_datasets)}")
                if missing_models:
                    raise ValueError(f"训练父模型尚未登记或缺失: {sorted(missing_models)}")
                if item.status == "succeeded" and not item.input_dataset_ids:
                    raise ValueError("成功训练必须记录实际消费的 dataset_id")
            elif item.kind in {"merge", "evaluation"} and item.status in {"running", "partial", "succeeded", "failed"}:
                missing_models = set(item.input_model_ids) - models
                if missing_models:
                    raise ValueError(f"模型操作输入尚未登记或缺失: {sorted(missing_models)}")
                if item.status == "succeeded" and not item.input_model_ids:
                    raise ValueError("成功的模型合并/评测必须记录输入 model_id")
            if item.kind == "registration" and item.status == "succeeded" and item.dataset_id:
                datasets.add(item.dataset_id)
            if item.kind in {"training", "merge"} and item.status == "succeeded" and item.model_id:
                models.add(item.model_id)
        return self

    @property
    def state_sha256(self) -> str:
        return sha256_json(self.model_dump(mode="json"))

    def append(self, item: WorkflowEvidence) -> TargetedWorkflowState:
        """Return a new snapshot; existing evidence is never edited in place."""
        return TargetedWorkflowState(
            plan_id=self.plan_id,
            plan_version=self.plan_version,
            stage=self.stage,
            starting_model_id=self.starting_model_id,
            initial_dataset_ids=self.initial_dataset_ids,
            evidence=(*self.evidence, item),
        )


def workflow_state_path(workspace: str | Path, plan_id: str, plan_version: int) -> Path:
    if not re.fullmatch(r"plan-targeted-[0-9a-f]{20}", plan_id):
        raise ValueError("无效 plan_id")
    if plan_version < 1:
        raise ValueError("plan_version 必须为正数")
    return Path(workspace) / "runs" / "targeted-workflows" / plan_id / f"v{plan_version}" / "state.json"


def save_workflow_state(state: TargetedWorkflowState, path: str | Path) -> None:
    """Atomically save the latest read model snapshot."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_sha256": state.state_sha256,
        "state": state.model_dump(mode="json"),
    }
    fd, temporary_name = tempfile.mkstemp(prefix=".targeted-workflow-", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def load_workflow_state(path: str | Path) -> TargetedWorkflowState:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("state"), dict):
        raise TypeError("targeted workflow state 必须是对象")
    state = TargetedWorkflowState.model_validate(payload["state"])
    if payload.get("state_sha256") != state.state_sha256:
        raise ValueError("targeted workflow state 摘要不匹配")
    return state
