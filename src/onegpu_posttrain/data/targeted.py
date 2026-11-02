"""Versioned contracts for evaluation-driven data targeting.

These models deliberately keep absent evidence distinct from a measured zero.
They contain no provider calls or filesystem side effects.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..lineage import sha256_json


class TargetedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ScenarioLabel(TargetedModel):
    scenario_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    title: str = Field(min_length=1)
    values: list[str] = Field(min_length=1)


def label_values(value: object) -> set[str]:
    """Normalize only declared string tags; do not infer labels from prose."""
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        candidates = value
    else:
        return set()
    return {item.strip() for item in candidates if item.strip()}


class ScenarioDefinition(TargetedModel):
    """A versioned mapping from existing dataset labels to named scenarios."""

    schema_version: Literal["1.0"] = "1.0"
    taxonomy_id: str = Field(min_length=1)
    taxonomy_version: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    label_source: Literal["metadata", "sidecar"]
    label_field: str = Field(min_length=1)
    sidecar_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    scenarios: list[ScenarioLabel] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_definition(self) -> ScenarioDefinition:
        ids = [scenario.scenario_id for scenario in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("场景 ID 不能重复")
        if self.label_source == "sidecar" and self.sidecar_sha256 is None:
            raise ValueError("sidecar 标签必须固定 sidecar_sha256")
        if self.label_source == "metadata" and self.sidecar_sha256 is not None:
            raise ValueError("metadata 标签不能声明 sidecar_sha256")
        return self

    @property
    def config_sha256(self) -> str:
        return sha256_json(self.model_dump(mode="json"))

    @property
    def definition_id(self) -> str:
        return f"scenario-def-{self.config_sha256[:20]}"


class ScenarioDiagnosis(TargetedModel):
    scenario_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    sample_count: int = Field(ge=0)
    denominator: int | None = Field(default=None, ge=0)
    correct: int = Field(default=0, ge=0)
    incorrect: int = Field(default=0, ge=0)
    parse_failed: int = Field(default=0, ge=0)
    unsupported: int = Field(default=0, ge=0)
    excluded: int = Field(default=0, ge=0)
    missing_prediction: int = Field(default=0, ge=0)
    accuracy: float | None = Field(default=None, ge=0, le=1)
    status: Literal["available", "insufficient_evidence", "not_evaluated"]

    @model_validator(mode="after")
    def validate_evidence(self) -> ScenarioDiagnosis:
        observed = (self.correct + self.incorrect + self.parse_failed + self.unsupported
                    + self.excluded + self.missing_prediction)
        if observed != self.sample_count:
            raise ValueError("各结果计数之和必须等于 sample_count")
        scored = self.correct + self.incorrect
        if self.denominator is None:
            if scored or self.accuracy is not None or self.status != "not_evaluated":
                raise ValueError("有可评分样本时必须提供 denominator；缺失分母不能产生指标")
            return self
        if self.denominator != scored:
            raise ValueError("accuracy denominator 必须等于 correct + incorrect")
        if self.denominator == 0:
            if self.accuracy is not None or self.status != "not_evaluated":
                raise ValueError("空/无可评分场景必须是 not_evaluated，accuracy 为 null")
        else:
            expected = self.correct / self.denominator
            if self.accuracy is None or abs(self.accuracy - expected) > 1e-12:
                raise ValueError("accuracy 必须由 correct / denominator 得出")
            if self.status == "not_evaluated":
                raise ValueError("存在可评分样本时不能标成 not_evaluated")
        return self


class ScenarioReport(TargetedModel):
    schema_version: Literal["1.0"] = "1.0"
    eval_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_version: str = Field(min_length=1)
    predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario_definition_id: str = Field(pattern=r"^scenario-def-[0-9a-f]{20}$")
    scenario_definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    label_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    threshold_source: str = Field(min_length=1)
    minimum_samples_required: int = Field(ge=1)
    input_record_count: int = Field(ge=0)
    prediction_count: int = Field(ge=0)
    unmatched_prediction_count: int = Field(default=0, ge=0)
    missing_prediction_count: int = Field(default=0, ge=0)
    unknown_label_count: int = Field(default=0, ge=0)
    overlapping_sample_count: int = Field(default=0, ge=0)
    scenarios: list[ScenarioDiagnosis]

    @model_validator(mode="after")
    def validate_scenarios(self) -> ScenarioReport:
        if self.prediction_count - self.unmatched_prediction_count + self.missing_prediction_count != self.input_record_count:
            raise ValueError("预测、未匹配与缺失计数必须覆盖输入评测记录")
        ids = [scenario.scenario_id for scenario in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("诊断报告中的场景 ID 不能重复")
        for scenario in self.scenarios:
            if scenario.denominator is not None and scenario.denominator > 0:
                expected = "available" if scenario.denominator >= self.minimum_samples_required else "insufficient_evidence"
                if scenario.status != expected:
                    raise ValueError(f"场景 {scenario.scenario_id} 的证据状态应为 {expected}")
        return self

    @property
    def report_sha256(self) -> str:
        return sha256_json(self.model_dump(mode="json"))

    @property
    def report_id(self) -> str:
        return f"scenario-report-{self.report_sha256[:20]}"


class DataIdentity(TargetedModel):
    dataset_id: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    role: Literal["train_seed", "diagnostic_dev", "final_holdout"]


class RoleRecord(TargetedModel):
    dataset_id: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_id: str = Field(min_length=1)
    problem_id: str = Field(min_length=1)
    role: Literal["train_seed", "diagnostic_dev", "final_holdout"]
    parent_record_ids: list[str] = Field(default_factory=list)


class DataRoleRegistry(TargetedModel):
    """Append-only snapshot of record roles and derivation ancestry."""

    schema_version: Literal["1.0"] = "1.0"
    version: int = Field(default=1, ge=1)
    records: list[RoleRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_role_graph(self) -> DataRoleRegistry:
        by_id = {record.record_id: record for record in self.records}
        if len(by_id) != len(self.records):
            raise ValueError("role registry 中 record_id 必须全局唯一")
        dataset_hashes: dict[str, str] = {}
        for record in self.records:
            prior_hash = dataset_hashes.setdefault(record.dataset_id, record.dataset_sha256)
            if prior_hash != record.dataset_sha256:
                raise ValueError(f"dataset_id {record.dataset_id} 对应多个 content hash")
        priority = {"train_seed": 0, "diagnostic_dev": 1, "final_holdout": 2}
        for record in self.records:
            if len(record.parent_record_ids) != len(set(record.parent_record_ids)):
                raise ValueError(f"{record.record_id} 的 parent_record_ids 不能重复")
            for parent_id in record.parent_record_ids:
                parent = by_id.get(parent_id)
                if parent is None:
                    raise ValueError(f"缺少派生记录的已登记祖先: {parent_id}")
                if priority[parent.role] > priority[record.role]:
                    raise ValueError("派生记录不能将 diagnostic_dev/final_holdout 祖先降级为训练数据")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(record_id: str) -> None:
            if record_id in visiting:
                raise ValueError("role registry 的 parent_record_ids 存在循环")
            if record_id in visited:
                return
            visiting.add(record_id)
            for parent_id in by_id[record_id].parent_record_ids:
                visit(parent_id)
            visiting.remove(record_id)
            visited.add(record_id)

        for record_id in by_id:
            visit(record_id)
        return self

    @property
    def registry_sha256(self) -> str:
        return sha256_json(self.model_dump(mode="json"))

    @property
    def registry_id(self) -> str:
        return f"role-registry-v{self.version}-{self.registry_sha256[:16]}"

    def extend(self, records: list[RoleRecord]) -> DataRoleRegistry:
        """Return a new version; this snapshot and its original roles stay unchanged."""
        return DataRoleRegistry(version=self.version + 1, records=[*self.records, *records])


class ModelIdentity(TargetedModel):
    model_id: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvaluationIdentity(TargetedModel):
    eval_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_version: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    role: Literal["diagnostic_dev", "final_holdout"]


class EvaluationContract(TargetedModel):
    """Pre-registered re-evaluation dataset/protocol, before a result exists."""

    dataset_id: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_version: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    role: Literal["diagnostic_dev", "final_holdout"]


class MetricTarget(TargetedModel):
    name: str = Field(min_length=1)
    direction: Literal["higher", "lower"]
    threshold: float = Field(allow_inf_nan=False)
    threshold_source: str = Field(min_length=1)
    minimum_samples: int = Field(ge=1)


class GenerationBudget(TargetedModel):
    max_requests: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    max_retries: int = Field(default=0, ge=0)
    timeout_seconds: float = Field(default=120, gt=0, le=600, allow_inf_nan=False)
    target_candidates: int = Field(ge=1)


class GenerationParameters(TargetedModel):
    temperature: float = Field(default=0.2, ge=0, le=2, allow_inf_nan=False)
    provider_seed: int | None = Field(default=None, ge=0)


class TargetedGenerationPlan(TargetedModel):
    """Immutable preregistration tying a diagnosis to isolated data and re-eval."""

    schema_version: Literal["1.0"] = "1.0"
    version: int = Field(default=1, ge=1)
    scenario_report_id: str = Field(pattern=r"^scenario-report-[0-9a-f]{20}$")
    scenario_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario_definition_id: str = Field(pattern=r"^scenario-def-[0-9a-f]{20}$")
    scenario_definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    role_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_scenario_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    target: MetricTarget
    diagnostic: EvaluationIdentity
    training_seed: DataIdentity
    starting_model: ModelIdentity
    final_evaluation: EvaluationContract
    stage: Literal["sft", "preference"]
    comparison_design: Literal["before_after_pipeline", "matched_budget_ab"]
    provider_profile_id: str = Field(min_length=1)
    prompt_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_parameters: GenerationParameters = Field(default_factory=GenerationParameters)
    budget: GenerationBudget
    training_budget: int = Field(ge=1)
    seed: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_roles_and_lineage(self) -> TargetedGenerationPlan:
        if self.diagnostic.role != "diagnostic_dev":
            raise ValueError("诊断评测必须使用 diagnostic_dev 角色")
        if self.final_evaluation.role != "final_holdout":
            raise ValueError("正式复测必须使用独立 final_holdout 角色")
        if self.training_seed.role != "train_seed":
            raise ValueError("生成种子必须使用 train_seed 角色")
        if self.starting_model.model_id != self.diagnostic.model_id:
            raise ValueError("起点模型必须与诊断评测使用的模型身份一致")
        if self.diagnostic.dataset_id == self.training_seed.dataset_id:
            raise ValueError("diagnostic_dev 不能兼作训练种子")
        if self.final_evaluation.dataset_id in {self.training_seed.dataset_id, self.diagnostic.dataset_id}:
            raise ValueError("final_holdout 必须与训练种子和诊断集分离")
        return self

    @property
    def config_sha256(self) -> str:
        return sha256_json(self.model_dump(mode="json"))

    @property
    def plan_id(self) -> str:
        return f"plan-targeted-{self.config_sha256[:20]}"
