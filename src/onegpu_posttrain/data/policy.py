"""Portable data policies shared by CLI, builders and the optional editor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class PolicyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PolicyRules(PolicyModel):
    correctness: Literal["filter", "report_only"] = "filter"
    completeness: Literal["filter", "report_only"] = "filter"
    answer_parse: Literal["filter", "report_only"] = "filter"
    min_length_ratio: float | None = Field(default=None, gt=0)
    max_length_ratio: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def check_ratio(self):
        if (self.min_length_ratio is None) != (self.max_length_ratio is None):
            raise ValueError("长度比下限与上限必须同时设置或同时为 null")
        if self.min_length_ratio is not None and self.min_length_ratio > self.max_length_ratio:
            raise ValueError("长度比下限不能大于上限")
        return self


class ResourceLimits(PolicyModel):
    max_length: int = Field(default=1024, ge=1, le=131072)
    target_count: int = Field(default=200, ge=1)
    max_scan_rows: int | None = Field(default=None, ge=1)


class PolicyTokenizer(PolicyModel):
    repo: str = Field(default="Qwen/Qwen3-0.6B-Base", min_length=1)
    revision: str = Field(default="da87bfb608c14b7cf20ba1ce41287e8de496c0cd", min_length=1)
    # Operator-configured local path, never an arbitrary browser path.
    local_path: str | None = None
    test_only: bool = False


class PolicyInvariants(PolicyModel):
    canonical_schema: Literal[True] = True
    unique_records: Literal[True] = True
    train_eval_separation: Literal[True] = True
    benchmark_screening: Literal[True] = True
    provenance: Literal[True] = True


class DataPolicy(PolicyModel):
    schema_version: Literal["1.0"] = "1.0"
    version: str = "1"
    title: str = "数据策略"
    parent_policy_id: str | None = None
    stage: Literal["sft", "preference"]
    usage: Literal["exploratory", "baseline"] = "exploratory"
    prompt_template_id: str = "qwen3-math-v1"
    rules: PolicyRules = Field(default_factory=PolicyRules)
    resources: ResourceLimits = Field(default_factory=ResourceLimits)
    tokenizer: PolicyTokenizer = Field(default_factory=PolicyTokenizer)
    invariants: PolicyInvariants = Field(default_factory=PolicyInvariants)

    @model_validator(mode="after")
    def check_stage(self):
        if self.stage == "sft" and self.rules.min_length_ratio is not None:
            raise ValueError("SFT 不支持 Preference 长度比规则")
        if self.stage == "preference" and self.rules.correctness != "filter":
            raise ValueError("正确性排序的 Preference 必须保留已知 true/false 排序依据")
        if self.usage == "baseline":
            if self.tokenizer.test_only:
                raise ValueError("baseline 不允许 test-only tokenizer")
            if "report_only" in self.rules.model_dump().values():
                raise ValueError("report_only 仅适用于显式 exploratory 用途")
        return self

    @property
    def config_sha256(self) -> str:
        payload = self.model_dump(mode="json", exclude={"title", "parent_policy_id"})
        payload["tokenizer"].pop("local_path", None)
        return digest_json(payload)

    @property
    def policy_id(self) -> str:
        return f"policy-{self.stage}-{self.config_sha256}"

    def artifact(self) -> dict:
        config = self.model_dump(mode="json")
        config["tokenizer"]["local_path"] = None
        return {"policy_id": self.policy_id, "config_sha256": self.config_sha256,
                "config": config}


def load_policy(path: str | Path) -> DataPolicy:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return parse_policy(payload)


def parse_policy(payload: dict) -> DataPolicy:
    if isinstance(payload, dict) and "config" in payload:
        if set(payload) != {"policy_id", "config_sha256", "config"}:
            raise ValueError("Unknown policy artifact fields")
        result = DataPolicy.model_validate(payload["config"])
        if payload["policy_id"] != result.policy_id or payload["config_sha256"] != result.config_sha256:
            raise ValueError("策略身份/摘要与配置不一致")
        return result
    return DataPolicy.model_validate(payload)


def policy_templates() -> list[dict]:
    items = []
    for stage in ("sft", "preference"):
        for relaxed in (False, True):
            rules = {"min_length_ratio": 0.5, "max_length_ratio": 2.0} if stage == "preference" else {}
            if relaxed:
                rules.update(completeness="report_only", answer_parse="report_only")
                if stage == "sft":
                    rules["correctness"] = "report_only"
            policy = DataPolicy(stage=stage, rules=PolicyRules(**rules),
                                title=f"{stage.upper()} · {'仅报告语义检查' if relaxed else '基线筛选'}")
            items.append(policy.artifact())
    return items


class SplitSnapshot(PolicyModel):
    schema_version: Literal["1.0"] = "1.0"
    protocol: Literal["policy-problem-split-v1"] = "policy-problem-split-v1"
    seed: int = 42
    sft_weight: int = Field(default=5000, ge=0)
    alignment_weight: int = Field(default=3000, ge=0)
    holdout_weight: int = Field(default=500, ge=1)

    @model_validator(mode="after")
    def check_weights(self):
        if self.sft_weight + self.alignment_weight == 0:
            raise ValueError("至少需要一个训练分组")
        return self

    @property
    def snapshot_id(self) -> str:
        return "split-" + digest_json(self.model_dump(mode="json"))

    def pool(self, problem_id: str) -> str:
        value = int(hashlib.sha256(f"{self.seed}:{problem_id}".encode()).hexdigest()[:16], 16)
        value %= self.sft_weight + self.alignment_weight + self.holdout_weight
        if value < self.sft_weight:
            return "sft"
        return "alignment" if value < self.sft_weight + self.alignment_weight else "holdout"


class SourceDefinition(PolicyModel):
    """Trusted local operator configuration; requests reference its discovered ID."""

    title: str
    input_scope: Literal["raw_source", "existing_dataset"] = "raw_source"
    input_jsonl: str | None = None
    source_cache: str | None = None
    source_repo: str
    source_revision: str
    source_config: str | None = None
    expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    benchmark_jsonl: str
    license_status: Literal["declared", "unavailable"]
    license_note: str = Field(min_length=1)
    existing_stage: Literal["sft", "preference"] | None = None
    split: SplitSnapshot = Field(default_factory=SplitSnapshot)
    tokenizer: PolicyTokenizer = Field(default_factory=PolicyTokenizer)

    @model_validator(mode="after")
    def check_source(self):
        if (self.input_jsonl is None) == (self.source_cache is None):
            raise ValueError("必须且只能指定本地 input_jsonl 或 source_cache")
        if self.input_scope == "existing_dataset" and self.existing_stage is None:
            raise ValueError("已筛选数据必须声明已有训练阶段")
        return self


def policy_diff(a: DataPolicy, b: DataPolicy) -> list[dict]:
    def flatten(value: dict, prefix: str = "") -> dict:
        out = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(item, dict):
                out.update(flatten(item, name))
            else:
                out[name] = item
        return out
    left = flatten({k: v for k, v in a.artifact()["config"].items() if k not in {"title", "parent_policy_id"}})
    right = flatten({k: v for k, v in b.artifact()["config"].items() if k not in {"title", "parent_policy_id"}})
    return [{"field": key, "a": left.get(key), "b": right.get(key)}
            for key in sorted(left.keys() | right.keys()) if left.get(key) != right.get(key)]


def exploration_preflight(manifest, *, record_count: int, recipe_max_length: int | None) -> dict:
    metadata = manifest.metadata
    if record_count <= 0 or manifest.sample_count <= 0:
        raise ValueError("空数据不能启动训练")
    if metadata.get("usage") != "exploratory":
        return {}
    policy = parse_policy(metadata["data_policy"])
    maximum = metadata.get("max_rendered_tokens")
    if recipe_max_length is None or maximum is None:
        raise ValueError("探索数据需要明确的训练 max_length 与实测渲染长度")
    if maximum > recipe_max_length:
        raise ValueError("数据完整长度超过 recipe max_length；不能静默截断完整答案")
    if metadata.get("policy_integrity") != "passed":
        raise ValueError("探索数据的固定完整性要求未通过")
    return {"usage": "exploratory", "policy_id": policy.policy_id,
            "policy_config_sha256": policy.config_sha256, "actual_record_count": record_count,
            "official_mini_acceptance": "not_assessed", "warnings": ["小规模探索不构成正式 Mini 或稳定能力结论"]}
