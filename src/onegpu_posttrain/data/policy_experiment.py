"""Pre-registered file-based policy experiments and conservative result joins."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Literal
from uuid import uuid4

import yaml
from pydantic import Field, model_validator

from ..lineage import EvaluationLineage, ModelLineage, RunRecord, sha256_file
from ..reporting.data_report import write_markdown_report
from .io import read_jsonl
from .manifest import load_manifest, validate_manifest_evidence
from .policy import DataPolicy, PolicyModel, digest_json, policy_diff


class Budget(PolicyModel):
    unit: Literal["examples", "preference_pairs", "tokens", "optimizer_steps", "completions"] = "examples"
    value: int = Field(default=1, ge=1)


class ExperimentArm(PolicyModel):
    policy: DataPolicy
    manifest: str | None = None
    budget: Budget = Field(default_factory=Budget)


class ExperimentControls(PolicyModel):
    parent_model_id: str | None = None
    parent_model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    recipe_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    training: dict = Field(default_factory=dict)
    seeds: list[int] = Field(default_factory=lambda: [42], min_length=1)

    @model_validator(mode="after")
    def unique_seeds(self):
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seed 不能重复")
        return self


class EvaluationContract(PolicyModel):
    dataset_ids: list[str] = Field(default_factory=list)
    dataset_sha256: dict[str, str] = Field(default_factory=dict)
    roles: dict[str, Literal["diagnostic_dev", "final_holdout"]] = Field(default_factory=dict)
    protocol_version: str | None = None
    config_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class MetricDefinition(PolicyModel):
    name: str
    unit: Literal["ratio", "number"] = "ratio"
    direction: Literal["higher", "lower"] = "higher"


class DataEffectExperiment(PolicyModel):
    schema_version: Literal["1.0"] = "1.0"
    experiment_id: str = Field(default_factory=lambda: f"exp-data-{uuid4().hex[:16]}", pattern=r"^exp-[a-z0-9-]+$")
    question: str = Field(min_length=1)
    evidence_kind: Literal["real", "fixture"] = "real"
    experiment_type: Literal["pipeline_effect", "controlled_quality_ablation"] = "pipeline_effect"
    independent_variable: str | None = None
    source_pool_id: str | None = None
    split_snapshot_id: str | None = None
    a: ExperimentArm
    b: ExperimentArm
    controls: ExperimentControls = Field(default_factory=ExperimentControls)
    evaluation: EvaluationContract = Field(default_factory=EvaluationContract)
    primary_metrics: list[MetricDefinition] = Field(default_factory=lambda: [MetricDefinition(name="internal_math.accuracy")], min_length=1)
    guardrail_metrics: list[MetricDefinition] = Field(default_factory=list)

    @property
    def config_sha256(self) -> str:
        return digest_json(self.model_dump(mode="json"))


def contained(root: Path, reference: str) -> Path:
    candidate = (root / reference).resolve(strict=True)
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("证据引用越出指定 artifact root")
    return candidate


def write_new_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load_experiment(path: str | Path) -> DataEffectExperiment:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "plan" in payload:
        plan = DataEffectExperiment.model_validate(payload["plan"])
        if payload.get("config_sha256") != plan.config_sha256:
            raise ValueError("实验登记摘要不一致")
        return plan
    return DataEffectExperiment.model_validate(payload)


def definition_issues(plan: DataEffectExperiment) -> list[str]:
    issues = []
    controls, contract = plan.controls, plan.evaluation
    if not plan.source_pool_id or not plan.split_snapshot_id:
        issues.append("来源池与固定 split 尚未准备")
    if not all((controls.parent_model_id, controls.parent_model_sha256, controls.recipe_sha256, controls.training)):
        issues.append("共同模型起点/摘要、recipe 摘要或训练配置不完整")
    if not contract.protocol_version or not contract.config_sha256 or not contract.dataset_ids:
        issues.append("评测协议、解析配置摘要或数据集身份未固定")
    if set(contract.dataset_ids) != set(contract.roles) or set(contract.dataset_ids) != set(contract.dataset_sha256):
        issues.append("评测集合需要逐项声明角色与内容摘要")
    if any(len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
           for value in contract.dataset_sha256.values()):
        issues.append("评测内容摘要无效")
    if plan.a.policy.stage != plan.b.policy.stage:
        issues.append("A/B 阶段不同")
    if plan.experiment_type == "controlled_quality_ablation":
        changes = policy_diff(plan.a.policy, plan.b.policy)
        if len(changes) != 1 or changes[0]["field"] != plan.independent_variable:
            issues.append("受控实验必须且只能改变一个已登记变量")
        if plan.a.budget != plan.b.budget:
            issues.append("受控实验预算的单位与值必须匹配")
    return issues


def _arm_data(arm: ExperimentArm, plan: DataEffectExperiment, root: Path) -> tuple[dict, list[str]]:
    if not arm.manifest:
        return {"status": "not_built"}, ["实验组尚无数据 manifest"]
    path = contained(root, arm.manifest)
    manifest = load_manifest(path)
    issues = validate_manifest_evidence(path)
    md = manifest.metadata
    if md.get("data_policy", {}).get("policy_id") != arm.policy.policy_id:
        issues.append("数据未由登记的策略构建")
    if manifest.stage != arm.policy.stage:
        issues.append("数据阶段与策略不一致")
    if md.get("source_pool_id") != plan.source_pool_id or md.get("split_snapshot_id") != plan.split_snapshot_id:
        issues.append("数据来源池/split 与实验登记不一致")
    canonical = path.parent.parent / "canonical" / f"{manifest.stage}.jsonl"
    if sha256_file(canonical) != manifest.hashes.get("canonical_records_sha256"):
        issues.append("canonical 数据摘要不一致")
    rows = list(read_jsonl(canonical))
    if len(rows) != manifest.sample_count:
        issues.append("canonical 数量与 manifest 不一致")
    if not rows:
        issues.append("空数据不能作为训练实验组")
    tokens = sum(r.get("metadata", {}).get("rendered_tokens", 0)
                 or r.get("metadata", {}).get("chosen_tokens", 0) + r.get("metadata", {}).get("rejected_tokens", 0)
                 for r in rows)
    quality = json.loads((path.parent / manifest.quality_report.path).read_text(encoding="utf-8")) if manifest.quality_report else None
    processing = json.loads((path.parent / manifest.processing_report.path).read_text(encoding="utf-8")) if manifest.processing_report else {}
    coverage = next((s.get("details", {}).get("coverage") for s in processing.get("processing_steps", [])
                     if s.get("details", {}).get("coverage")), None)
    if quality and quality["gate"]["summary_status"] == "FAIL":
        issues.append("数据结构质量 gate 失败")
    data = {"dataset_id": manifest.dataset_id, "content_sha256": manifest.hashes.get("canonical_records_sha256"),
            "policy_id": arm.policy.policy_id, "usage": md.get("usage"), "records": manifest.sample_count,
            "dataset_rendered_tokens": tokens, "unit": "examples" if manifest.stage == "sft" else "preference_pairs",
            "quality": quality, "coverage": coverage, "status": "invalid" if issues else "available"}
    return data, issues


def register_experiment(plan: DataEffectExperiment, *, root: str | Path, output: str | Path) -> dict:
    root = Path(root).resolve(strict=True)
    issues = definition_issues(plan)
    for arm in (plan.a, plan.b):
        _, errors = _arm_data(arm, plan, root)
        issues.extend(errors)
    if issues:
        raise ValueError("实验不能登记：" + "; ".join(issues))
    payload = {"schema_version": "1.0", "experiment_id": plan.experiment_id,
               "registration_id": "registration-" + uuid4().hex, "status": "registered",
               "registered_at": datetime.now(timezone.utc).isoformat(),
               "config_sha256": plan.config_sha256, "plan": plan.model_dump(mode="json")}
    write_new_json(output, payload)
    return payload


def _index(root: Path) -> tuple[dict, dict, dict]:
    runs, evaluations, models = {}, {}, {}
    visited = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {"node_modules", "canonical", "logs", "private"}
                   and not (Path(directory) / d).is_symlink()]
        for name in files:
            if name not in {"lineage.json", "evaluation_run.json", "evaluation_lineage.json"} and Path(directory).name != "model_artifacts":
                continue
            visited += 1
            if visited > 10000:
                raise ValueError("证据扫描达到上限，请缩小 root")
            path = Path(directory) / name
            if path.is_symlink() or path.stat().st_size > 2 * 1024 * 1024:
                continue
            try:
                if name == "evaluation_lineage.json":
                    item = EvaluationLineage.model_validate_json(path.read_text(encoding="utf-8"))
                    target, key = evaluations, item.eval_id
                elif name in {"lineage.json", "evaluation_run.json"}:
                    item = RunRecord.model_validate_json(path.read_text(encoding="utf-8"))
                    target, key = runs, item.run_id
                else:
                    item = ModelLineage.model_validate_json(path.read_text(encoding="utf-8"))
                    target, key = models, item.model_id
            except (ValueError, OSError):
                continue  # Malformed selected evidence will be reported as unavailable, not a pass.
            if key in target:
                raise ValueError(f"重复/歧义证据身份：{key}")
            target[key] = (item, path)
    return runs, evaluations, models


def _budget_observed(arm: ExperimentArm, data: dict, run_dir: Path) -> int | None:
    if arm.budget.unit in {"examples", "preference_pairs"}:
        return data.get("records") if arm.budget.unit == data.get("unit") else None
    if arm.budget.unit == "tokens":
        return data.get("dataset_rendered_tokens")
    if arm.budget.unit == "optimizer_steps":
        states = list((run_dir / "train").rglob("trainer_state.json")) if (run_dir / "train").is_dir() else []
        steps = [json.loads(p.read_text(encoding="utf-8")).get("global_step") for p in states if not p.is_symlink()]
        steps = [value for value in steps if type(value) is int]
        return max(steps) if steps else None
    return None


def compare_experiment(plan: DataEffectExperiment, *, root: str | Path,
                       selections: dict[str, list[dict]] | None = None, output: str | Path | None = None,
                       registration: dict | None = None) -> dict:
    root = Path(root).resolve(strict=True)
    selections = selections or {"a": [], "b": []}
    issues = definition_issues(plan)
    if any(selections.values()) and (not registration or registration.get("status") != "registered"
                                    or registration.get("config_sha256") != plan.config_sha256):
        issues.append("缺少匹配的训练前实验登记；事后计划不能作为预登记证据")
    data, results = {}, {"a": [], "b": []}
    for label in ("a", "b"):
        try:
            data[label], errors = _arm_data(getattr(plan, label), plan, root)
            issues.extend(f"{label}: {error}" for error in errors)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            data[label] = {"status": "unavailable"}
            issues.append(f"{label}: 数据证据不可用：{exc}")
    runs, evaluations, models = _index(root) if any(selections.values()) else ({}, {}, {})
    for label in ("a", "b"):
        arm = getattr(plan, label)
        found_seeds = []
        for selection in selections.get(label, []):
            run_pair = runs.get(selection.get("run_id"))
            eval_pair = evaluations.get(selection.get("eval_id"))
            item = {"run_id": selection.get("run_id"), "eval_id": selection.get("eval_id"),
                    "status": "missing", "metrics": {}, "cost": {"wall_seconds": None, "peak_vram": None}}
            results[label].append(item)
            if run_pair is None or eval_pair is None:
                issues.append(f"{label}: 缺少登记的运行或评测身份")
                continue
            run, run_path = run_pair
            evaluation, _ = eval_pair
            item["status"] = run.status
            item["metrics"] = evaluation.metrics
            item["model_id"] = evaluation.model_id
            if run.status != "completed" or evaluation.status != "completed":
                issues.append(f"{label}: 运行/评测未完成（{run.status}/{evaluation.status}）")
            if run.lineage_status != "complete":
                issues.append(f"{label}: 训练血缘不完整")
            if registration and registration.get("registered_at"):
                registered = datetime.fromisoformat(registration["registered_at"].replace("Z", "+00:00"))
                started = datetime.fromisoformat(run.started_at.replace("Z", "+00:00"))
                if registered > started:
                    issues.append(f"{label}: 实验登记晚于训练启动")
            if run.run_type != "train" or run.experiment_id != plan.experiment_id:
                issues.append(f"{label}: train run 不属于此实验")
            if not any(ref.id == data[label].get("dataset_id") and ref.role == "train"
                       and ref.digest and ref.digest.value == data[label].get("content_sha256") for ref in run.inputs):
                issues.append(f"{label}: train run 没有消费登记的数据身份/摘要")
            if not any(ref.id == plan.controls.parent_model_id and ref.role == "parent"
                       and ref.digest and ref.digest.value == plan.controls.parent_model_sha256 for ref in run.inputs):
                issues.append(f"{label}: 模型起点或摘要不同")
            try:
                expected_recipe = contained(root, run.recipe_path) if run.recipe_path else None
            except (OSError, ValueError):
                expected_recipe = None
            if expected_recipe is None or sha256_file(expected_recipe) != plan.controls.recipe_sha256:
                issues.append(f"{label}: recipe 摘要不匹配")
            resolved_path = run_path.parent / "resolved_config.json"
            if not resolved_path.is_file():
                issues.append(f"{label}: 缺少实际解析配置")
                resolved = {}
            else:
                resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
                if digest_json(resolved) != run.resolved_config_sha256:
                    issues.append(f"{label}: 解析配置摘要不一致")
            training = resolved.get("training", {})
            seed = training.get("seed")
            item["seed"] = seed
            found_seeds.append(seed)
            expected_training = {**plan.controls.training, "seed": seed}
            if seed not in plan.controls.seeds or training != expected_training:
                issues.append(f"{label}: 训练配置或 seed 与登记不一致")
            outputs = {ref.id for ref in run.outputs if ref.kind == "model"}
            belongs = evaluation.model_id in outputs
            if not belongs and evaluation.model_id in models:
                model, _ = models[evaluation.model_id]
                merge = runs.get(model.produced_by_run_id)
                belongs = bool(model.adapter_id in outputs and merge and merge[0].run_type == "merge"
                               and merge[0].status == "completed"
                               and any(ref.id == model.adapter_id for ref in merge[0].inputs)
                               and any(ref.id == model.model_id for ref in merge[0].outputs))
            if not belongs:
                issues.append(f"{label}: 评测模型并非此训练 run 的产物")
            contract = plan.evaluation
            if (set(evaluation.dataset_ids) != set(contract.dataset_ids)
                    or evaluation.protocol_version != contract.protocol_version
                    or evaluation.config_sha256 != contract.config_sha256):
                issues.append(f"{label}: 评测数据集合/协议/配置不一致")
            eval_run_pair = runs.get(evaluation.produced_by_run_id)
            if eval_run_pair is None:
                issues.append(f"{label}: 缺少评测执行记录")
            else:
                eval_run = eval_run_pair[0]
                if eval_run.run_type != "evaluate" or eval_run.status != "completed" or not any(
                        ref.id == evaluation.model_id and ref.kind == "model" for ref in eval_run.inputs):
                    issues.append(f"{label}: 评测执行记录与评测模型不一致")
                for dataset_id, expected_hash in contract.dataset_sha256.items():
                    if not any(ref.id == dataset_id and ref.digest and ref.digest.value == expected_hash
                               for ref in eval_run.inputs):
                        issues.append(f"{label}: 评测样本内容摘要证据不足：{dataset_id}")
            observed = _budget_observed(arm, data[label], run_path.parent)
            item["budget_observed"] = observed
            if observed != arm.budget.value:
                issues.append(f"{label}: 实测预算缺失或与登记不同（{observed}/{arm.budget.value} {arm.budget.unit}）")
            run_json = run_path.parent / "run.json"
            if run_json.is_file():
                raw = json.loads(run_json.read_text(encoding="utf-8"))
                execution = raw.get("train", {})
                item["cost"] = {"wall_seconds": execution.get("wall_seconds"), "peak_vram": execution.get("peak_vram_mib")}
        if selections.get(label) and sorted(found_seeds, key=str) != sorted(plan.controls.seeds, key=str):
            issues.append(f"{label}: 实际 seed 集合缺失、重复或不同")
    metrics = []
    for definition in [*plan.primary_metrics, *plan.guardrail_metrics]:
        metric = {"metric": definition.name, "unit": definition.unit, "direction": definition.direction,
                  "a": None, "b": None, "delta": None, "per_seed": []}
        for seed in plan.controls.seeds:
            pair = [next((r for r in results[label] if r.get("seed") == seed), None) for label in ("a", "b")]
            evidence = [r.get("metrics", {}).get(definition.name, {}) if r else {} for r in pair]
            values = [e.get("value") for e in evidence]
            denominators = [e.get("denominator") for e in evidence]
            if all(pair) and (any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                              or (definition.unit == "ratio" and (not all(type(n) is int and n > 0 for n in denominators)
                                                                  or denominators[0] != denominators[1]))):
                issues.append(f"{definition.name}: 指标或分母缺失/不兼容")
            metric["per_seed"].append({"seed": seed, "a": values[0], "b": values[1], "denominators": denominators})
        for index, label in enumerate(("a", "b")):
            values = [entry[label] for entry in metric["per_seed"]
                      if type(entry[label]) in (int, float) and math.isfinite(entry[label])]
            metric[label] = mean(values) if values else None
        metrics.append(metric)
    has_all = all(results.values()) and all(len(results[label]) == len(plan.controls.seeds) for label in ("a", "b"))
    status = "COMPARABLE" if has_all and not issues else "NOT_COMPARABLE" if any(results.values()) else "NOT_EVALUATED"
    if status == "COMPARABLE":
        for metric in metrics:
            metric["delta"] = metric["b"] - metric["a"]
            differences = [r["b"] - r["a"] for r in metric["per_seed"]]
            metric["delta_std"] = stdev(differences) if len(differences) > 1 else None
    report = {"schema_version": "1.0", "report_id": "der-" + uuid4().hex,
              "created_at": datetime.now(timezone.utc).isoformat(), "experiment_id": plan.experiment_id,
              "plan_sha256": plan.config_sha256, "experiment_type": plan.experiment_type,
              "evidence_kind": plan.evidence_kind,
              "comparison_status": status, "compatibility_issues": list(dict.fromkeys(issues)),
              "policy_changes": policy_diff(plan.a.policy, plan.b.policy), "data": data,
              "runs": results, "model_metrics": metrics,
              "limitations": ["单 seed 仅是描述性观察；所有比例需查看实际分母",
                              "流水线效果可能同时包含数据量、长度和难度变化，不能单独归因于数据质量",
                              "tokens 预算为数据集渲染 token 总量，不等于训练累计 token 或全部计算成本"]}
    if output:
        write_new_json(output, report)
        write_markdown_report(Path(output).with_suffix(".md"), "数据策略对照报告", report)
    return report
