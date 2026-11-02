"""Deterministic, lineage-preserving composition of registered stage datasets."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..lineage import sha256_json
from .dataset_import import RegistrationResult, register_jsonl
from .importer import validate_canonical_jsonl
from .manifest import (
    DatasetComposition,
    DatasetManifest,
    DatasetParent,
    load_manifest,
    resolve_canonical_path,
    sha256_file,
)
from .registry import DataSource, TrainingStage

_DATASET_ROOTS = ("data/processed", "artifacts/ui-demo", "artifacts/synthetic")
_BLOCKED_ROLES = {"diagnostic_dev", "final_holdout", "holdout", "eval", "evaluation", "benchmark"}
_MAX_PARENT_BYTES = 256 * 1024 * 1024
_MAX_PARENT_RECORDS = 250_000


class MixtureInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    stage: TrainingStage
    display_name: str = Field(default="组合数据集", min_length=1, max_length=120)
    dataset_ids: list[str] = Field(min_length=2, max_length=8)
    strategy: Literal["concat", "weighted_sample"] = "concat"
    budget: int | None = Field(default=None, ge=1, le=250_000)
    weights: dict[str, float] = Field(default_factory=dict, max_length=8)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)

    @field_validator("display_name")
    @classmethod
    def nonblank_display_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("组合数据集名称不能为空")
        return value

    @field_validator("dataset_ids")
    @classmethod
    def unique_ids(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 200 for value in values):
            raise ValueError("dataset ID 不能为空且长度不能超过 200")
        if len(values) != len(set(values)):
            raise ValueError("组合数据集 ID 不能重复")
        return values

    @model_validator(mode="after")
    def validate_strategy(self):
        if self.strategy == "concat" and (self.budget is not None or self.weights):
            raise ValueError("拼接策略不接受采样预算或权重")
        if self.strategy == "weighted_sample" and self.budget is None:
            raise ValueError("加权采样必须指定目标预算")
        if self.weights and set(self.weights) != set(self.dataset_ids):
            raise ValueError("权重必须恰好覆盖所有所选数据集")
        if any(not math.isfinite(value) or value <= 0 for value in self.weights.values()):
            raise ValueError("权重必须为有限正数")
        return self


class _Parent:
    def __init__(self, workspace: Path, manifest_path: Path, manifest: DatasetManifest, canonical_path: Path):
        self.workspace = workspace
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.canonical_path = canonical_path
        self.content_sha256 = manifest.hashes["canonical_records_sha256"]
        self.rows: list[dict] = []
        self.roles: set[str] = set()
        self.ids: list[str] = []


def _find_parent(workspace: Path, dataset_id: str) -> _Parent:
    matches: list[tuple[Path, DatasetManifest]] = []
    visited = 0
    for relative in _DATASET_ROOTS:
        root = workspace / relative
        if not root.is_dir() or root.is_symlink():
            continue
        for current, dirs, files in os.walk(root, followlinks=False):
            here = Path(current)
            dirs[:] = [name for name in dirs if not name.startswith(".") and not (here / name).is_symlink()]
            if here.name != "manifests":
                continue
            for name in files:
                if Path(name).suffix.lower() not in {".yaml", ".yml", ".json"}:
                    continue
                visited += 1
                if visited > 5000:
                    raise ValueError("本地 manifest 数量超过组合索引上限 5000")
                path = here / name
                if path.is_symlink():
                    continue
                try:
                    manifest = load_manifest(path)
                except (OSError, ValueError, yaml.YAMLError):
                    continue
                if manifest.dataset_id == dataset_id:
                    matches.append((path, manifest))
    if len(matches) != 1:
        raise ValueError(f"数据集 {dataset_id} 必须唯一匹配一个本地登记 manifest（找到 {len(matches)} 个）")
    path, manifest = matches[0]
    if manifest.lineage is None or manifest.processing_report is None:
        raise ValueError(f"数据集 {dataset_id} 缺少登记 lineage 或 processing report")
    canonical = resolve_canonical_path(path, manifest)
    return _Parent(workspace, path, manifest, canonical)


def _check_parent(parent: _Parent, stage: str, ancestors: set[str] | None = None) -> None:
    ancestors = set() if ancestors is None else ancestors
    dataset_id = parent.manifest.dataset_id
    if dataset_id in ancestors:
        raise ValueError(f"数据血缘存在循环引用：{dataset_id}")
    ancestors.add(dataset_id)
    manifest = parent.manifest
    if manifest.stage != stage:
        raise ValueError(f"数据集 {manifest.dataset_id} 阶段为 {manifest.stage}，不能加入 {stage} 组合")
    if manifest.sample_count > _MAX_PARENT_RECORDS or parent.canonical_path.stat().st_size > _MAX_PARENT_BYTES:
        raise ValueError(f"数据集 {manifest.dataset_id} 超过组合输入上限（{_MAX_PARENT_RECORDS} 条 / 256 MiB）")
    role = manifest.metadata.get("dataset_role") or manifest.metadata.get("role") or manifest.metadata.get("split")
    if isinstance(role, str) and role.lower() in _BLOCKED_ROLES:
        raise ValueError(f"数据集 {manifest.dataset_id} 角色 {role} 属于隔离/评测数据，不能用于训练组合")
    # The whole file is schema-validated and hashed before it can be selected.
    roles: set[str] = set()
    rows: list[dict] = []

    def capture(row: dict) -> None:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        row_role = metadata.get("dataset_role") or metadata.get("role") or metadata.get("split")
        if isinstance(row_role, str):
            roles.add(row_role.lower())
        rows.append(row)

    with parent.canonical_path.open("rb") as stream:
        validation = validate_canonical_jsonl(stream, stage, on_record=capture)
    if not validation.valid or validation.record_count != manifest.sample_count:
        raise ValueError(f"数据集 {manifest.dataset_id} canonical schema 无效或记录数与 manifest 不一致")
    forbidden = roles & _BLOCKED_ROLES
    if forbidden:
        raise ValueError(f"数据集 {manifest.dataset_id} 含隔离角色 {', '.join(sorted(forbidden))}，不能用于训练组合")
    if manifest.composition is not None:
        declared = manifest.composition.parents
        if manifest.lineage is None or manifest.lineage.parents != [item.dataset_id for item in declared]:
            raise ValueError(f"数据集 {manifest.dataset_id} 的组合父级与 lineage 不一致")
        for item in declared:
            ancestor = _find_parent(parent.workspace, item.dataset_id)
            if ancestor.content_sha256 != item.content_sha256:
                raise ValueError(f"父数据集 {item.dataset_id} 摘要已变化，不能信任组合数据血缘")
            _check_parent(ancestor, stage, ancestors)
    ancestors.remove(dataset_id)
    parent.rows, parent.roles = rows, roles
    parent.ids = [row["id"] for row in rows]


def _validate_grpo_contract(workspace: Path, parents: list[_Parent]) -> None:
    # Keep composition readiness identical to the stage Launcher contract check:
    # a boolean declaration alone is not evidence that the allowlisted reward
    # implementation runs or that every row has a verified reference.
    from ..ui.training_preflight import _inspect_grpo_reward_contract

    contracts = []
    for parent in parents:
        metadata = parent.manifest.metadata
        if (metadata.get("reward_contract_verified") is not True
                or not metadata.get("reward_contract_id")
                or not metadata.get("reward_contract_protocol_sha256")):
            raise ValueError(f"GRPO 数据集 {parent.manifest.dataset_id} 缺少已验证、可执行的 reward/reference 契约")
        _, blockers, _ = _inspect_grpo_reward_contract(workspace, parent.manifest_path, parent.manifest)
        if blockers:
            raise ValueError(f"GRPO 数据集 {parent.manifest.dataset_id} reward/reference 契约不可用：{blockers[0]}")
        contracts.append((metadata.get("reward_contract_id"), metadata.get("reward_contract_protocol_sha256"),
                          metadata.get("reward_contract_implementation_sha256")))
    if len(set(contracts)) != 1:
        raise ValueError("GRPO 数据集 reward/reference 契约不一致")


def _allocations(parents: list[_Parent], budget: int, weights: dict[str, float]) -> list[int]:
    total = sum(len(parent.rows) for parent in parents)
    if budget > total:
        raise ValueError(f"采样预算 {budget} 超过可用记录总数 {total}；当前策略不允许有放回抽样")
    normalized = [weights.get(parent.manifest.dataset_id, 1.0) for parent in parents]
    scale = sum(normalized)
    quotas = [budget * weight / scale for weight in normalized]
    counts = [min(len(parent.rows), math.floor(quota)) for parent, quota in zip(parents, quotas)]
    remaining = budget - sum(counts)
    order = sorted(range(len(parents)), key=lambda i: (-(quotas[i] - math.floor(quotas[i])), i))
    while remaining:
        progressed = False
        for index in order:
            if counts[index] < len(parents[index].rows):
                counts[index] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise ValueError("目标预算无法在无放回条件下分配")
    return counts


def _normalized_weights(input: MixtureInput) -> dict[str, float]:
    raw = input.weights or {dataset_id: 1.0 for dataset_id in input.dataset_ids}
    total = sum(raw.values())
    return {dataset_id: raw[dataset_id] / total for dataset_id in input.dataset_ids}


def preflight_mixture(workspace: str | Path, request: MixtureInput) -> dict:
    root = Path(workspace).resolve(strict=True)
    parents = [_find_parent(root, dataset_id) for dataset_id in request.dataset_ids]
    for parent in parents:
        _check_parent(parent, request.stage)
    if request.stage == "grpo":
        _validate_grpo_contract(root, parents)
    selected_counts = ([len(parent.rows) for parent in parents] if request.strategy == "concat"
                       else _allocations(parents, int(request.budget), request.weights))
    weights = _normalized_weights(request) if request.strategy == "weighted_sample" else {}
    config = {
        "protocol": "dataset-mixture-v1", "stage": request.stage, "display_name": request.display_name,
        "strategy": request.strategy,
        "parents": [{"dataset_id": p.manifest.dataset_id, "content_sha256": p.content_sha256} for p in parents],
        "weights": weights, "budget": request.budget if request.strategy == "weighted_sample" else None,
        "seed": request.seed if request.strategy == "weighted_sample" else None,
        "selection": "deterministic-without-replacement-v1",
    }
    return {
        "status": "ready", "stage": request.stage, "strategy": request.strategy,
        "dataset_count": len(parents), "total_records": sum(selected_counts),
        "available_records": sum(len(parent.rows) for parent in parents),
        "config": config, "config_sha256": sha256_json(config),
        "parents": [{"dataset_id": parent.manifest.dataset_id, "stage": parent.manifest.stage,
                     "record_count": len(parent.rows), "content_sha256": parent.content_sha256,
                     "quality_report_id": parent.manifest.quality_report.report_id if parent.manifest.quality_report else None,
                     "quality_status": parent.manifest.metadata.get("quality_status", "reported"),
                     "source_kind": parent.manifest.origin.kind if parent.manifest.origin else "unknown",
                     "selected_records": selected_counts[index], "normalized_weight": weights.get(parent.manifest.dataset_id)}
                    for index, parent in enumerate(parents)],
        "warnings": ["组合数据将作为新数据集单独质检；父数据集质量状态不会自动继承。",
                     "数据集组合不会核验许可授权，所有父来源的许可声明需分别确认。"],
    }


def build_mixture(workspace: str | Path, request: MixtureInput) -> tuple[RegistrationResult, dict]:
    root = Path(workspace).resolve(strict=True)
    preview = preflight_mixture(root, request)
    parents = [_find_parent(root, dataset_id) for dataset_id in request.dataset_ids]
    for parent in parents:
        _check_parent(parent, request.stage)
    if request.stage == "grpo":
        _validate_grpo_contract(root, parents)
    config_sha256 = preview["config_sha256"]
    weights = _normalized_weights(request) if request.strategy == "weighted_sample" else {}
    selected: list[tuple[_Parent, int]] = []
    if request.strategy == "concat":
        selected = [(parent, index) for parent in parents for index in range(len(parent.rows))]
    else:
        counts = _allocations(parents, int(request.budget), request.weights)
        rng = random.Random(request.seed)
        for parent, count in zip(parents, counts):
            selected.extend((parent, index) for index in sorted(rng.sample(range(len(parent.rows)), count)))

    output_counts = {parent.manifest.dataset_id: 0 for parent in parents}
    seen_ids: dict[str, set[str]] = {parent.manifest.dataset_id: set() for parent in parents}
    provenance_path = None
    jsonl_path = None
    try:
        with tempfile.NamedTemporaryFile("w+b", prefix=".dataset-mixture-", suffix=".jsonl", dir=root, delete=False) as output:
            jsonl_path = Path(output.name)
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", prefix=".dataset-mixture-provenance-", suffix=".jsonl", dir=root, delete=False) as provenance:
                provenance_path = Path(provenance.name)
                for output_index, (parent, row_index) in enumerate(selected):
                    row = parent.rows[row_index]
                    output.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n")
                    record_id = row.get("id")
                    provenance.write(json.dumps({
                        "output_index": output_index,
                        "dataset_id": parent.manifest.dataset_id,
                        "parent_content_sha256": parent.content_sha256,
                        "record_id": record_id if isinstance(record_id, str) else None,
                        "source_line": row_index + 1,
                    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
                    output_counts[parent.manifest.dataset_id] += 1
                    if isinstance(record_id, str):
                        seen_ids[parent.manifest.dataset_id].add(record_id)
                output.flush(); os.fsync(output.fileno())
                provenance.flush(); os.fsync(provenance.fileno())

        cross_source_duplicates = sum(
            len(seen_ids[left.manifest.dataset_id] & seen_ids[right.manifest.dataset_id])
            for i, left in enumerate(parents) for right in parents[i + 1:]
        )
        provenance_digest = sha256_file(provenance_path)
        composition = DatasetComposition(
            stage=request.stage, strategy=request.strategy,
            parents=[DatasetParent(dataset_id=parent.manifest.dataset_id, content_sha256=parent.content_sha256,
                                   quality_report_id=parent.manifest.quality_report.report_id if parent.manifest.quality_report else None)
                     for parent in parents],
            config_sha256=config_sha256, output_record_count=len(selected),
            seed=request.seed if request.strategy == "weighted_sample" else None,
            weights=weights, budget=request.budget if request.strategy == "weighted_sample" else None,
            provenance_sha256=provenance_digest,
        )
        source = DataSource(
            source_id=f"source-mixture-{config_sha256[:20]}", kind="dataset_mixture", name=request.display_name,
            content_sha256=None, declared_by="adapter", revision_status="not_provided",
            license_status="unknown", attribution="许可证状态继承自多个父数据集；请逐一确认授权。",
        )
        metadata = {
            "composition_protocol": "dataset-mixture-v1", "display_name": request.display_name,
            "parent_dataset_ids": request.dataset_ids,
            "composition_config_sha256": config_sha256, "composition_strategy": request.strategy,
            "source_record_counts": output_counts, "cross_source_duplicate_ids": cross_source_duplicates,
            "dataset_role": "train",
            **({"reward_contract_verified": parents[0].manifest.metadata.get("reward_contract_verified"),
                "reward_contract_id": parents[0].manifest.metadata.get("reward_contract_id"),
                "reward_contract_protocol_sha256": parents[0].manifest.metadata.get("reward_contract_protocol_sha256"),
                "reward_contract_implementation_sha256": parents[0].manifest.metadata.get("reward_contract_implementation_sha256")}
               if request.stage == "grpo" else {}),
        }
        # Verify every input immediately before the existing atomic registration commit.
        for parent in parents:
            if sha256_file(parent.canonical_path) != parent.content_sha256:
                raise ValueError(f"父数据集 {parent.manifest.dataset_id} 在组合期间发生变化；已拒绝登记")
        with jsonl_path.open("rb") as canonical_stream:
            result = register_jsonl(
                canonical_stream, workspace=root, output_root=root / "data" / "processed" / "dataset-mixtures",
                stage=request.stage, origin=source, registration_metadata=metadata,
                identity_material_sha256=config_sha256, composition=composition, provenance_file=provenance_path,
            )
        return result, preview | {"dataset_id": result.dataset_id, "display_name": request.display_name,
                                 "canonical_sha256": result.canonical_sha256,
                                 "record_count": result.record_count, "quality_status": "reported",
                                 "source_record_counts": output_counts,
                                 "cross_source_duplicate_ids": cross_source_duplicates}
    finally:
        if jsonl_path:
            jsonl_path.unlink(missing_ok=True)
        if provenance_path:
            provenance_path.unlink(missing_ok=True)
