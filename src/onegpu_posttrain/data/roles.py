"""Immutable data-role snapshots and safe selection of targeted training seeds."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..lineage import sha256_file
from .decontaminate import BenchmarkIndex
from .io import read_jsonl
from .targeted import DataRoleRegistry, RoleRecord, ScenarioDefinition, label_values


@dataclass(frozen=True)
class SeedSelection:
    records: tuple[dict[str, Any], ...]
    report: dict[str, Any]


@dataclass(frozen=True)
class CandidateCheck:
    accepted: bool
    reasons: tuple[str, ...]
    contamination_hits: tuple[dict[str, Any], ...]
    contamination_index_id: str
    contamination_index_sha256: str
    fuzzy_threshold: float


def _record_question(row: dict[str, Any]) -> str:
    for field in ("question", "problem", "raw_problem", "prompt", "text"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def role_records_from_jsonl(path: str | Path, *, dataset_id: str, role: str) -> list[RoleRecord]:
    dataset_sha256 = sha256_file(path)
    records: list[RoleRecord] = []
    for row in read_jsonl(path):
        record_id = row.get("id")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        problem_id = row.get("problem_id") or metadata.get("problem_id")
        if not isinstance(record_id, str) or not isinstance(problem_id, str):
            raise TypeError("角色登记要求每条 canonical 记录有字符串 id/problem_id")
        parents = metadata.get("parent_record_ids", metadata.get("source_record_ids", []))
        if not isinstance(parents, list) or any(not isinstance(item, str) for item in parents):
            raise TypeError(f"{record_id} 的祖先必须是字符串 ID 列表")
        records.append(RoleRecord(
            dataset_id=dataset_id,
            dataset_sha256=dataset_sha256,
            record_id=record_id,
            problem_id=problem_id,
            role=role,
            parent_record_ids=parents,
        ))
    return records


def register_dataset(
    registry: DataRoleRegistry | None,
    path: str | Path,
    *,
    dataset_id: str,
    role: str,
) -> DataRoleRegistry:
    """Create or extend a role snapshot without mutating an earlier version."""
    rows = role_records_from_jsonl(path, dataset_id=dataset_id, role=role)
    if registry is None:
        return DataRoleRegistry(records=rows)
    return registry.extend(rows)


def save_role_registry(registry: DataRoleRegistry, path: str | Path) -> dict[str, str]:
    """Write a new content-addressed snapshot; refuse to overwrite history."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "registry_id": registry.registry_id,
        "registry_sha256": registry.registry_sha256,
        "registry": registry.model_dump(mode="json"),
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return {"registry_id": registry.registry_id, "registry_sha256": registry.registry_sha256, "path": str(output)}


def load_role_registry(path: str | Path) -> DataRoleRegistry:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    registry = DataRoleRegistry.model_validate(payload["registry"])
    if payload.get("registry_id") != registry.registry_id or payload.get("registry_sha256") != registry.registry_sha256:
        raise ValueError("role registry 身份摘要不匹配")
    return registry


def _roles_by_record(registry: DataRoleRegistry) -> dict[str, RoleRecord]:
    return {record.record_id: record for record in registry.records}


def select_training_seeds(
    dataset_path: str | Path,
    *,
    dataset_id: str,
    registry: DataRoleRegistry,
    definition: ScenarioDefinition,
    scenario_id: str,
    target_count: int,
    seed: int,
    protected_index: BenchmarkIndex | None,
    protected_index_id: str | None,
    protected_index_sha256: str | None,
    label_sidecar_path: str | Path | None = None,
    label_sidecar_sha256: str | None = None,
    fuzzy_threshold: float = 0.85,
) -> SeedSelection:
    """Select only train_seed records with matching labels and no protected overlap.

    An absent contamination index is an error, not evidence of zero contamination.
    """
    if protected_index is None or not protected_index_id or not protected_index_sha256:
        raise ValueError("缺少 diagnostic/final_holdout 污染索引；不能声明通过隔离检查")
    if len(protected_index_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in protected_index_sha256):
        raise ValueError("protected_index_sha256 必须是 SHA256")
    if target_count < 1 or seed < 0:
        raise ValueError("target_count 必须为正数且 seed 不能为负")
    scenario = next((item for item in definition.scenarios if item.scenario_id == scenario_id), None)
    if scenario is None:
        raise ValueError(f"scenario definition 中不存在 {scenario_id}")
    sidecar_labels: dict[str, set[str]] = {}
    if label_sidecar_path is not None:
        actual_sidecar_hash = sha256_file(label_sidecar_path)
        if label_sidecar_sha256 != actual_sidecar_hash:
            raise ValueError("训练种子标签 sidecar 必须提供匹配的 SHA256")
        for label_row in read_jsonl(label_sidecar_path):
            problem_id = label_row.get("problem_id")
            if not isinstance(problem_id, str) or not problem_id.strip():
                raise ValueError("训练种子标签 sidecar 每行必须有 problem_id")
            if problem_id in sidecar_labels:
                raise ValueError(f"训练种子标签 sidecar problem_id 重复: {problem_id}")
            sidecar_labels[problem_id] = label_values(label_row.get(definition.label_field))
    elif label_sidecar_sha256 is not None:
        raise ValueError("声明了 sidecar 摘要但没有提供文件")
    role_by_id = _roles_by_record(registry)
    protected_problems = {
        item.problem_id for item in registry.records if item.role in {"diagnostic_dev", "final_holdout"}
    }
    target_values = set(scenario.values)
    candidates: list[tuple[str, dict[str, Any]]] = []
    rejected: dict[str, int] = {}
    dataset_hash = sha256_file(dataset_path)
    registered_dataset_hashes = {
        record.dataset_sha256 for record in registry.records if record.dataset_id == dataset_id
    }
    if registered_dataset_hashes != {dataset_hash}:
        raise ValueError("训练种子数据集摘要与 role registry 不匹配")

    for row in read_jsonl(dataset_path):
        record_id = row.get("id")
        role_record = role_by_id.get(record_id) if isinstance(record_id, str) else None
        if role_record is None or role_record.dataset_id != dataset_id:
            rejected["unregistered_record"] = rejected.get("unregistered_record", 0) + 1
            continue
        if role_record.role != "train_seed":
            rejected["role_not_train_seed"] = rejected.get("role_not_train_seed", 0) + 1
            continue
        if role_record.problem_id in protected_problems:
            rejected["protected_problem_id_overlap"] = rejected.get("protected_problem_id_overlap", 0) + 1
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        labels = (sidecar_labels.get(role_record.problem_id, set()) if label_sidecar_path is not None
                  else label_values(metadata.get(definition.label_field)))
        if not labels.intersection(target_values):
            rejected["scenario_label_mismatch_or_unknown"] = rejected.get("scenario_label_mismatch_or_unknown", 0) + 1
            continue
        question = _record_question(row)
        if not question:
            rejected["missing_problem_text"] = rejected.get("missing_problem_text", 0) + 1
            continue
        hits = protected_index.check(question, fuzzy_threshold=fuzzy_threshold)
        if hits:
            rejected["contamination"] = rejected.get("contamination", 0) + 1
            continue
        candidates.append((record_id, row))

    candidates.sort(key=lambda item: hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest())
    selected = tuple(row for _, row in candidates[:target_count])
    return SeedSelection(
        records=selected,
        report={
            "schema_version": "1.0",
            "dataset_id": dataset_id,
            "dataset_sha256": dataset_hash,
            "role_registry_id": registry.registry_id,
            "role_registry_sha256": registry.registry_sha256,
            "scenario_definition_id": definition.definition_id,
            "scenario_id": scenario_id,
            "label_sidecar_sha256": label_sidecar_sha256,
            "selection_method": "sha256(seed,record_id):v1",
            "seed": seed,
            "target_count": target_count,
            "matched_before_selection": len(candidates),
            "selected_count": len(selected),
            "shortfall": max(0, target_count - len(selected)),
            "rejected": rejected,
            "contamination_index_id": protected_index_id,
            "contamination_index_sha256": protected_index_sha256,
            "contamination_index": "exact+shingle-jaccard",
            "fuzzy_threshold": fuzzy_threshold,
            "contamination_index_checked": True,
            "contamination_limitations": ["无法证明未命中语义改写或未被该固定索引覆盖的问题"],
        },
    )


def check_generated_candidate(
    candidate: dict[str, Any],
    *,
    registry: DataRoleRegistry,
    protected_index: BenchmarkIndex | None,
    protected_index_id: str | None,
    protected_index_sha256: str | None,
    fuzzy_threshold: float = 0.85,
) -> CandidateCheck:
    """Reject candidates without train-only ancestry or protected-text clearance."""
    if protected_index is None or not protected_index_id or not protected_index_sha256:
        raise ValueError("缺少 diagnostic/final_holdout 污染索引；不能声明通过隔离检查")
    if len(protected_index_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in protected_index_sha256):
        raise ValueError("protected_index_sha256 必须是 SHA256")
    reasons: list[str] = []
    hits_out: list[dict[str, Any]] = []
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    parents = metadata.get("parent_record_ids", [])
    if not isinstance(parents, list) or not parents:
        reasons.append("missing_training_ancestor")
    else:
        by_id = _roles_by_record(registry)
        for parent_id in parents:
            parent = by_id.get(parent_id) if isinstance(parent_id, str) else None
            if parent is None:
                reasons.append("unregistered_ancestor")
            elif parent.role != "train_seed":
                reasons.append("protected_ancestor")
    protected_problems = {
        record.problem_id for record in registry.records
        if record.role in {"diagnostic_dev", "final_holdout"}
    }
    problem_id = candidate.get("problem_id") or metadata.get("problem_id")
    if isinstance(problem_id, str) and problem_id in protected_problems:
        reasons.append("protected_problem_id_overlap")
    question = _record_question(candidate)
    if not question:
        reasons.append("missing_problem_text")
    else:
        hits = protected_index.check(question, fuzzy_threshold=fuzzy_threshold)
        if hits:
            reasons.append("protected_text_match")
            hits_out = [hit.to_dict() for hit in hits]
    return CandidateCheck(
        accepted=not reasons,
        reasons=tuple(sorted(set(reasons))),
        contamination_hits=tuple(hits_out),
        contamination_index_id=protected_index_id,
        contamination_index_sha256=protected_index_sha256,
        fuzzy_threshold=fuzzy_threshold,
    )
