"""Deterministic, auditable training/validation partition helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from ..lineage import sha256_json

SPLIT_ALGORITHM = "onegpu-group-sha256-v1"
MIN_VALIDATION_RATIO = 0.05
MAX_VALIDATION_RATIO = 0.4


class TrainingValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["none", "registered", "random_split"] = "none"
    dataset_id: str | None = Field(default=None, min_length=1, max_length=200)
    ratio: float | None = Field(default=None, gt=0, lt=1)
    seed: StrictInt | None = Field(default=None, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def validate_mode_fields(self):
        if self.mode == "registered":
            if not self.dataset_id or self.ratio is not None or self.seed is not None:
                raise ValueError("独立验证模式必须只提供 dataset_id")
        elif self.mode == "random_split":
            if self.dataset_id is not None or self.ratio is None or self.seed is None:
                raise ValueError("随机切分模式必须提供 ratio 和 seed，且不能提供 dataset_id")
            if not MIN_VALIDATION_RATIO <= self.ratio <= MAX_VALIDATION_RATIO:
                raise ValueError(f"验证比例必须在 {MIN_VALIDATION_RATIO:.0%}–{MAX_VALIDATION_RATIO:.0%} 之间")
        elif self.dataset_id is not None or self.ratio is not None or self.seed is not None:
            raise ValueError("不使用验证时不得附带验证集、比例或随机种子")
        return self


def _record_identity(row: Any, index: int) -> tuple[str, str]:
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    row_digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    metadata = row.get("metadata") if isinstance(row, dict) and isinstance(row.get("metadata"), dict) else {}
    group = next((metadata.get(key) for key in (
        "problem_id", "source_group_id", "group_id", "source_record_id", "source_id",
    ) if isinstance(metadata.get(key), (str, int)) and str(metadata[key]).strip()), None)
    if group is None and isinstance(row, dict):
        # Prompts duplicated across variants are kept on one side of the split.
        prompt = row.get("prompt", row.get("messages", row.get("text", row.get("input"))))
        if prompt is not None:
            group = "prompt:" + hashlib.sha256(json.dumps(
                prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest()
    # Rows without a grouping signal remain atomic but are independently assigned.
    return row_digest, str(group) if group is not None else f"row:{index}:{row_digest}"


def read_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"JSONL 第 {line_number} 行为空，无法稳定切分")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 第 {line_number} 行格式无效") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL 第 {line_number} 行必须是对象")
            records.append(row)
    return records


def plan_random_split(records: list[dict[str, Any]], *, ratio: float, seed: int) -> dict[str, Any]:
    """Partition whole semantic groups deterministically and expose verifiable identities."""
    if not MIN_VALIDATION_RATIO <= ratio <= MAX_VALIDATION_RATIO:
        raise ValueError(f"验证比例必须在 {MIN_VALIDATION_RATIO:.0%}–{MAX_VALIDATION_RATIO:.0%} 之间")
    if seed < 0 or seed > 2**31 - 1:
        raise ValueError("随机种子必须在 0–2147483647 之间")
    if len(records) < 2:
        raise ValueError("随机切分至少需要 2 条记录")
    row_identities = [_record_identity(row, index) for index, row in enumerate(records)]
    groups: dict[str, list[int]] = {}
    for index, (_, group_id) in enumerate(row_identities):
        groups.setdefault(group_id, []).append(index)
    if len(groups) < 2:
        raise ValueError("可切分的独立题目/来源分组不足 2 组")

    target = max(1, min(len(records) - 1, round(len(records) * ratio)))
    ordered = sorted(groups.items(), key=lambda item: hashlib.sha256(
        f"{SPLIT_ALGORITHM}:{seed}:{item[0]}".encode("utf-8")
    ).hexdigest())
    validation_groups: set[str] = set()
    validation_count = 0
    for group_id, indexes in ordered:
        if validation_count >= target:
            break
        # Never consume the final training group.
        if len(validation_groups) == len(groups) - 1:
            break
        validation_groups.add(group_id)
        validation_count += len(indexes)
    if not validation_groups or validation_count >= len(records):
        raise ValueError("按题目/来源分组切分后无法同时保留非空训练集和验证集")

    validation_indices = [index for index, (_, group) in enumerate(row_identities) if group in validation_groups]
    validation_index_set = set(validation_indices)
    train_indices = [index for index in range(len(records)) if index not in validation_index_set]
    train_ids = [row_identities[index][0] for index in train_indices]
    validation_ids = [row_identities[index][0] for index in validation_indices]
    # Preserve order and duplicate records; identity commits to indexed members, not just a set of hashes.
    train_members_sha256 = sha256_json([{"index": index, "sha256": row_identities[index][0]} for index in train_indices])
    validation_members_sha256 = sha256_json([
        {"index": index, "sha256": row_identities[index][0]} for index in validation_indices
    ])
    membership_manifest = _membership_manifest_payload(
        algorithm=SPLIT_ALGORITHM, ratio=ratio, seed=seed,
        train_indices=train_indices, validation_indices=validation_indices,
        train_record_ids=train_ids, validation_record_ids=validation_ids,
    )
    return {
        "mode": "random_split", "algorithm": SPLIT_ALGORITHM, "ratio": ratio, "seed": seed,
        "train_indices": train_indices, "validation_indices": validation_indices,
        "train_record_ids": train_ids, "validation_record_ids": validation_ids,
        "train_count": len(train_indices), "validation_count": len(validation_indices),
        "train_members_sha256": train_members_sha256,
        "validation_members_sha256": validation_members_sha256,
        "membership_manifest_sha256": sha256_json(membership_manifest),
        "split_identity_sha256": sha256_json({
            "algorithm": SPLIT_ALGORITHM, "ratio": ratio, "seed": seed,
            "train_members_sha256": train_members_sha256,
            "validation_members_sha256": validation_members_sha256,
        }),
    }


def _membership_manifest_payload(*, algorithm: str, ratio: float, seed: int,
                                 train_indices: list[int], validation_indices: list[int],
                                 train_record_ids: list[str], validation_record_ids: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "training-validation-members-v1",
        "algorithm": algorithm, "ratio": ratio, "seed": seed,
        "train_members": [{"source_index": index, "record_sha256": digest}
                           for index, digest in zip(train_indices, train_record_ids, strict=True)],
        "validation_members": [{"source_index": index, "record_sha256": digest}
                               for index, digest in zip(validation_indices, validation_record_ids, strict=True)],
    }


def record_ids(records: list[dict[str, Any]]) -> set[str]:
    return {identity for index, row in enumerate(records) for identity, _ in [_record_identity(row, index)]}


def write_jsonl_records(path: str | Path, records: list[dict[str, Any]]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return digest


def apply_native_partition(native_path: str | Path, plan: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Materialize immutable trainer-native train/validation files using a verified index plan."""
    source = Path(native_path)
    rows = read_jsonl_records(source)
    train_indices = plan.get("train_indices")
    validation_indices = plan.get("validation_indices")
    if not isinstance(train_indices, list) or not isinstance(validation_indices, list):
        raise ValueError("validation split plan has no member indexes")
    if sorted(train_indices + validation_indices) != list(range(len(rows))):
        raise ValueError("validation split plan does not cover the native dataset exactly once")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    train_path, validation_path = target / "train.jsonl", target / "validation.jsonl"
    train_sha = write_jsonl_records(train_path, [rows[index] for index in train_indices])
    validation_sha = write_jsonl_records(validation_path, [rows[index] for index in validation_indices])
    manifest = _membership_manifest_payload(
        algorithm=str(plan.get("algorithm")), ratio=float(plan.get("ratio")), seed=int(plan.get("seed")),
        train_indices=train_indices, validation_indices=validation_indices,
        train_record_ids=plan.get("train_record_ids", []),
        validation_record_ids=plan.get("validation_record_ids", []),
    )
    manifest_path = target / "split_manifest.json"
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with manifest_path.open("xb") as stream:
        stream.write(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    expected_manifest_sha = plan.get("membership_manifest_sha256")
    if expected_manifest_sha and manifest_sha != expected_manifest_sha:
        raise ValueError("materialized split membership manifest differs from preflight identity")
    return {"train_path": str(train_path), "validation_path": str(validation_path),
            "train_native_sha256": train_sha, "validation_native_sha256": validation_sha,
            "membership_manifest_path": str(manifest_path), "membership_manifest_sha256": manifest_sha,
            "train_count": len(train_indices), "validation_count": len(validation_indices)}


def inspect_registered_validation_dataset(workspace: str | Path, dataset_id: str, stage: str,
                                          allowed_roles: list[str] | None = None) -> dict[str, Any]:
    """Verify a registered, non-holdout validation set without accepting caller paths."""
    from ..data.cpt_corpus import validate_cpt_corpus_jsonl
    from ..data.importer import validate_canonical_jsonl
    from ..data.manifest import load_manifest, resolve_canonical_path, sha256_file, validate_manifest_evidence
    from .training_preflight import _find_manifest, _has_symlink_between

    root = Path(workspace).resolve(strict=True)
    manifest_path = _find_manifest(root, dataset_id)
    if _has_symlink_between(manifest_path, root):
        raise ValueError("validation manifest path cannot contain a symbolic link")
    manifest = load_manifest(manifest_path)
    if manifest.stage != ("preference" if stage in {"dpo", "simpo"} else stage):
        raise ValueError("独立验证集的数据阶段与训练阶段不匹配")
    if manifest.lineage is None or manifest.processing_report is None:
        raise ValueError("独立验证集缺少登记 lineage 或 processing report")
    errors = validate_manifest_evidence(manifest_path)
    if errors:
        raise ValueError("独立验证集登记证据无效：" + "; ".join(errors[:3]))
    canonical = resolve_canonical_path(manifest_path, manifest)
    digest = sha256_file(canonical)
    if digest != manifest.hashes.get("canonical_records_sha256") or digest != manifest.lineage.content_sha256:
        raise ValueError("独立验证集 canonical hash 与登记摘要不一致")
    if manifest.sample_count < 2:
        raise ValueError("独立验证集至少需要 2 条样本")
    metadata = manifest.metadata if isinstance(manifest.metadata, dict) else {}
    declared_role = metadata.get("dataset_role") or metadata.get("role") or metadata.get("split")
    role_allowlist = set(allowed_roles or ["validation", "diagnostic_dev"])
    role_allowlist.discard("final_holdout")
    if declared_role not in role_allowlist:
        raise ValueError("独立验证集角色不符合当前训练方案；final_holdout 不可用于训练期验证")
    rows = read_jsonl_records(canonical)
    for index, row in enumerate(rows, start=1):
        row_metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        row_role = row_metadata.get("data_role") or row_metadata.get("dataset_role") or row_metadata.get("role")
        if row_role == "final_holdout":
            raise ValueError(f"第 {index} 条记录属于 final_holdout，禁止用于训练期验证")
        if stage == "cpt" and row_role != "diagnostic_dev":
            raise ValueError(f"第 {index} 条 CPT 验证记录必须标记 diagnostic_dev")
    with canonical.open("rb") as stream:
        if stage == "cpt":
            validation = validate_cpt_corpus_jsonl(stream, expected_role="diagnostic_dev")
            valid, count = validation.valid, validation.record_count
        else:
            validation = validate_canonical_jsonl(stream, "preference" if stage in {"dpo", "simpo"} else stage)
            valid, count = validation.valid, validation.record_count
    if not valid or count != manifest.sample_count:
        raise ValueError("独立验证集 canonical schema、角色或样本数无效")
    return {"mode": "registered", "dataset_id": manifest.dataset_id, "stage": manifest.stage,
            "role": declared_role, "record_count": count, "manifest_sha256": sha256_file(manifest_path),
            "content_sha256": digest, "canonical_path": str(canonical)}

