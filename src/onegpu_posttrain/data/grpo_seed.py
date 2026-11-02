"""Strict OpenR1 SFT-seed derivation for ordinary GRPO dataset preparation."""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..lineage import sha256_json
from .dataset_import import register_jsonl
from .dataset_mixture import _DATASET_ROOTS, _find_parent
from .grpo_prompt_core import build_grpo_prompt_record
from .manifest import load_manifest, resolve_canonical_path, validate_manifest_evidence
from .registry import DataSource
from .targeted_grpo import ExecutableRewardContract, validate_reward_contract

PROTOCOL = "grpo-openr1-seed-v1"
OPENR1_REPO = "open-r1/OpenR1-Math-220k"
MAX_SEED_RECORDS = 250_000
MAX_SEED_BYTES = 256 * 1024 * 1024


class GRPOSeedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    dataset_ids: list[str] = Field(min_length=1, max_length=8)
    target_count: int = Field(ge=1, le=250_000)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)

    @field_validator("dataset_ids")
    @classmethod
    def unique_dataset_ids(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 200 for value in values):
            raise ValueError("dataset ID must be non-empty and no longer than 200 characters")
        if len(values) != len(set(values)):
            raise ValueError("dataset IDs must be unique")
        return values


def _is_trusted_manifest(manifest) -> bool:
    revision = manifest.source.revision if manifest.source else None
    return (
        manifest.stage == "sft"
        and manifest.source is not None
        and manifest.source.repo == OPENR1_REPO
        and isinstance(revision, str)
        and len(revision) == 40
        and all(ch in "0123456789abcdef" for ch in revision.lower())
        and manifest.metadata.get("build_protocol") == "openr1-build-v1"
        and manifest.metadata.get("dataset_role", "train_seed") == "train_seed"
    )


def _read_seed_parent(workspace: Path, dataset_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], Counter[str]]:
    parent = _find_parent(workspace, dataset_id)
    manifest = parent.manifest
    if not _is_trusted_manifest(manifest):
        raise ValueError(f"数据集 {dataset_id} 来源不满足固定 OpenR1 build protocol / revision / train_seed 契约")
    if manifest.lineage is None or manifest.processing_report is None:
        raise ValueError(f"数据集 {dataset_id} 缺少 lineage 或 processing report")
    evidence_errors = validate_manifest_evidence(parent.manifest_path)
    if evidence_errors:
        raise ValueError(f"数据集 {dataset_id} 登记证据无效：{evidence_errors[0]}")
    processing_path = parent.manifest_path.parent / manifest.processing_report.path
    try:
        processing_report = json.loads(processing_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"数据集 {dataset_id} 的处理报告不可读取") from exc
    if processing_report.get("processing_protocol") != "openr1-build-v1":
        raise ValueError(f"数据集 {dataset_id} 的处理报告不是 OpenR1 build protocol")
    transform = manifest.lineage.transform or {}
    source_evidence = manifest.lineage.source_evidence or {}
    if (transform.get("name") != "build_openr1_artifacts"
            or transform.get("version") != "openr1-build-v1"
            or source_evidence.get("repo") != OPENR1_REPO
            or source_evidence.get("revision") != manifest.source.revision):
        raise ValueError(f"数据集 {dataset_id} 缺少与来源一致的 OpenR1 构建血缘")
    build_run_path = parent.manifest_path.parent.parent / "runs" / manifest.lineage.produced_by_run_id / "lineage.json"
    try:
        build_run = json.loads(build_run_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"数据集 {dataset_id} 的 OpenR1 构建 Run 不可验证") from exc
    source_refs = [ref for ref in build_run.get("inputs", [])
                   if ref.get("kind") == "file" and ref.get("role") == "source" and ref.get("id") == OPENR1_REPO]
    output_refs = [ref for ref in build_run.get("outputs", [])
                   if ref.get("kind") == "dataset" and ref.get("id") == dataset_id]
    if (build_run.get("run_type") != "data_build" or build_run.get("run_id") != manifest.lineage.produced_by_run_id
            or not source_refs or not output_refs):
        raise ValueError(f"数据集 {dataset_id} 的来源/构建 Run 血缘不匹配")
    if manifest.sample_count > MAX_SEED_RECORDS or parent.canonical_path.stat().st_size > MAX_SEED_BYTES:
        raise ValueError(f"数据集 {dataset_id} 超过 GRPO 派生读取上限")
    expected_digest = manifest.hashes.get("canonical_records_sha256")
    if not expected_digest or hashlib.sha256(parent.canonical_path.read_bytes()).hexdigest() != expected_digest:
        raise ValueError(f"数据集 {dataset_id} canonical SHA256 与登记摘要不符")
    rows = []
    rejected: Counter[str] = Counter()
    with parent.canonical_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                rejected["invalid_json"] += 1
                continue
            metadata = row.get("metadata") if isinstance(row, dict) else None
            if not isinstance(metadata, dict):
                rejected["missing_record_metadata"] += 1
                continue
            rid = metadata.get("source_record_id")
            answer = metadata.get("parsed_answer")
            prompt = row.get("raw_problem")
            if metadata.get("correctness_math_verify") is not True:
                rejected["upstream_correctness_evidence_missing"] += 1
                continue
            if not isinstance(rid, str) or not rid.startswith("openr1:"):
                rejected["trusted_source_record_id_missing"] += 1
                continue
            if not isinstance(prompt, str) or not prompt.strip():
                rejected["prompt_missing"] += 1
                continue
            if not isinstance(answer, (str, int, float)) or isinstance(answer, bool) or not str(answer).strip():
                rejected["parsed_reference_answer_missing"] += 1
                continue
            problem_id = metadata.get("problem_id")
            if not isinstance(problem_id, str) or not problem_id.strip():
                rejected["problem_id_missing"] += 1
                continue
            rows.append({"record_id": rid, "problem_id": problem_id, "prompt": prompt.strip(),
                         "answer": str(answer).strip(), "source_dataset_id": manifest.dataset_id,
                         "source_dataset_sha256": expected_digest})
    return ({"dataset_id": dataset_id, "content_sha256": expected_digest,
             "revision": manifest.source.revision}, rows, rejected)


def preflight_grpo_seed(workspace: str | Path, request: GRPOSeedRequest) -> dict[str, Any]:
    root = Path(workspace).resolve(strict=True)
    contract = ExecutableRewardContract.create()
    validate_reward_contract(contract)
    parents = []
    candidates = []
    rejected: Counter[str] = Counter()
    for dataset_id in request.dataset_ids:
        if not isinstance(dataset_id, str) or not dataset_id.strip() or len(dataset_id) > 200:
            raise ValueError("无效 dataset ID")
        parent, rows, parent_rejected = _read_seed_parent(root, dataset_id)
        parents.append(parent)
        candidates.extend(rows)
        rejected.update(parent_rejected)
    if len({item["record_id"] for item in candidates}) != len(candidates):
        raise ValueError("所选 seed 数据集间存在重复 OpenR1 source record ID；请先去重组合")
    usable = min(request.target_count, len(candidates))
    return {
        "protocol": PROTOCOL,
        "requested_count": request.target_count,
        "eligible_count": len(candidates),
        "available_count": usable,
        "shortfall": max(0, request.target_count - len(candidates)),
        "rejections": dict(rejected),
        "parents": parents,
        "reward_contract": contract.model_dump(mode="json"),
        "reference_evidence": "上游 OpenR1 manifest 固定来源与 revision，并逐条要求 correctness_math_verify=true；此标签由上游提供，本系统不声称独立重验。",
        "quality_status": "report_only",
        "limitations": ["上游正确性标签未经本系统独立验证", "许可状态沿用父数据登记；not_provided 或未核验需使用者自行确认", "尚无模型 rollout，因此 reward variance 未测量"],
    }


def list_grpo_seed_candidates(workspace: str | Path) -> list[dict[str, Any]]:
    root = Path(workspace).resolve(strict=True)
    found: dict[str, dict[str, Any]] = {}
    visited = 0
    for relative in _DATASET_ROOTS:
        data_root = root / relative
        if not data_root.is_dir() or data_root.is_symlink():
            continue
        for current, dirs, files in os.walk(data_root, followlinks=False):
            here = Path(current)
            dirs[:] = [name for name in dirs if not name.startswith(".") and not (here / name).is_symlink()]
            if here.name != "manifests":
                continue
            for name in files:
                if Path(name).suffix.lower() not in {".yaml", ".yml", ".json"}:
                    continue
                visited += 1
                if visited > 5000:
                    raise ValueError("本地 manifest 数量超过 GRPO seed 索引上限 5000")
                path = here / name
                if path.is_symlink():
                    continue
                try:
                    manifest = load_manifest(path)
                    if not _is_trusted_manifest(manifest):
                        continue
                    parent, rows, rejected = _read_seed_parent(root, manifest.dataset_id)
                except (OSError, ValueError, yaml.YAMLError):
                    continue
                found[manifest.dataset_id] = {
                    **parent, "display_name": manifest.metadata.get("display_name") or manifest.dataset_id,
                    "sample_count": manifest.sample_count, "eligible_count": len(rows),
                    "rejected_count": sum(rejected.values()),
                    "license_status": manifest.origin.license_status if manifest.origin else "not_provided",
                    "license_id": manifest.origin.license_id if manifest.origin else None,
                    "evidence_scope": "上游 OpenR1 标签；未在本系统独立重验",
                }
    return sorted(found.values(), key=lambda item: (item["display_name"], item["dataset_id"]))


def build_grpo_seed_dataset(workspace: str | Path, request: GRPOSeedRequest) -> dict[str, Any]:
    root = Path(workspace).resolve(strict=True)
    report = preflight_grpo_seed(root, request)
    if report["available_count"] != request.target_count:
        raise ValueError(f"可信 seed 仅 {report['eligible_count']} 条，少于目标 {request.target_count}；降低目标数或选择更多合规数据集")
    rows: list[dict[str, Any]] = []
    parent_by_id = {p["dataset_id"]: p for p in report["parents"]}
    seeds: list[dict[str, Any]] = []
    for dataset_id in request.dataset_ids:
        refreshed_parent, values, _ = _read_seed_parent(root, dataset_id)
        if refreshed_parent["content_sha256"] != parent_by_id[dataset_id]["content_sha256"]:
            raise ValueError(f"数据集 {dataset_id} 在预检后内容发生变化；请重新预检")
        seeds.extend(values)
    selected = sorted(seeds, key=lambda row: (sha256_json({"seed": request.seed, "id": row["record_id"]}), row["record_id"]))[:request.target_count]
    contract = ExecutableRewardContract.model_validate(report["reward_contract"])
    validate_reward_contract(contract)
    derivation_preflight = {
        "protocol": report["protocol"], "requested_count": report["requested_count"],
        "eligible_count": report["eligible_count"], "available_count": report["available_count"],
        "rejections": report["rejections"], "parents": report["parents"],
        "reward_contract": report["reward_contract"], "reference_evidence": report["reference_evidence"],
        "quality_status": report["quality_status"], "limitations": report["limitations"],
    }
    derivation_preflight_sha256 = sha256_json(derivation_preflight)
    for seed in selected:
        source = parent_by_id[seed["source_dataset_id"]]
        identity = sha256_json({"protocol": PROTOCOL, "seed": request.seed, "source_record_id": seed["record_id"], "prompt": seed["prompt"]})[:24]
        rows.append(build_grpo_prompt_record(
            record_id=f"grpo-seed-{identity}", prompt=seed["prompt"], answer=seed["answer"],
            problem_id=seed["problem_id"], parent_record_ids=[seed["record_id"]], contract=contract,
            # The training preflight accepts the registered train-seed
            # reference contract. The more specific upstream evidence scope
            # remains explicit in metadata below.
            reference_source="registered_train_seed",
            context={"source_record_id": seed["record_id"], "source_dataset_id": seed["source_dataset_id"],
                     "source_dataset_sha256": seed["source_dataset_sha256"],
                     "derivation_preflight_sha256": derivation_preflight_sha256,
                     "correctness_evidence": "upstream_openr1_math_verify",
                     "reference_verification_scope": "upstream_openr1_math_verify"},
        ))
    payload = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8") for row in rows)
    parent_refs = [{"dataset_id": item["dataset_id"], "content_sha256": item["content_sha256"]} for item in report["parents"]]
    source = DataSource(source_id="source-grpo-derived-" + sha256_json(parent_refs)[:20], kind="dataset_derived",
                        name="OpenR1 train seed 派生的 GRPO prompts",
                        uri="dataset://" + ",".join(request.dataset_ids), revision=None,
                        revision_status="unknown", license_status="not_provided", declared_by="adapter")
    registered = register_jsonl(
        BytesIO(payload), workspace=root, output_root=root / "data/processed/grpo-seed-derived",
        stage="grpo", origin=source,
        identity_material_sha256=sha256_json({"parents": parent_refs, "seed": request.seed, "target": request.target_count,
                                               "contract": contract.contract_id}),
        registration_metadata={"derivation_protocol": PROTOCOL, "parent_dataset_ids": request.dataset_ids,
                               "parent_dataset_refs": parent_refs, "reward_contract_verified": True,
                               "derivation_preflight": derivation_preflight,
                               "derivation_preflight_sha256": derivation_preflight_sha256,
                               "derivation_config_sha256": sha256_json({"protocol": PROTOCOL, "parents": parent_refs,
                                                                         "seed": request.seed, "target_count": request.target_count,
                                                                         "reward_contract_id": contract.contract_id}),
                               "derivation_seed": request.seed, "requested_count": request.target_count,
                               "selected_source_record_ids_sha256": sha256_json([row["record_id"] for row in selected]),
                               "reward_contract_id": contract.contract_id, "reward_contract_version": contract.version,
                               "reward_contract_protocol_sha256": contract.protocol_sha256,
                               "reward_contract_implementation_sha256": contract.implementation_sha256,
                               "reference_verification_scope": "upstream_openr1_math_verify",
                               "quality_status": "report_only", "usage": "exploratory"},
    )
    return {"dataset_id": registered.dataset_id, "record_count": registered.record_count,
            "canonical_sha256": registered.canonical_sha256, "reused": registered.reused,
            "parents": report["parents"], "reward_contract": report["reward_contract"],
            "derivation_preflight_sha256": derivation_preflight_sha256,
            "reference_evidence": report["reference_evidence"], "quality_status": "report_only",
            "limitations": report["limitations"], "registration_path": registered.path.relative_to(root).as_posix()}
