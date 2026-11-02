"""Leakage-checked adapter for held-out DPO/SimPO preference evaluation."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..data.importer import validate_canonical_jsonl
from ..data.manifest import DatasetManifest, load_manifest, sha256_file, validate_manifest_evidence
from ..lineage import ModelLineage, RunRecord, sha256_directory
from .stage_profiles import stage_profile_catalog
from .training_preflight import _find_manifest, _find_model, _has_symlink_between

_RUN_ID = re.compile(r"^run-[a-z0-9][a-z0-9-]{0,119}$")
_PROFILE_ID = "preference-heldout-v1"


class PreferenceEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(min_length=1, max_length=100)
    model_id: str = Field(min_length=1, max_length=200)
    dataset_id: str = Field(min_length=1, max_length=200)


class PreferenceEvaluationAdapter:
    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve(strict=True)

    def preflight(self, request: PreferenceEvaluationRequest) -> dict[str, Any]:
        blockers: list[str] = []
        warnings: list[str] = []
        checks: list[dict[str, Any]] = []
        model_summary: dict[str, Any] = {"model_id": request.model_id}
        dataset_summary: dict[str, Any] = {"dataset_id": request.dataset_id}
        try:
            registry_path, model = _find_model(self.workspace, request.model_id)
            if model.kind != "merged" or model.stage not in {"dpo", "simpo"}:
                raise ValueError("held-out Preference profile 只接受已合并的 DPO/SimPO 模型")
            if not model.produced_by_run_id or not model.adapter_id:
                raise ValueError("DPO/SimPO merged model 缺少 merge Run 或 adapter lineage")
            model_summary.update({"stage": model.stage, "kind": model.kind, "digest": model.digest.value,
                                  "merge_run_id": model.produced_by_run_id})
            training_run, train_dataset = self._training_evidence(registry_path, model)
            model_summary["training_run_id"] = training_run.run_id
            checks.append(_check("model:registered_merged_dpo_simpo", True, "DPO/SimPO merged model 与成功训练/合并 lineage 有效"))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            blockers.append(f"模型 lineage 不可用：{str(exc)[:300]}")
            training_run = None
            train_dataset = None
            checks.append(_check("model:training_lineage", False, "无法确认目标模型对应的训练数据身份"))

        try:
            manifest_path = _find_manifest(self.workspace, request.dataset_id)
            manifest = load_manifest(manifest_path)
            if manifest.stage != "preference_eval":
                raise ValueError(f"评测数据阶段为 {manifest.stage}；必须选择 preference_eval holdout")
            evidence_errors = validate_manifest_evidence(manifest_path)
            if evidence_errors:
                raise ValueError("登记数据证据无效：" + "; ".join(evidence_errors[:3]))
            canonical = _canonical_path(self.workspace, manifest_path, manifest)
            if train_dataset is not None:
                train_manifest_path = _find_manifest(self.workspace, train_dataset.id)
                train_manifest = load_manifest(train_manifest_path)
                train_canonical = _canonical_path(self.workspace, train_manifest_path, train_manifest)
                if (train_dataset.id == manifest.dataset_id
                        or train_dataset.digest is None
                        or train_dataset.digest.value == manifest.hashes.get("canonical_records_sha256")):
                    raise ValueError("held-out Preference 数据与该模型训练 dataset ID/content hash 相同")
                train_problems = _problem_ids(train_canonical, stage="preference")
                eval_problems = _problem_ids(canonical, stage="preference", require_holdout=True)
                overlap = train_problems & eval_problems
                if overlap:
                    raise ValueError(f"held-out 与训练数据存在 {len(overlap)} 个相同 problem_id；拒绝泄漏评测")
                checks.append(_check("dataset:problem_id_disjoint", True,
                                     f"训练 {len(train_problems)} 个 / holdout {len(eval_problems)} 个 problem_id，无交集"))
            else:
                _problem_ids(canonical, stage="preference", require_holdout=True)
            dataset_summary.update({"stage": manifest.stage, "record_count": manifest.sample_count,
                                    "content_sha256": manifest.hashes.get("canonical_records_sha256"),
                                    "manifest_sha256": sha256_file(manifest_path), "role": "preference_eval"})
            checks.append(_check("dataset:registered_preference_eval", True,
                                 "manifest、质量/处理证据、Preference schema 与 evaluation_only 标记有效"))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            blockers.append(f"held-out Preference 数据不可用：{str(exc)[:300]}")
            checks.append(_check("dataset:heldout_role_and_evidence", False, "必须是登记的独立 preference_eval 数据集"))

        if request.profile_id != _PROFILE_ID:
            blockers.append("评测 profile_id 不是批准的 preference-heldout-v1")
        profile_available = any(
            profile.stage in {"dpo", "simpo"}
            and any(item.profile_id == request.profile_id for item in profile.evaluation_profiles)
            for profile in stage_profile_catalog(self.workspace).profiles
        )
        if not profile_available:
            blockers.append("模型阶段没有声明该 Preference held-out profile")
        return {
            "schema_version": "1.0", "read_only": True, "profile_id": request.profile_id,
            "protocol": "response-mean-logprob-v1", "status": "blocked" if blockers else "ready_for_cli",
            "model": model_summary, "training_dataset": _dataset_ref_summary(train_dataset),
            "evaluation_dataset": dataset_summary, "checks": checks,
            "blockers": list(dict.fromkeys(blockers)), "warnings": warnings, "execution_started": False,
            "shared_task_benchmark": "separate-evaluation-profile",
        }

    def build_command(self, request: PreferenceEvaluationRequest, *, run_id: str) -> list[str]:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("无效 evaluation run_id")
        summary = self.preflight(request)
        if summary["status"] != "ready_for_cli":
            raise ValueError("Preference held-out 评测阻断：" + "; ".join(summary["blockers"]))
        output = self.workspace / "runs" / "evaluations" / run_id
        if output.exists() or output.is_symlink():
            raise FileExistsError("评测 Run 输出已存在；历史记录不会覆盖")
        return [sys.executable, "-m", "onegpu_posttrain", "run-registered-preference-eval",
                "--workspace", str(self.workspace), "--profile-id", request.profile_id,
                "--model-id", request.model_id, "--dataset-id", request.dataset_id,
                "--run-id", run_id, "--output", output.relative_to(self.workspace).as_posix()]

    def resolve_inputs(self, request: PreferenceEvaluationRequest) -> dict[str, Any]:
        """Resolve only the registered model and canonical held-out dataset after preflight."""
        summary = self.preflight(request)
        if summary["status"] != "ready_for_cli":
            raise ValueError("Preference held-out 评测阻断：" + "; ".join(summary["blockers"]))
        model_registry, model = _find_model(self.workspace, request.model_id)
        manifest_path = _find_manifest(self.workspace, request.dataset_id)
        manifest = load_manifest(manifest_path)
        model_path = Path(model.path or "")
        if not model_path.is_absolute():
            model_path = self.workspace / model_path
        model_path = model_path.resolve(strict=True)
        model_path.relative_to(self.workspace)
        if _has_symlink_between(model_path, self.workspace):
            raise ValueError("registered model path contains a symbolic link")
        canonical = _canonical_path(self.workspace, manifest_path, manifest)
        if _has_symlink_between(canonical, self.workspace):
            raise ValueError("held-out canonical path contains a symbolic link")
        return {"model": model, "model_registry_path": model_registry,
                "model_path": model_path, "manifest": manifest,
                "manifest_path": manifest_path, "canonical_path": canonical,
                "preflight": summary}

    def _training_evidence(self, merged_registry_path: Path, model: ModelLineage) -> tuple[RunRecord, Any]:
        merge_path = merged_registry_path.parent.parent / "merge_runs" / str(model.produced_by_run_id) / "lineage.json"
        if _has_symlink_between(merge_path, self.workspace) or not merge_path.is_file():
            raise ValueError("merged model 对应的 merge Run lineage 缺失")
        merge_run = RunRecord.model_validate_json(merge_path.read_text(encoding="utf-8"))
        if (merge_run.run_id != model.produced_by_run_id or merge_run.run_type != "merge"
                or merge_run.status != "completed"
                or not any(item.id == model.model_id and item.kind == "model" for item in merge_run.outputs)):
            raise ValueError("merge Run 状态或 merged model 输出身份不匹配")
        merged_path = Path(model.path or "")
        if not merged_path.is_absolute():
            merged_path = self.workspace / merged_path
        if _has_symlink_between(merged_path, self.workspace):
            raise ValueError("merged model path contains a symbolic link")
        merged_path = merged_path.resolve(strict=True)
        merged_path.relative_to(self.workspace)
        observed_merged_digest = sha256_directory(merged_path) if merged_path.is_dir() else sha256_file(merged_path)
        if model.digest.status != "available" or model.digest.value != observed_merged_digest:
            raise ValueError("merged model 内容摘要与 model registry 不一致")
        adapter_item = next((item for item in merge_run.inputs if item.id == model.adapter_id and item.kind == "model"), None)
        if adapter_item is None:
            raise ValueError("merge Run 没有引用 model 注册的训练 adapter")
        adapter_paths = list((self.workspace / "runs").glob(f"**/model_artifacts/{model.adapter_id}.json"))
        adapter_paths = [path for path in adapter_paths if not _has_symlink_between(path, self.workspace)
                         and path.is_file() and path.resolve().is_relative_to(self.workspace)]
        if len(adapter_paths) != 1:
            raise ValueError("训练 adapter 注册身份不唯一")
        adapter = ModelLineage.model_validate_json(adapter_paths[0].read_text(encoding="utf-8"))
        if adapter.produced_by_run_id is None or adapter.stage != model.stage:
            raise ValueError("训练 adapter 缺少正确阶段的 producing training Run")
        adapter_path = Path(adapter.path or "")
        if not adapter_path.is_absolute():
            adapter_path = self.workspace / adapter_path
        if _has_symlink_between(adapter_path, self.workspace):
            raise ValueError("training adapter path contains a symbolic link")
        adapter_path = adapter_path.resolve(strict=True)
        adapter_path.relative_to(self.workspace)
        observed_adapter_digest = sha256_directory(adapter_path) if adapter_path.is_dir() else sha256_file(adapter_path)
        if adapter.digest.status != "available" or adapter.digest.value != observed_adapter_digest:
            raise ValueError("training adapter 内容摘要与 model registry 不一致")
        if adapter_item.digest is None or adapter_item.digest.value != observed_adapter_digest:
            raise ValueError("merge Run 的 adapter 输入摘要与已登记 adapter 不一致")
        training_root = adapter_paths[0].parent.parent
        train_path = training_root / "lineage.json"
        if _has_symlink_between(train_path, self.workspace) or not train_path.is_file():
            raise ValueError("producing training Run lineage 缺失")
        training_run = RunRecord.model_validate_json(train_path.read_text(encoding="utf-8"))
        if (training_run.run_id != adapter.produced_by_run_id or training_run.run_type != "train"
                or training_run.status != "completed"
                or not any(item.id == adapter.model_id and item.kind == "model" for item in training_run.outputs)):
            raise ValueError("producing training Run 状态或 adapter 输出身份不匹配")
        training_adapter_output = next(item for item in training_run.outputs
                                       if item.id == adapter.model_id and item.kind == "model")
        if (training_adapter_output.digest is None
                or training_adapter_output.digest.value != observed_adapter_digest):
            raise ValueError("training Run 的 adapter 输出摘要与已登记 adapter 不一致")
        datasets = [item for item in training_run.inputs if item.kind == "dataset" and item.role == "train"]
        if len(datasets) != 1:
            raise ValueError("目标模型的训练 Run 必须恰有一个可追溯的 Preference train dataset")
        train_manifest_path = _find_manifest(self.workspace, datasets[0].id)
        train_manifest = load_manifest(train_manifest_path)
        if train_manifest.stage != "preference":
            raise ValueError("模型训练 dataset 不是 Preference train 数据")
        if datasets[0].digest is None or datasets[0].digest.value != train_manifest.hashes.get("canonical_records_sha256"):
            raise ValueError("训练 Run dataset digest 与登记 manifest 不一致")
        return training_run, datasets[0]


def _canonical_path(root: Path, manifest_path: Path, manifest: DatasetManifest) -> Path:
    if manifest.processing_report is None:
        raise ValueError("manifest 缺少处理报告")
    artifact_root = manifest_path.parent.parent.resolve(strict=True)
    report_path = (manifest_path.parent / manifest.processing_report.path).resolve(strict=True)
    report_path.relative_to(artifact_root)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("build_id") != manifest.processing_report.report_id:
        raise ValueError("processing report identity 不匹配")
    output = report.get("outputs", {}).get(manifest.stage)
    if not isinstance(output, dict) or output.get("dataset_id") != manifest.dataset_id:
        raise ValueError("processing report 未登记此 stage/dataset_id")
    canonical = (artifact_root / str(output.get("path", ""))).resolve(strict=True)
    canonical.relative_to(artifact_root)
    if (sha256_file(canonical) != manifest.hashes.get("canonical_records_sha256")
            or manifest.lineage is None or manifest.lineage.content_sha256 != manifest.hashes.get("canonical_records_sha256")):
        raise ValueError("canonical 内容摘要与 manifest/data lineage 不一致")
    return canonical


def _problem_ids(path: Path, *, stage: str, require_holdout: bool = False) -> set[str]:
    problems: set[str] = set()
    def inspect(row: dict) -> None:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        problem_id = metadata.get("problem_id")
        if not isinstance(problem_id, str) or not problem_id:
            raise ValueError("Preference record 缺少 problem_id，无法证明 train/holdout 隔离")
        if require_holdout and metadata.get("evaluation_only") is not True:
            raise ValueError("preference_eval record 缺少 evaluation_only=true 标记")
        problems.add(problem_id)
    with path.open("rb") as stream:
        validation = validate_canonical_jsonl(stream, stage, on_record=inspect)
    if not validation.valid or validation.record_count == 0:
        raise ValueError("Preference pair schema 无效或 holdout 为空")
    return problems


def _dataset_ref_summary(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {"dataset_id": value.id, "content_sha256": value.digest.value if value.digest else None,
            "role": value.role}


def _check(check_id: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"check_id": check_id, "status": "pass" if passed else "fail", "detail": detail}
