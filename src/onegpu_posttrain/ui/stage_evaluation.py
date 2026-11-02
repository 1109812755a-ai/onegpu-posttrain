"""Allowlisted stage evaluation planning for registered merged models."""

from __future__ import annotations

import re
import shutil
import sys
import importlib.util
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..data.manifest import load_manifest, validate_manifest_evidence
from ..evaluation.choice_accuracy import choice_comparability_key
from ..evaluation.choice_judge import (
    CHOICE_JUDGE_PROMPT_VERSION,
    CHOICE_LLM_JUDGE_PROTOCOL,
    choice_judge_comparability_key,
    load_choice_judge_profile,
)
from ..lineage import ModelLineage, RunRecord, new_run_id, sha256_directory, sha256_file, sha256_json
from ..subprocess_utils import child_process_env
from .preference_evaluation_adapter import (
    PreferenceEvaluationAdapter,
    PreferenceEvaluationRequest,
)
from .stage_profiles import stage_profile_catalog
from .training_preflight import _find_manifest, _has_symlink_between

_RUN_ID = re.compile(r"^run-[a-z0-9][a-z0-9-]{0,119}$")
_SHARED_PROFILE = "shared-math-eval-v1"
_PROFILE_STAGE = {
    "cpt-math-v1": "cpt",
    "sft-math-eval-v1": "sft",
    "preference-heldout-v1": "preference",
    "grpo-math-eval-v1": "grpo",
    _SHARED_PROFILE: None,
    "choice-accuracy-v1": None,
}


class StageEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    training_run_id: str | None = Field(default=None, pattern=r"^run-[a-z0-9][a-z0-9-]{0,119}$")
    model_id: str | None = Field(default=None, min_length=1, max_length=200)
    profile_id: str = Field(min_length=1, max_length=100)
    dataset_id: str | None = Field(default=None, min_length=1, max_length=200)
    scoring_strategy: Literal["exact_match", "bailian_llm_judge"] = "exact_match"

    @model_validator(mode="after")
    def require_one_model_source(self):
        if (self.training_run_id is None) == (self.model_id is None):
            raise ValueError("必须且只能指定 training_run_id 或 model_id")
        if self.scoring_strategy != "exact_match" and self.profile_id != "choice-accuracy-v1":
            raise ValueError("评分策略仅适用于客观题评测")
        return self


class StageEvaluationAdapter:
    """Resolve only successful, registered stage models and fixed evaluator commands."""

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve(strict=True)

    def preflight(self, request: StageEvaluationRequest) -> dict[str, Any]:
        blockers: list[str] = []
        dataset_summary: dict[str, Any] | None = None
        try:
            model, training_run = (self._model_for_training_run(request.training_run_id)
                                   if request.training_run_id else self._registered_base_model(request.model_id or ""))
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            return {"ok": False, "blockers": [str(exc)], "warnings": []}

        model_stage = "base" if model.kind == "base" else (model.stage or "")
        profile = self._profile_for_model(model_stage, request.profile_id)
        judge_profile = None
        if profile is None:
            blockers.append("所选评测 profile 不适用于该模型阶段")
        if request.profile_id == "preference-heldout-v1":
            if not request.dataset_id:
                blockers.append("Preference held-out 评测需要选择已登记的 preference_eval 数据集")
            else:
                result = PreferenceEvaluationAdapter(self.workspace).preflight(
                    PreferenceEvaluationRequest(profile_id=request.profile_id, model_id=model.model_id,
                                                dataset_id=request.dataset_id)
                )
                blockers.extend(result.get("blockers", []))
                dataset_summary = result.get("evaluation_dataset")
        elif request.profile_id == "cpt-math-v1":
            if not request.dataset_id:
                blockers.append("CPT 领域 PPL 评测需要选择已登记的 cpt_holdout 数据集")
            else:
                try:
                    manifest_path = _find_manifest(self.workspace, request.dataset_id)
                    manifest = load_manifest(manifest_path)
                    role = ((manifest.metadata or {}).get("dataset_role") or (manifest.metadata or {}).get("role")
                            or ("cpt_holdout" if manifest.stage == "cpt_holdout" else None))
                    if manifest.stage not in {"cpt", "cpt_holdout"} or role not in {"cpt_holdout", "eval", "evaluation"}:
                        blockers.append("所选数据必须是 CPT 阶段的 holdout/evaluation 数据，不能是训练语料")
                    train_datasets = [item for item in (training_run.inputs if training_run else [])
                                      if item.kind == "dataset" and item.role == "train"]
                    observed_hash = manifest.hashes.get("canonical_records_sha256")
                    if any(item.id == manifest.dataset_id or (item.digest and item.digest.value == observed_hash)
                           for item in train_datasets):
                        blockers.append("CPT holdout 与该模型训练数据身份或内容摘要相同，拒绝泄漏评测")
                    evidence = validate_manifest_evidence(manifest_path)
                    if evidence:
                        blockers.append("CPT holdout 登记证据无效：" + "; ".join(evidence[:3]))
                    else:
                        dataset_summary = {"dataset_id": manifest.dataset_id,
                                           "manifest_sha256": sha256_file(manifest_path),
                                           "content_sha256": manifest.hashes.get("canonical_records_sha256"),
                                           "record_count": manifest.sample_count, "role": role}
                except (OSError, ValueError, TypeError) as exc:
                    blockers.append(f"CPT holdout 不可用：{str(exc)[:240]}")
        elif request.profile_id == "choice-accuracy-v1":
            if not request.dataset_id:
                blockers.append("客观题准确率评测需要选择已登记的 evaluation-only 数据集")
            else:
                try:
                    manifest_path = _find_manifest(self.workspace, request.dataset_id)
                    manifest = load_manifest(manifest_path)
                    metadata = manifest.metadata if isinstance(manifest.metadata, dict) else {}
                    role = metadata.get("dataset_role") or metadata.get("role")
                    if manifest.stage != "eval" or role != "evaluation_only":
                        blockers.append("客观题评测只接受 stage=eval 且 role=evaluation_only 的登记数据集")
                    train_datasets = [item for item in (training_run.inputs if training_run else [])
                                      if item.kind == "dataset" and item.role == "train"]
                    observed_hash = manifest.hashes.get("canonical_records_sha256")
                    if any(item.id == manifest.dataset_id or
                           (item.digest and item.digest.value == observed_hash)
                           for item in train_datasets):
                        blockers.append("评测集与该模型训练数据 ID 或 canonical 摘要相同，拒绝泄漏评测")
                    evidence = validate_manifest_evidence(manifest_path)
                    if evidence:
                        blockers.append("客观题评测集登记证据无效：" + "; ".join(evidence[:3]))
                    if role == "evaluation_only" and manifest.stage == "eval" and not evidence:
                        canonical_path = self._canonical_dataset_path(manifest_path, manifest)
                        if not observed_hash or sha256_file(canonical_path) != observed_hash:
                            blockers.append("客观题评测集 canonical 摘要与 manifest 不匹配")
                        else:
                            protocol_version = (CHOICE_LLM_JUDGE_PROTOCOL
                                                if request.scoring_strategy == "bailian_llm_judge"
                                                else "choice-accuracy-v1")
                            comparability_key = choice_comparability_key(observed_hash, manifest.sample_count)
                            if request.scoring_strategy == "bailian_llm_judge":
                                try:
                                    judge_profile = load_choice_judge_profile(self.workspace)
                                    from .sft_generation_setup import _credential_available
                                    if not _credential_available(self.workspace, judge_profile.api_key_env):
                                        blockers.append("缺少阿里百炼评判凭据；请配置本地 provider 凭据后重试")
                                    if importlib.util.find_spec("httpx") is None:
                                        blockers.append("当前 Python 环境未安装阿里百炼评判所需的 httpx 依赖")
                                    comparability_key = choice_judge_comparability_key(
                                        observed_hash, manifest.sample_count, judge_profile
                                    )
                                except (OSError, ValueError, TypeError, RuntimeError) as exc:
                                    blockers.append(f"阿里百炼评判配置不可用：{str(exc)[:200]}")
                            dataset_summary = {
                                "dataset_id": manifest.dataset_id,
                                "manifest_sha256": sha256_file(manifest_path),
                                "content_sha256": observed_hash,
                                "record_count": manifest.sample_count,
                                "role": role,
                                "protocol_version": protocol_version,
                                "comparability_key": comparability_key,
                            }
                except (OSError, ValueError, TypeError) as exc:
                    blockers.append(f"客观题评测集不可用：{str(exc)[:240]}")
        elif request.dataset_id:
            blockers.append("此评测 profile 不接受额外数据集")

        if request.profile_id not in {"preference-heldout-v1", "choice-accuracy-v1"} and not shutil.which(
            "evalscope", path=child_process_env().get("PATH")
        ):
            blockers.append("当前项目 Python 环境未安装 EvalScope")

        run_id = new_run_id("evaluate", label=f"{model.stage}-{request.profile_id}")
        judge_disclosure = None
        if request.profile_id == "choice-accuracy-v1" and request.scoring_strategy == "bailian_llm_judge":
            judge_disclosure = {
                "provider": "阿里云百炼",
                "provider_profile_id": judge_profile.profile_id if judge_profile else None,
                "provider_model": judge_profile.model if judge_profile else None,
                "protocol_version": CHOICE_LLM_JUDGE_PROTOCOL,
                "prompt_version": CHOICE_JUDGE_PROMPT_VERSION,
                "expected_request_count": (dataset_summary or {}).get("record_count", 0),
                "external_data_notice": "题目、选项、标准答案和被测模型原始回答将发送到阿里云百炼。",
                "billing_notice": "百炼 API 可能产生费用；此处显示预计请求数，不估算人民币金额。",
            }
        return {
            "ok": not blockers,
            "blockers": list(dict.fromkeys(blockers)),
            "warnings": ["评测会独占本地 GPU；请求样本数与实际评测分母可能不同。"],
            "training_run_id": request.training_run_id,
            "model_id": model.model_id,
            "profile_id": request.profile_id,
            "protocol_version": ((dataset_summary or {}).get("protocol_version")
                                 if request.profile_id == "choice-accuracy-v1" else profile.protocol if profile else None),
            "model": {"model_id": model.model_id, "stage": model_stage, "sha256": model.digest.value,
                      "path": (self._model_path(model).relative_to(self.workspace).as_posix()
                               if self._model_path(model).is_relative_to(self.workspace) else "本地外部 Base 模型")},
            "dataset_id": request.dataset_id,
            "dataset": dataset_summary,
            "evaluation_run_id": run_id,
            "output": f"runs/evaluations/{run_id}",
            "metrics": ([f"choice_llm_judge.{kind}.{metric}" for kind in
                         ("single_choice", "multiple_choice", "true_false", "overall")
                         for metric in ("accuracy", "decision_coverage")]
                        if judge_disclosure else profile.metrics if profile else []),
            "scoring_strategy": request.scoring_strategy,
            "external_judge": judge_disclosure,
        }

    def build_command(self, request: StageEvaluationRequest, run_id: str) -> list[str]:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("无效评测 run_id")
        summary = self.preflight(request)
        if not summary["ok"]:
            raise ValueError("阶段评测预检阻断：" + "; ".join(summary["blockers"]))
        model, training_run = (self._model_for_training_run(request.training_run_id)
                               if request.training_run_id else self._registered_base_model(request.model_id or ""))
        model_path = self._model_path(model)
        output = f"runs/evaluations/{run_id}"
        if request.profile_id == "preference-heldout-v1":
            return [sys.executable, "-m", "onegpu_posttrain", "run-registered-preference-eval",
                    "--workspace", str(self.workspace), "--profile-id", request.profile_id,
                    "--model-id", model.model_id, "--dataset-id", str(request.dataset_id),
                    "--run-id", run_id, "--output", output]
        if request.profile_id == "choice-accuracy-v1":
            command = [sys.executable, "-m", "onegpu_posttrain", "run-registered-choice-eval",
                    "--workspace", str(self.workspace),
                    *( ["--training-run-id", request.training_run_id] if request.training_run_id else
                       ["--model-id", request.model_id or ""]),
                    "--profile-id", request.profile_id, "--dataset-id", str(request.dataset_id),
                    "--run-id", run_id, "--output", output]
            if request.scoring_strategy == "bailian_llm_judge":
                command.extend(["--scoring-strategy", "bailian_llm_judge"])
            return command
        if request.profile_id == "grpo-math-eval-v1":
            command = [sys.executable, "-m", "onegpu_posttrain", "run-registered-grpo-eval",
                       "--workspace", str(self.workspace), "--model-id", model.model_id,
                       "--run-id", run_id, "--output", output]
            if training_run and training_run.targeted_plan_id is not None:
                command.extend(["--targeted-plan-id", training_run.targeted_plan_id,
                                "--targeted-plan-version", str(training_run.targeted_plan_version)])
            return command
        config = "recipes/mini/30-eval-math-v1.yaml" if request.profile_id == "sft-math-eval-v1" else "recipes/smoke/00-base-eval.yaml"
        command = [sys.executable, "-m", "onegpu_posttrain", "run-model-eval",
                   "--config", config, "--model", str(model_path), "--output", output,
                   "--run-id", run_id, "--run-label", request.profile_id,
                   "--stage", model.stage or "model", "--limit", "5", "--stream-output"]
        if model.kind == "base":
            command.extend(["--registered-model-id", model.model_id])
        if training_run and training_run.targeted_plan_id is not None:
            command.extend(["--targeted-plan-id", training_run.targeted_plan_id,
                            "--targeted-plan-version", str(training_run.targeted_plan_version)])
        if request.profile_id == "cpt-math-v1" and request.dataset_id:
            manifest_path = _find_manifest(self.workspace, request.dataset_id)
            manifest = load_manifest(manifest_path)
            metadata = manifest.metadata or {}
            canonical = metadata.get("canonical_path")
            if not isinstance(canonical, str) or not canonical:
                raise ValueError("CPT holdout manifest 缺少 canonical_path")
            data_root = manifest_path.parent.parent.resolve(strict=True)
            input_path = (data_root / canonical).resolve(strict=True)
            input_path.relative_to(data_root)
            if _has_symlink_between(input_path, data_root):
                raise ValueError("CPT holdout 路径不能经过符号链接")
            command.extend(["--domain-holdout", str(input_path), "--domain-limit", "20"])
        return command

    def _profile_for_model(self, stage: str, profile_id: str):
        expected_stage = _PROFILE_STAGE.get(profile_id, "unknown")
        if stage == "base" and profile_id in {_SHARED_PROFILE, "choice-accuracy-v1"}:
            catalog = stage_profile_catalog(self.workspace)
            return next((evaluation for item in catalog.profiles
                         for evaluation in item.evaluation_profiles if evaluation.profile_id == profile_id), None)
        stage_matches = expected_stage is None or expected_stage == stage or (
            expected_stage == "preference" and stage in {"dpo", "simpo"}
        )
        if not stage_matches:
            return None
        catalog = stage_profile_catalog(self.workspace)
        for stage_profile in catalog.profiles:
            if stage_profile.stage == stage:
                match = next((item for item in stage_profile.evaluation_profiles if item.profile_id == profile_id), None)
                if match:
                    return match
        return None

    def _model_for_training_run(self, run_id: str) -> tuple[ModelLineage, RunRecord]:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("无效训练 run_id")
        models_root = self.workspace / "runs"
        if models_root.is_symlink() or not models_root.is_dir():
            raise ValueError("找不到本地模型登记目录")
        candidates: list[ModelLineage] = []
        for path in list(models_root.glob("**/model_artifacts/*.json"))[:2000]:
            if _has_symlink_between(path, self.workspace) or not path.is_file():
                continue
            try:
                model = ModelLineage.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if model.kind == "merged" and model.produced_by_run_id:
                merge_id = model.produced_by_run_id
                merge_paths = list(models_root.glob(f"**/merge_runs/{merge_id}/lineage.json"))
                for merge_path in merge_paths[:2]:
                    if _has_symlink_between(merge_path, self.workspace) or not merge_path.is_file():
                        continue
                    try:
                        merge = RunRecord.model_validate_json(merge_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError, TypeError):
                        continue
                    if merge.status != "completed" or not any(ref.id == model.model_id for ref in merge.outputs):
                        continue
                    adapter_ids = {ref.id for ref in merge.inputs if ref.kind == "model"}
                    for adapter_id in adapter_ids:
                        adapter_paths = list(models_root.glob(f"**/model_artifacts/{adapter_id}.json"))[:2]
                        for adapter_path in adapter_paths:
                            try:
                                adapter = ModelLineage.model_validate_json(adapter_path.read_text(encoding="utf-8"))
                            except (OSError, ValueError, TypeError):
                                continue
                            if adapter.produced_by_run_id == run_id and adapter.stage == model.stage:
                                candidates.append(model)
                                break
        unique = {item.model_id: item for item in candidates}
        if len(unique) != 1:
            raise ValueError("训练 Run 必须唯一关联一个已登记 merged model；缺少模型或存在歧义")
        model = next(iter(unique.values()))
        model_path = self._model_path(model)
        digest = sha256_directory(model_path) if model_path.is_dir() else sha256_file(model_path)
        if model.digest.status != "available" or model.digest.value != digest:
            raise ValueError("模型文件摘要与登记不匹配")
        training_paths = list(models_root.glob("**/lineage.json"))[:5000]
        train_records = []
        for path in training_paths:
            if _has_symlink_between(path, self.workspace) or not path.is_file():
                continue
            try:
                record = RunRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if record.run_id == run_id and record.run_type == "train" and record.status == "completed":
                train_records.append(record)
        if len(train_records) != 1:
            raise ValueError("训练 Run lineage 缺失、不成功或身份不唯一")
        return model, train_records[0]

    def _registered_base_model(self, model_id: str) -> tuple[ModelLineage, None]:
        from .model_playground_catalog import ModelPlaygroundCatalog

        catalog = ModelPlaygroundCatalog(self.workspace)
        models_root = self.workspace / "runs"
        paths = [path for path in list(models_root.glob("**/model_artifacts/*.json"))[:2000]
                 if not _has_symlink_between(path, self.workspace) and path.is_file()]
        matches: list[ModelLineage] = []
        for path in paths:
            try:
                item = ModelLineage.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if item.model_id == model_id and item.kind == "base":
                matches.append(item)
        if len(matches) != 1:
            raise ValueError("base 模型登记缺失或存在重复身份")
        model = matches[0]
        if model.digest.status != "available":
            raise ValueError("base 模型摘要与已登记身份不匹配")
        resolved = catalog._allowed_local_path(model.path or "", "base")
        if resolved is None or not (resolved / "config.json").is_file():
            raise ValueError("base 模型路径缺失、越界或不可访问")
        actual = sha256_directory(resolved) if resolved.is_dir() else sha256_file(resolved)
        if actual != model.digest.value:
            raise ValueError("base 模型摘要与已登记身份不匹配")
        return model, None

    def _model_path(self, model: ModelLineage) -> Path:
        path = Path(model.path or "")
        if not path.is_absolute():
            path = self.workspace / path
        if model.kind == "base":
            from .model_playground_catalog import ModelPlaygroundCatalog
            resolved = ModelPlaygroundCatalog(self.workspace)._allowed_local_path(model.path or "", "base")
            if resolved is None:
                raise ValueError("base 模型路径或摘要不可用")
        else:
            if _has_symlink_between(path, self.workspace):
                raise ValueError("模型路径不能经过符号链接")
            resolved = path.resolve(strict=True)
            resolved.relative_to(self.workspace)
        return resolved

    def _canonical_dataset_path(self, manifest_path: Path, manifest) -> Path:
        metadata = manifest.metadata if isinstance(manifest.metadata, dict) else {}
        canonical = metadata.get("canonical_path")
        if not isinstance(canonical, str) or not canonical:
            raise ValueError("evaluation manifest 缺少 canonical_path")
        root = manifest_path.parent.parent.resolve(strict=True)
        path = (root / canonical).resolve(strict=True)
        path.relative_to(root)
        if _has_symlink_between(path, root):
            raise ValueError("evaluation 数据路径不能经过符号链接")
        return path
