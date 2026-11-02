"""Confirmed local execution for registered non-Mini stage training adapters."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, field_validator, model_validator

from ..lineage import new_run_id, sha256_file, sha256_json
from .cpt_stage_adapter import CPTStageAdapter
from .grpo_stage_adapter import GRPOStageAdapter, GRPOStageRequest
from .launcher import SFTLauncher
from .preflight_confirmations import StagePreflightConfirmations
from .preference_stage_adapter import PreferenceStageAdapter, PreferenceStageRequest
from .resolved_stage_recipe import materialize_stage_recipe
from .stage_profiles import resolve_training_parameters, stage_profile_catalog
from .resolved_stage_recipe import materialize_stage_recipe
from .training_preflight import StagePreflightRequest, preflight_stage
from .training_validation import (
    TrainingValidationConfig, inspect_registered_validation_dataset, plan_random_split,
    read_jsonl_records,
)


class RegisteredStageTrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Literal["cpt", "sft", "dpo", "simpo", "grpo"]
    profile_id: str = Field(min_length=1, max_length=100)
    dataset_ids: list[str] = Field(min_length=1, max_length=8)
    parent_model_id: str | None = Field(default=None, max_length=200)
    parameter_overrides: dict[str, StrictInt | StrictFloat] = Field(default_factory=dict, max_length=16)
    memory_mode_id: str | None = Field(default=None, max_length=64)
    validation: TrainingValidationConfig = Field(default_factory=TrainingValidationConfig)
    checkpoint_policy: Literal["final", "best"] = "final"

    @field_validator("dataset_ids")
    @classmethod
    def one_registered_input(cls, value: list[str]) -> list[str]:
        if len(value) != 1:
            raise ValueError("请先将多个数据集组合并登记为一个 dataset ID，再启动阶段训练")
        if not value[0].strip() or len(value[0]) > 200:
            raise ValueError("dataset_id 无效")
        return value

    @model_validator(mode="after")
    def require_registered_parent(self):
        if self.stage != "cpt" and not (self.parent_model_id or "").strip():
            raise ValueError("该阶段必须选择已登记父模型")
        return self


class RegisteredStageTraining:
    """Keep the stage CLI allowlist, preflight token, and shared process supervisor together."""

    def __init__(self, workspace: str | Path, launcher: SFTLauncher,
                 confirmations: StagePreflightConfirmations):
        self.workspace = Path(workspace).resolve(strict=True)
        self.launcher = launcher
        self.confirmations = confirmations

    def _preflight(self, request: RegisteredStageTrainingRequest) -> tuple[StagePreflightRequest, dict]:
        dataset_id = request.dataset_ids[0]
        if request.stage == "cpt":
            adapter = CPTStageAdapter(self.workspace)
            result = adapter.preflight(dataset_id)
            parent_identity = result.get("parent_model", {})
            marker = f"external-base:{parent_identity.get('model_ref', '')}@{parent_identity.get('revision', '')}"
            internal = self._identity_request(request, marker)
            dataset = result.get("dataset", {})
            profile = next(item for item in stage_profile_catalog(self.workspace).profiles
                           if item.profile_id == request.profile_id)
            try:
                effective = resolve_training_parameters(profile, request.parameter_overrides)
            except (TypeError, ValueError) as exc:
                summary = {**result, "status": "blocked", "datasets": [result.get("dataset", {})],
                           "parent_model": {"model_id": marker, "stage": "base", "kind": "external_base"},
                           "training_parameters": {"defaults": {}, "overrides": dict(request.parameter_overrides),
                                                   "effective": {}, "config_sha256": None},
                           "blockers": [*result.get("blockers", []), str(exc)],
                           "execution_started": False}
                internal = self._identity_request(request, marker)
                return internal, self._attach_validation(request, internal, summary)
            recipe_path = self.workspace / profile.approved_recipe
            recipe_data = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
            effective_training = {**recipe_data["training"], **effective}
            config_digest = sha256_json({"profile_id": profile.profile_id,
                                         "recipe_sha256": sha256_file(recipe_path),
                                         "training": effective_training})
            parent = {"model_id": marker, "stage": "base", "kind": "external_base",
                      "model_ref": parent_identity.get("model_ref"), "revision": parent_identity.get("revision"),
                      "digest": parent_identity.get("config_sha256"),
                      "path_available": parent_identity.get("local_weights_available", False)}
            summary = {**result, "status": "ready_for_confirmation" if result.get("status") == "ready_for_cli" else "blocked",
                       "datasets": [dataset], "parent_model": parent,
                       "approved_recipe": result.get("approved_recipe"),
                       "training_parameters": {
                           "defaults": {item.key: item.default for item in profile.training_parameters},
                           "overrides": dict(request.parameter_overrides), "effective": effective,
                           "config_sha256": config_digest,
                       },
                       "execution_started": False}
            return internal, self._attach_validation(request, internal, summary)

        internal = self._identity_request(request, request.parent_model_id or "")
        if request.stage == "sft":
            summary = preflight_stage(self.workspace, internal, launcher_enabled=True)
        elif request.stage in {"dpo", "simpo"}:
            summary = PreferenceStageAdapter(self.workspace).preflight(
                PreferenceStageRequest(profile_id=request.profile_id, dataset_id=dataset_id,
                                       parent_model_id=request.parent_model_id or "",
                                       parameter_overrides=request.parameter_overrides,
                                       memory_mode_id=request.memory_mode_id),
                launcher_enabled=True,
            )
        else:
            summary = GRPOStageAdapter(self.workspace).preflight(
                GRPOStageRequest(profile_id=request.profile_id, dataset_id=dataset_id,
                                 parent_model_id=request.parent_model_id or "",
                                 parameter_overrides=request.parameter_overrides,
                                 memory_mode_id=request.memory_mode_id),
            )
        # Adapters return ready_for_cli to distinguish CLI support from registered Web launch;
        # this service turns that into an expiring, one-time Web confirmation capability.
        if summary.get("status") == "ready_for_cli":
            summary["status"] = "ready_for_confirmation"
        return internal, self._attach_validation(request, internal, summary)

    @staticmethod
    def _identity_request(request: RegisteredStageTrainingRequest, parent_model_id: str) -> StagePreflightRequest:
        return StagePreflightRequest(stage=request.stage, profile_id=request.profile_id,
                                     dataset_ids=request.dataset_ids, parent_model_id=parent_model_id,
                                     parameter_overrides=request.parameter_overrides,
                                     memory_mode_id=request.memory_mode_id,
                                     validation=request.validation)

    def _attach_validation(self, request: RegisteredStageTrainingRequest,
                           internal: StagePreflightRequest, summary: dict) -> dict:
        """Resolve validation identities from registered evidence or deterministic local membership."""
        from .training_preflight import _find_manifest
        from ..data.manifest import load_manifest, resolve_canonical_path, sha256_file

        validation = request.validation
        profile = next(item for item in stage_profile_catalog(self.workspace).profiles
                       if item.profile_id == request.profile_id)
        result = dict(summary)
        validation_summary: dict = {"mode": validation.mode}
        blockers = list(result.get("blockers", []))
        if validation.mode != "none" and not profile.training_validation_supported:
            blockers.append(f"阶段方案 {profile.profile_id} 未声明支持训练期验证")
        if validation.mode == "registered":
            try:
                val = inspect_registered_validation_dataset(
                    self.workspace, validation.dataset_id or "", request.stage,
                    profile.validation_dataset_roles,
                )
                if val["record_count"] < profile.validation_min_samples:
                    raise ValueError(f"独立验证集至少需要 {profile.validation_min_samples} 条样本")
                train_path = _find_manifest(self.workspace, request.dataset_ids[0])
                train_manifest = load_manifest(train_path)
                train_canonical = resolve_canonical_path(train_path, train_manifest)
                train_rows = read_jsonl_records(train_canonical)
                val_rows = read_jsonl_records(val["canonical_path"])
                if sha256_file(train_canonical) == val["content_sha256"]:
                    raise ValueError("训练集与独立验证集 canonical 文件完全相同")
                train_members = {str(row.get("id")) for row in train_rows if row.get("id") is not None}
                val_members = {str(row.get("id")) for row in val_rows if row.get("id") is not None}
                if train_members & val_members:
                    raise ValueError("训练集与独立验证集存在重复 record ID")
                validation_summary = {key: value for key, value in val.items() if key != "canonical_path"}
            except (OSError, ValueError, TypeError, KeyError) as exc:
                blockers.append(f"独立验证集不可用：{str(exc)[:220]}")
                validation_summary = {"mode": "registered", "dataset_id": validation.dataset_id,
                                      "status": "blocked"}
        elif validation.mode == "random_split":
            try:
                train_path = _find_manifest(self.workspace, request.dataset_ids[0])
                train_manifest = load_manifest(train_path)
                canonical = resolve_canonical_path(train_path, train_manifest)
                rows = read_jsonl_records(canonical)
                plan = plan_random_split(rows, ratio=float(validation.ratio), seed=int(validation.seed))
                if len(plan["train_indices"]) < profile.validation_min_samples or plan["validation_count"] < profile.validation_min_samples:
                    raise ValueError(f"训练集和验证集均至少需要 {profile.validation_min_samples} 条样本")
                validation_summary = {key: value for key, value in plan.items()
                                      if key not in {"train_indices", "validation_indices", "train_record_ids", "validation_record_ids"}}
                validation_summary.update({"source_dataset_id": train_manifest.dataset_id,
                                           "source_manifest_sha256": sha256_file(train_path),
                                           "source_content_sha256": sha256_file(canonical)})
            except (OSError, ValueError, TypeError, KeyError) as exc:
                blockers.append(f"随机切分无法执行：{str(exc)[:220]}")
                validation_summary = {"mode": "random_split", "status": "blocked"}
        validation_summary.setdefault("status", "ready" if not blockers else "blocked")
        if request.checkpoint_policy == "best":
            if validation.mode == "none" or validation_summary.get("status") != "ready":
                blockers.append("选择 best 必须启用有效的训练期验证数据。")
            elif not profile.training_validation_supported:
                blockers.append(f"阶段方案 {profile.profile_id} 不支持基于验证指标选择 checkpoint。")
            else:
                try:
                    recipe = yaml.safe_load((self.workspace / profile.approved_recipe).read_text(encoding="utf-8"))
                    training_config = recipe["training"]
                    strategy = training_config.get("eval_strategy", "epoch")
                    if strategy not in {"epoch", "steps"}:
                        raise ValueError("验证策略必须是 epoch 或 steps")
                    if strategy == "steps":
                        eval_steps = training_config.get("eval_steps", 10)
                        save_steps = training_config.get("save_steps", 10)
                        if (isinstance(eval_steps, bool) or not isinstance(eval_steps, (int, float)) or eval_steps <= 0
                                or isinstance(save_steps, bool) or not isinstance(save_steps, (int, float))
                                or save_steps <= 0 or save_steps % eval_steps):
                            raise ValueError("save_steps 必须是 eval_steps 的正整数倍")
                except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
                    blockers.append(f"best 检查点验证/保存频率不可用：{str(exc)[:180]}")
        checkpoint_selection = {
            "policy": request.checkpoint_policy,
            "metric": profile.checkpoint_metric if request.checkpoint_policy == "best" else None,
            "greater_is_better": profile.checkpoint_greater_is_better if request.checkpoint_policy == "best" else None,
        }
        training = dict(result.get("training_parameters") or {})
        old_config_hash = training.get("config_sha256")
        training["validation"] = validation_summary
        training["checkpoint_selection"] = checkpoint_selection
        if old_config_hash:
            training["config_sha256"] = sha256_json({"training_config_sha256": old_config_hash,
                                                       "validation": validation_summary,
                                                       "checkpoint_selection": checkpoint_selection})
        result["training_parameters"] = training
        result["validation"] = validation_summary
        result["blockers"] = list(dict.fromkeys(blockers))
        if result["blockers"]:
            result["status"] = "blocked"
        return result

    def preflight(self, request: RegisteredStageTrainingRequest) -> dict:
        _, summary = self._preflight(request)
        return summary

    def confirm(self, request: RegisteredStageTrainingRequest) -> dict:
        internal, summary = self._preflight(request)
        if summary.get("status") != "ready_for_confirmation":
            return summary
        return self.confirmations.issue(internal, summary,
                                        run_id=new_run_id("train", label=f"{request.stage}-{request.profile_id}"))

    def start(self, request: RegisteredStageTrainingRequest, token: str,
              acknowledged_dataset_id: str | None = None) -> dict:
        internal, current = self._preflight(request)
        if current.get("status") != "ready_for_confirmation":
            raise ValueError("当前阶段输入预检不通过，请修复阻断项后重新预检")
        confirmed = self.confirmations.consume(token, internal, current)
        dataset_id = request.dataset_ids[0]
        run_id = str(confirmed["run_id"])
        output_location = str(confirmed["output_location"])
        dataset_summary = (current.get("datasets") or [{}])[0]
        license_status = str(dataset_summary.get("license_status", "unknown"))
        if license_status != "verified" and acknowledged_dataset_id != dataset_id:
            raise ValueError("数据来源/许可未核验；必须确认使用并提交完全匹配的 dataset_id")
        if license_status == "verified" and acknowledged_dataset_id not in {None, dataset_id}:
            raise ValueError("许可确认的数据集 ID 与预检不匹配")

        config_path, training_parameters = materialize_stage_recipe(
            self.workspace, request.profile_id, run_id, request.parameter_overrides,
            validation=current.get("validation"), memory_mode_id=request.memory_mode_id,
            checkpoint_policy=request.checkpoint_policy,
            parent_model_id=(request.parent_model_id if request.stage == "sft" else None),
        )
        if request.stage == "sft":
            from ..adapters.registered_dataset import adapt_registered_dataset
            from ..data.manifest import load_manifest, resolve_canonical_path
            from .training_preflight import _find_manifest, resolve_registered_sft_parent_path
            from .training_validation import apply_native_partition, plan_random_split, read_jsonl_records

            model_path, model = resolve_registered_sft_parent_path(self.workspace, request.parent_model_id or "")
            adapter_root = self.workspace / ".onegpu-posttrain" / "launcher" / "adapted" / run_id
            for directory in (self.workspace / ".onegpu-posttrain",
                              self.workspace / ".onegpu-posttrain" / "launcher", adapter_root.parent):
                if directory.is_symlink():
                    raise ValueError("SFT 数据适配目录不能经过符号链接")
                directory.mkdir(exist_ok=True)
                if not directory.resolve(strict=True).is_relative_to(self.workspace):
                    raise ValueError("SFT 数据适配目录越过工作区")
            if adapter_root.exists() or adapter_root.is_symlink():
                raise ValueError("本次 SFT 数据适配目录已存在")
            manifest_path = _find_manifest(self.workspace, dataset_id)
            adapted = adapt_registered_dataset(manifest_path, adapter_root)
            native_path = adapter_root / adapted["native_path"]
            if not native_path.is_file() or native_path.is_symlink() or not native_path.resolve().is_relative_to(adapter_root.resolve()):
                raise ValueError("SFT 数据适配结果缺失或越界")
            validation_summary = current.get("validation") or {"mode": "none"}
            validation_mode = validation_summary.get("mode", "none")
            validation_native_path = None
            if validation_mode == "registered":
                validation_id = request.validation.dataset_id or ""
                validation_manifest = _find_manifest(self.workspace, validation_id)
                validation_adapted = adapt_registered_dataset(validation_manifest, adapter_root / "validation-adapter")
                validation_native_path = adapter_root / "validation-adapter" / validation_adapted["native_path"]
            elif validation_mode == "random_split":
                train_manifest = load_manifest(manifest_path)
                canonical_path = resolve_canonical_path(manifest_path, train_manifest)
                plan = plan_random_split(read_jsonl_records(canonical_path),
                                         ratio=float(request.validation.ratio), seed=int(request.validation.seed))
                if plan["split_identity_sha256"] != validation_summary.get("split_identity_sha256"):
                    raise ValueError("SFT 随机验证成员在确认后发生变化")
                split = apply_native_partition(native_path, plan, adapter_root / "validation-split")
                native_path = Path(split["train_path"])
                validation_native_path = Path(split["validation_path"])
                if split["membership_manifest_sha256"] != validation_summary.get("membership_manifest_sha256"):
                    raise ValueError("SFT 随机验证成员摘要在确认后发生变化")
            argv = [sys.executable, "-m", "onegpu_posttrain", "run-sft", "--config", config_path,
                    "--model", str(model_path), "--dataset", str(native_path),
                    "--output", output_location, "--run-id", run_id, "--merge"]
            if validation_native_path is not None:
                argv.extend(["--validation-dataset", str(validation_native_path)])
            parent_id = model.model_id
            training_parameters["parent_model_id"] = model.model_id
            training_parameters["parent_model_sha256"] = model.digest.value
        elif request.stage == "cpt":
            argv = CPTStageAdapter(self.workspace).build_command(dataset_id, output_location,
                                                                 config_path=config_path)
            parent_id = str((current.get("parent_model") or {}).get("model_id", "external-base"))
        elif request.stage in {"dpo", "simpo"}:
            argv = PreferenceStageAdapter(self.workspace).build_command(
                PreferenceStageRequest(profile_id=request.profile_id, dataset_id=dataset_id,
                                       parent_model_id=request.parent_model_id or ""), run_id=run_id,
                config_path=config_path,
            )
            parent_id = request.parent_model_id or ""
        else:
            argv = GRPOStageAdapter(self.workspace).build_command(
                GRPOStageRequest(profile_id=request.profile_id, dataset_id=dataset_id,
                                 parent_model_id=request.parent_model_id or ""), run_id=run_id,
                config_path=config_path,
            )
            parent_id = request.parent_model_id or ""
        return self.launcher.start_stage_cli(
            run_id=run_id, stage=request.stage, profile_id=request.profile_id,
            output_location=output_location, argv=argv, dataset_id=dataset_id,
            parent_model_id=parent_id, acknowledged_dataset_id=acknowledged_dataset_id,
            license_status=license_status, training_parameters=training_parameters,
            cleanup_paths=([str(Path(".onegpu-posttrain") / "launcher" / "adapted" / run_id)]
                           if request.stage == "sft" else None),
        )

