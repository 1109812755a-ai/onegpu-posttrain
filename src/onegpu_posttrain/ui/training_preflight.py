"""Read-only validation of registered stage inputs before a future run confirmation."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
)

from ..data.cpt_corpus import validate_cpt_corpus_jsonl
from ..data.importer import validate_canonical_jsonl
from ..data.manifest import (
    DatasetManifest,
    load_manifest,
    resolve_canonical_path,
    sha256_file,
    validate_manifest_evidence,
)
from ..lineage import ModelLineage, sha256_directory, sha256_json
from ..subprocess_utils import child_process_env
from .launcher import _gpu_snapshot
from .model_playground_catalog import ModelPlaygroundCatalog
from .model_artifact_validation import validate_model_producing_run
from .stage_profiles import (
    StageProfile,
    resolve_training_memory_mode,
    resolve_training_parameters,
    stage_profile_catalog,
)
from .training_memory import inspect_training_memory_mode
from .training_validation import TrainingValidationConfig


def console_command_available(name: str) -> bool:
    """Check a console script using the same PATH passed to child processes."""
    try:
        return shutil.which(name, path=child_process_env().get("PATH")) is not None
    except TypeError:
        # Preserve compatibility with lightweight monkeypatched resolvers that
        # implement the historical one-argument shutil.which signature.
        return shutil.which(name) is not None


class StagePreflightRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Literal["cpt", "sft", "dpo", "simpo", "grpo"]
    profile_id: str = Field(min_length=1, max_length=100)
    dataset_ids: list[str] = Field(min_length=1, max_length=8)
    parent_model_id: str = Field(min_length=1, max_length=200)
    parameter_overrides: dict[str, StrictInt | StrictFloat] = Field(default_factory=dict, max_length=16)
    memory_mode_id: StrictStr | None = Field(default=None, max_length=64)
    validation: TrainingValidationConfig = Field(default_factory=TrainingValidationConfig)
    checkpoint_policy: Literal["final", "best"] = "final"

    @field_validator("dataset_ids")
    @classmethod
    def unique_dataset_ids(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 200 for value in values):
            raise ValueError("dataset IDs must be non-empty and bounded")
        if len(set(values)) != len(values):
            raise ValueError("dataset IDs must be unique")
        return values


def preflight_stage(workspace: str | Path, request: StagePreflightRequest, *, launcher_enabled: bool,
                    allow_cli_adapter: bool = False) -> dict:
    """Validate identities/evidence; missing declared quality/license is a warning, not a hard gate."""
    root = Path(workspace).resolve(strict=True)
    catalog = stage_profile_catalog(root)
    profile = next((item for item in catalog.profiles if item.profile_id == request.profile_id), None)
    if profile is None or profile.stage != request.stage:
        return _result(request, None, [], ["profile_id 不存在或与所选阶段不匹配"])

    checks: list[dict] = []
    blockers: list[str] = []
    warnings: list[str] = []
    training_parameters_valid = True
    try:
        effective_parameters = resolve_training_parameters(profile, request.parameter_overrides)
        memory_mode = resolve_training_memory_mode(profile, request.memory_mode_id)
        memory_compatibility = inspect_training_memory_mode(memory_mode.mode_id)
        if not memory_compatibility["ok"]:
            blockers.extend(memory_compatibility["blockers"])
    except (TypeError, ValueError) as exc:
        training_parameters_valid = False
        effective_parameters = {}
        memory_mode = None
        memory_compatibility = {"ok": False, "dependencies": [], "blockers": [str(exc)]}
        blockers.append(str(exc))
    checks.append(_check("training_parameters", training_parameters_valid,
                         "训练参数在 profile 允许范围内" if training_parameters_valid else "训练参数或显存模式无效"))
    if memory_mode is not None and memory_mode.mode_id == "qlora-4bit-nf4":
        checks.append(_check("training_memory:qlora_dependencies", memory_compatibility["ok"],
                             "QLoRA 所需 bitsandbytes 与 CUDA PyTorch 可用" if memory_compatibility["ok"]
                             else "QLoRA 依赖缺失或 CUDA 不可用"))
    recipe_path = _workspace_file(root, profile.approved_recipe)
    recipe_digest = sha256_file(recipe_path) if recipe_path else None
    config_sha256 = None
    if recipe_path:
        try:
            recipe_data = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
            recipe_training = recipe_data.get("training") if isinstance(recipe_data, dict) else None
            if not isinstance(recipe_training, dict):
                raise TypeError("批准 recipe 缺少 training 配置")
            resolved_training = {**recipe_training, **effective_parameters,
                                 **(memory_mode.settings if memory_mode is not None else {})}
            if request.stage == "sft" and request.checkpoint_policy == "best":
                resolved_training.update({
                    "load_best_model_at_end": True,
                    "metric_for_best_model": profile.checkpoint_metric,
                    "greater_is_better": profile.checkpoint_greater_is_better,
                })
            config_sha256 = sha256_json({"profile_id": profile.profile_id,
                                         "recipe_sha256": recipe_digest,
                                         "training": resolved_training})
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            resolved_training = {}
            blockers.append(f"批准 recipe 配置无法解析：{str(exc)[:160]}")
            checks.append(_check("training_config", False, "无法解析批准 recipe 的训练配置"))
        else:
            checks.append(_check("training_config", True, "默认配置与自定义参数已合并"))
    else:
        resolved_training = {}
        config_sha256 = None
        checks.append(_check("training_config", False, "批准 recipe 不可用"))
    checks.append(_check("approved_recipe", recipe_path is not None,
                         "批准配方可读取" if recipe_path else "批准配方缺失或越界"))
    if recipe_path is None:
        blockers.append("批准 recipe 不可用")
    if request.stage != "sft" and not allow_cli_adapter:
        blockers.append("请通过阶段 Launcher 的受控 adapter 执行该阶段训练")
    elif not profile.execution_supported and not allow_cli_adapter:
        blockers.append(profile.execution_blocker or "当前阶段执行 adapter 尚未接通")
    if not launcher_enabled:
        blockers.append("本地训练 Launcher 未启用")
    swift_available = console_command_available("swift")
    checks.append(_check("local_runtime:swift", swift_available,
                         "ms-swift 命令可用" if swift_available else "当前进程 PATH 未发现 swift"))
    if not swift_available and (profile.execution_supported or allow_cli_adapter):
        blockers.append("当前 Python 环境未发现 ms-swift 的 swift 命令")

    dataset_summaries = []
    for dataset_id in request.dataset_ids:
        try:
            manifest_path = _find_manifest(root, dataset_id)
            manifest = load_manifest(manifest_path)
            summary, dataset_blockers, dataset_warnings = _inspect_dataset(root, manifest_path, manifest, profile)
            if request.stage == "grpo" and not dataset_blockers:
                contract_summary, contract_blockers, contract_checks = _inspect_grpo_reward_contract(
                    root, manifest_path, manifest,
                )
                summary["reward_contract"] = contract_summary
                summary.setdefault("checks", []).extend(contract_checks)
                dataset_blockers.extend(contract_blockers)
            dataset_summaries.append(summary)
            blockers.extend(dataset_blockers)
            warnings.extend(dataset_warnings)
            checks.extend(summary.pop("checks"))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            blockers.append(f"数据集 {dataset_id} 不可用：{str(exc)[:240]}")
            checks.append(_check(f"dataset:{dataset_id}", False, "未能验证登记身份或证据"))

    try:
        parent = _find_model(root, request.parent_model_id)
        parent_summary = _inspect_parent(root, parent, profile)
        checks.extend(parent_summary.pop("checks"))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parent_summary = {"model_id": request.parent_model_id, "status": "blocked"}
        blockers.append(f"父模型不可用：{str(exc)[:240]}")
        checks.append(_check("parent_model", False, "父模型登记或 lineage 校验失败"))

    if profile.parent_model_rule == "registered_sft_parent":
        config_sha256 = sha256_json({"profile_id": profile.profile_id,
                                     "recipe_sha256": recipe_digest,
                                     "training": resolved_training,
                                     "parent_model": {"model_id": request.parent_model_id,
                                                      "sha256": parent_summary.get("digest")}})

    return _result(request, profile, checks, blockers, warnings, dataset_summaries, parent_summary,
                   recipe_digest, resolved_training, config_sha256, memory_mode)


def _result(request, profile, checks, blockers, warnings=None, datasets=None, parent=None,
            recipe_digest: str | None = None, resolved_training: dict | None = None,
            config_sha256: str | None = None, memory_mode=None) -> dict:
    blockers = list(dict.fromkeys(blockers))
    warnings = list(dict.fromkeys(warnings or []))
    return {
        "schema_version": "1.0", "read_only": True,
        "stage": request.stage, "profile_id": request.profile_id,
        "algorithm": profile.algorithm if profile else None,
        "status": "blocked" if blockers else "ready_for_confirmation",
        "checks": checks, "blockers": blockers, "warnings": warnings,
        "datasets": datasets or [], "parent_model": parent,
        "approved_recipe": ({"path": profile.approved_recipe, "sha256": recipe_digest} if profile else None),
        "training_parameters": {
            "defaults": {item.key: item.default for item in profile.training_parameters} if profile else {},
            "overrides": request.parameter_overrides,
            "effective": resolved_training or {},
            "config_sha256": config_sha256,
        },
        "training_memory_mode": ({"mode_id": memory_mode.mode_id, "label_zh": memory_mode.label_zh,
                                  "settings": memory_mode.settings} if memory_mode is not None else None),
        "hardware": _gpu_snapshot(),
        "execution_started": False,
        "notice": "预检只核对本地登记证据，不会启动训练、下载模型或改变数据。",
    }


def _check(check_id: str, passed: bool, detail: str) -> dict:
    return {"check_id": check_id, "status": "pass" if passed else "fail", "detail": detail}


def _inspect_grpo_reward_contract(root: Path, manifest_path: Path, manifest: DatasetManifest):
    """Require a current executable reward contract and matching reference metadata per prompt."""
    from ..data.targeted_grpo import (
        ExecutableRewardContract,
        evaluate_reward_contract,
        validate_reward_contract,
    )

    checks: list[dict] = []
    blockers: list[str] = []
    metadata = manifest.metadata if isinstance(manifest.metadata, dict) else {}
    try:
        contract = ExecutableRewardContract.create()
        validate_reward_contract(contract)
        declared = {
            "reward_contract_verified": True,
            "reward_contract_id": contract.contract_id,
            "reward_contract_protocol_sha256": contract.protocol_sha256,
            "reward_contract_implementation_sha256": contract.implementation_sha256,
        }
        mismatches = [key for key, expected in declared.items() if metadata.get(key) != expected]
        if mismatches:
            raise ValueError("manifest contract identity missing or mismatched: " + ", ".join(mismatches))
        plugin = root / "plugins/grpo_rewards.py"
        if _has_symlink_between(plugin, root) or not plugin.is_file() or not plugin.resolve().is_relative_to(root):
            raise ValueError("approved GRPO reward plugin is missing, linked, or outside workspace")
        plugin_text = plugin.read_text(encoding="utf-8")
        expected_registrations = ('orms["onegpu_accuracy"]', 'orms["onegpu_boxed_format"]')
        if any(registration not in plugin_text for registration in expected_registrations):
            raise ValueError("approved GRPO plugin does not expose the required reward functions")
        probe = evaluate_reward_contract(contract, r"\boxed{2}", "2")
        if set(probe) != set(contract.rewards) or any(not math.isfinite(value) for value in probe.values()):
            raise ValueError("reward smoke execution returned missing or non-finite reward values")
        checks.append(_check("grpo:reward_contract_executable", True,
                             "当前 reward implementation/protocol 摘要一致，allowlisted reward 可执行"))
    except (OSError, ValueError, TypeError) as exc:
        return {"status": "blocked", "reason": str(exc)[:300]}, [f"GRPO reward/reference 契约不可用：{str(exc)[:240]}"], [
            _check("grpo:reward_contract_executable", False, str(exc)[:300])]

    try:
        canonical = resolve_canonical_path(manifest_path, manifest)
        rows_checked = 0
        invalid_reference = 0
        invalid_contract = 0
        with canonical.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                row = json.loads(line)
                row_metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                prompt = row.get("prompt")
                prompt_ok = (isinstance(prompt, list) and bool(prompt)
                             and all(isinstance(message.get("content"), str) and message["content"].strip()
                                     for message in prompt if isinstance(message, dict)))
                identity_ok = (row_metadata.get("reward_contract_id") == contract.contract_id
                               and row_metadata.get("reward_contract_protocol_sha256") == contract.protocol_sha256
                               and row_metadata.get("reward_contract_implementation_sha256") == contract.implementation_sha256)
                reference_ok = (isinstance(row.get("answer"), str) and bool(row["answer"].strip())
                                and isinstance(row_metadata.get("problem_id"), str)
                                and bool(row_metadata["problem_id"].strip())
                                and row_metadata.get("data_role") == "train_seed"
                                and row_metadata.get("reference_source") == "registered_train_seed"
                                and prompt_ok)
                if not reference_ok:
                    invalid_reference += 1
                if not identity_ok:
                    invalid_contract += 1
                rows_checked += 1
        if rows_checked != manifest.sample_count or rows_checked == 0:
            raise ValueError(f"canonical prompt count {rows_checked} does not match manifest {manifest.sample_count}")
        if invalid_reference or invalid_contract:
            raise ValueError(f"{invalid_reference} prompt/reference rows and {invalid_contract} contract identities are invalid")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        reason = str(exc)[:300]
        return {"status": "blocked", "reason": reason}, [f"GRPO prompt/reference 记录不可用：{reason}"], checks + [
            _check("grpo:reference_rows", False, reason)]

    plugin_digest = sha256_file(plugin)
    contract_summary = {
        "status": "verified", "contract_id": contract.contract_id,
        "protocol_sha256": contract.protocol_sha256,
        "implementation_sha256": contract.implementation_sha256,
        "plugin_path": "plugins/grpo_rewards.py", "plugin_sha256": plugin_digest,
        "reward_names": list(contract.rewards), "reference_rows": rows_checked,
    }
    checks.append(_check("grpo:reference_rows", True,
                         f"{rows_checked} 条记录均有 train_seed reference、problem_id 与一致 contract 摘要"))
    return contract_summary, blockers, checks


def _dataset_roots(root: Path) -> list[Path]:
    return [root / "data" / "processed", root / "artifacts" / "synthetic"]


def stage_workbench_options(workspace: str | Path) -> dict[str, list[dict[str, object]]]:
    """Return bounded, non-authoritative selectors; preflight revalidates every ID."""
    root = Path(workspace).resolve(strict=True)
    datasets: list[dict[str, object]] = []
    seen_datasets: set[str] = set()
    for artifact_root in _dataset_roots(root):
        if artifact_root.is_symlink() or not artifact_root.is_dir():
            continue
        manifest_paths = sorted(artifact_root.glob("**/manifests/*.yaml"))
        manifest_paths.extend(sorted(artifact_root.glob("**/manifests/*.json")))
        for path in manifest_paths[:1000]:
            if _has_symlink_between(path, root) or not path.is_file():
                continue
            try:
                manifest = load_manifest(path)
                if manifest.dataset_id in seen_datasets:
                    continue
                seen_datasets.add(manifest.dataset_id)
                metadata = manifest.metadata if isinstance(manifest.metadata, dict) else {}
                role = metadata.get("dataset_role") or metadata.get("role") or metadata.get("split")
                eligible = bool(manifest.lineage and manifest.processing_report and metadata.get("canonical_path"))
                block_reason = None if eligible else "缺少登记 lineage、processing report 或 canonical_path"
                # GRPO compatibility is stronger than a matching stage label:
                # each prompt must carry a current, executable reward contract
                # and trusted reference evidence. Surface that result in the
                # selector instead of letting users discover it only at preflight.
                if eligible and manifest.stage == "grpo":
                    try:
                        _, contract_blockers, _ = _inspect_grpo_reward_contract(root, path, manifest)
                    except (OSError, ValueError, TypeError, KeyError) as exc:
                        contract_blockers = [str(exc)[:240]]
                    if contract_blockers:
                        eligible = False
                        block_reason = contract_blockers[0]
                datasets.append({"dataset_id": manifest.dataset_id, "display_name": metadata.get("display_name"),
                                 "stage": manifest.stage,
                                 "role": role, "record_count": manifest.sample_count,
                                 "eligible": eligible,
                                 "block_reason": block_reason})
            except (OSError, ValueError, TypeError):
                continue
    models: list[dict[str, object]] = []
    model_indexes: dict[str, int] = {}
    model_records: dict[str, ModelLineage] = {}
    verified_models = {item["model_id"]: item for item in ModelPlaygroundCatalog(root).list_models()}
    registry_root = root / "runs"
    runs_by_id: dict[str, dict[str, object]] = {}
    if not registry_root.is_symlink() and registry_root.is_dir():
        for path in list(registry_root.glob("**/lineage.json"))[:2000]:
            if _has_symlink_between(path, root) or not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and isinstance(payload.get("run_id"), str):
                    runs_by_id.setdefault(payload["run_id"], payload)
            except (OSError, ValueError, TypeError):
                continue
        for path in list(registry_root.glob("**/model_artifacts/*.json"))[:2000]:
            if _has_symlink_between(path, root) or not path.is_file():
                continue
            try:
                model = ModelLineage.model_validate_json(path.read_text(encoding="utf-8"))
                if model.model_id in model_indexes:
                    models[model_indexes[model.model_id]]["registry_count"] += 1
                    continue
                model_path = Path(model.path) if model.path else None
                if model_path is not None and not model_path.is_absolute():
                    model_path = root / model_path
                path_available = False
                registered = verified_models.get(model.model_id) if model.kind == "base" else None
                if model.kind == "base":
                    path_available = bool(registered and registered.get("available"))
                elif model_path is not None and not _has_symlink_between(model_path, root):
                    try:
                        path_available = model_path.resolve(strict=True).is_relative_to(root) and model_path.exists()
                    except (OSError, RuntimeError):
                        pass
                models.append({"model_id": model.model_id, "stage": model.stage, "kind": model.kind,
                               "produced_by_run_id": model.produced_by_run_id,
                               "source_run_id": None,
                               "digest": model.digest.model_dump(mode="json"),
                               "path_available": path_available,
                               "unavailable_reason": (registered.get("unavailable_reason") if model.kind == "base" and registered
                                                      else None if path_available else "模型路径未通过本地登记校验"),
                               "registry_count": 1})
                model_indexes[model.model_id] = len(models) - 1
                model_records[model.model_id] = model
            except (OSError, ValueError, TypeError):
                continue
        for item in models:
            record = model_records.get(str(item["model_id"]))
            if record is not None:
                item["source_run_id"] = _model_source_training_run(record, runs_by_id, models)
    return {"datasets": datasets, "models": models}


def _model_source_training_run(model: ModelLineage, runs_by_id: dict[str, dict[str, object]],
                               registered_models: list[dict[str, object]]) -> str | None:
    """Resolve a UI link to the training Run that originated a model, including one verified merge hop."""
    producer_id = model.produced_by_run_id
    producer = runs_by_id.get(producer_id or "")
    if not producer or producer.get("status") != "completed":
        return None
    if producer.get("run_type") == "train":
        return producer_id
    if producer.get("run_type") != "merge":
        return None
    inputs = producer.get("inputs")
    if not isinstance(inputs, list):
        return None
    parent_ids = set(model.parent_model_ids)
    sources: set[str] = set()
    for ref in inputs:
        if not isinstance(ref, dict) or ref.get("kind") != "model" or ref.get("id") not in parent_ids:
            continue
        ref_digest = ref.get("digest")
        digest_value = ref_digest.get("value") if isinstance(ref_digest, dict) else None
        for parent in registered_models:
            if (parent.get("model_id") != ref.get("id") or parent.get("produced_by_run_id") is None
                    or (parent.get("digest") or {}).get("value") != digest_value):
                continue
            train_run_id = parent.get("source_run_id") or parent.get("produced_by_run_id")
            train_run = runs_by_id.get(str(train_run_id or ""))
            if train_run and train_run.get("run_type") == "train" and train_run.get("status") == "completed":
                sources.add(str(train_run_id))
    return next(iter(sources)) if len(sources) == 1 else None


def _find_manifest(root: Path, dataset_id: str) -> Path:
    matches = []
    for artifact_root in _dataset_roots(root):
        if artifact_root.is_symlink() or not artifact_root.is_dir():
            continue
        manifest_paths = sorted(artifact_root.glob("**/manifests/*.yaml"))
        manifest_paths.extend(sorted(artifact_root.glob("**/manifests/*.json")))
        for path in manifest_paths[:1000]:
            if _has_symlink_between(path, root) or not path.is_file() or not path.resolve().is_relative_to(root):
                continue
            try:
                manifest = load_manifest(path)
            except (OSError, ValueError, TypeError):
                continue
            if manifest.dataset_id == dataset_id:
                matches.append(path.resolve())
    if len(matches) != 1:
        raise ValueError("dataset_id 必须唯一匹配一个本地登记 manifest")
    return matches[0]


def _inspect_dataset(root: Path, path: Path, manifest: DatasetManifest, profile: StageProfile):
    blockers: list[str] = []
    warnings: list[str] = []
    checks: list[dict] = []
    if manifest.lineage is None or manifest.processing_report is None:
        raise ValueError("manifest 缺少数据血缘或处理报告，不能视为已登记训练数据")
    if manifest.stage != profile.dataset_stage:
        raise ValueError(f"数据阶段为 {manifest.stage}，profile 要求 {profile.dataset_stage}")
    metadata = manifest.metadata if isinstance(manifest.metadata, dict) else {}
    role = metadata.get("dataset_role") or metadata.get("role") or metadata.get("split")

    evidence_errors = validate_manifest_evidence(path)
    if evidence_errors:
        raise ValueError("登记报告/血缘摘要不一致：" + "; ".join(evidence_errors[:3]))
    checks.append(_check(f"dataset:{manifest.dataset_id}:manifest_evidence", True, "manifest、processing report 与质量报告引用一致"))

    artifact_root = path.parent.parent.resolve(strict=True)
    if not artifact_root.is_relative_to(root):
        raise ValueError("dataset artifact root 越出 workspace")
    if _has_symlink_between(path.parent.parent, root):
        raise ValueError("dataset artifact 路径不能经过符号链接")
    if _has_symlink_between(path, root):
        raise ValueError("dataset manifest 路径不能经过符号链接")
    canonical = resolve_canonical_path(path, manifest)
    digest = sha256_file(canonical)
    expected_digest = manifest.hashes.get("canonical_records_sha256")
    if not expected_digest or digest != expected_digest or manifest.lineage.content_sha256 != digest:
        raise ValueError("canonical 内容摘要与 manifest/数据血缘不一致")
    checks.append(_check(f"dataset:{manifest.dataset_id}:content_hash", True, "canonical SHA-256 与 manifest/lineage 一致"))

    row_roles: set[str] = set()
    preference_evidence = {"pairs": 0, "ordered_with_evidence": 0}

    def capture_role(row: dict) -> None:
        row_metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        row_role = row_metadata.get("dataset_role") or row_metadata.get("role") or row_metadata.get("split")
        if isinstance(row_role, str) and row_role:
            row_roles.add(row_role)
        if profile.dataset_stage == "preference":
            preference_evidence["pairs"] += 1
            correctness_evidence = row_metadata.get("chosen_correct") is True and row_metadata.get("rejected_correct") is False
            declared_evidence = row_metadata.get("ordering_evidence_status") in {"verified", "human_verified"}
            if correctness_evidence or declared_evidence:
                preference_evidence["ordered_with_evidence"] += 1

    with canonical.open("rb") as stream:
        if profile.dataset_stage == "cpt":
            validation = validate_cpt_corpus_jsonl(stream, expected_role="train_seed")
            valid, count = validation.valid, validation.record_count
            detail = "CPT canonical schema 与 train_seed 角色有效"
        else:
            validation = validate_canonical_jsonl(stream, profile.dataset_stage, on_record=capture_role)
            valid, count = validation.valid, validation.record_count
            detail = f"{profile.dataset_stage} canonical schema 有效"
    if not valid or count != manifest.sample_count:
        raise ValueError(f"canonical schema/样本数无效（manifest={manifest.sample_count}, 文件={count}）")
    checks.append(_check(f"dataset:{manifest.dataset_id}:canonical_schema", True, detail))
    if profile.dataset_stage == "preference":
        evidence_status = "verified" if preference_evidence["ordered_with_evidence"] == count else "unverified"
        checks.append(_check(f"dataset:{manifest.dataset_id}:preference_pairs", True,
                             f"{count} 个 pair 均通过 chosen/rejected 非空且不相同校验"))
        checks.append(_check(f"dataset:{manifest.dataset_id}:ordering_evidence", True,
                             f"{preference_evidence['ordered_with_evidence']}/{count} 个 pair 有可识别排序依据"))
        if evidence_status != "verified":
            warnings.append(
                f"数据集 {manifest.dataset_id} 有 {count - preference_evidence['ordered_with_evidence']}/{count} 个 Preference pair 未提供可识别的 chosen/rejected 排序证据；可继续探索训练，但结论应视为未核验"
            )
    else:
        evidence_status = None
    disallowed_roles = (row_roles - set(profile.dataset_roles)) | (
        {role} - set(profile.dataset_roles) if role else set()
    )
    if disallowed_roles:
        raise ValueError(f"数据角色 {', '.join(sorted(disallowed_roles))} 不允许用于训练，允许角色：{', '.join(profile.dataset_roles)}")
    checks.append(_check(f"dataset:{manifest.dataset_id}:stage_role", True,
                         f"阶段 {manifest.stage}，角色 {role or sorted(row_roles) or profile.dataset_roles[0]}"))
    if not role and not row_roles:
        role = profile.dataset_roles[0]

    if manifest.quality_report is None:
        warnings.append(f"数据集 {manifest.dataset_id} 没有关联质量报告；不因数量或缺失报告阻断探索性预检")
        quality_status = "not_provided"
    else:
        quality_status = str(metadata.get("quality_status", "reported"))
        if quality_status.upper() not in {"PASS", "PASSED", "READY"}:
            warnings.append(f"数据集 {manifest.dataset_id} 质量状态为 {quality_status}；请审阅报告，不自动判定为正式验收")

    license_status = manifest.origin.license_status if manifest.origin else "not_provided"
    if license_status != "verified":
        warnings.append(f"数据集 {manifest.dataset_id} 许可状态为 {license_status}；需按来源声明自行确认使用权限")

    summary = {"dataset_id": manifest.dataset_id, "stage": manifest.stage, "role": role or profile.dataset_roles[0],
             "record_count": manifest.sample_count, "content_sha256": digest,
             "manifest_sha256": sha256_file(path), "quality_status": quality_status,
             "license_status": license_status, "status": "valid", "checks": checks}
    if manifest.composition is not None:
        summary["composition"] = {
            "protocol": manifest.composition.protocol,
            "strategy": manifest.composition.strategy,
            "config_sha256": manifest.composition.config_sha256,
            "parents": [parent.model_dump(mode="json") for parent in manifest.composition.parents],
            "record_provenance": manifest.lineage.record_provenance,
        }
        checks.append(_check(f"dataset:{manifest.dataset_id}:composition_lineage", True,
                             f"组合包含 {len(manifest.composition.parents)} 个父数据集，逐条 provenance 摘要有效"))
    if evidence_status is not None:
        summary["ordering_evidence_status"] = evidence_status
        summary["ordering_evidence_count"] = preference_evidence["ordered_with_evidence"]
    return summary, blockers, warnings


def _find_model(root: Path, model_id: str) -> tuple[Path, ModelLineage]:
    registry_root = root / "runs"
    if registry_root.is_symlink() or not registry_root.is_dir():
        raise ValueError("找不到本地 model artifact registry")
    matches = []
    for path in list(registry_root.glob("**/model_artifacts/*.json"))[:2000]:
        if _has_symlink_between(path, root) or not path.is_file() or not path.resolve().is_relative_to(root):
            continue
        try:
            record = ModelLineage.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if record.model_id == model_id:
            matches.append((path.resolve(), record))
    if len(matches) != 1:
        raise ValueError("parent_model_id 必须唯一匹配本地 model artifact")
    return matches[0]


def _inspect_parent(root: Path, item: tuple[Path, ModelLineage], profile: StageProfile) -> dict:
    registry_path, model = item
    if model.stage not in profile.parent_model_stages:
        raise ValueError(f"父模型阶段为 {model.stage}，profile 要求 {profile.parent_model_stages}")
    if profile.parent_model_rule == "registered_sft_merged":
        if model.kind != "merged":
            raise ValueError("DPO/SimPO/GRPO 必须使用登记且已合并的 SFT 模型，不能使用 adapter")
        if not model.produced_by_run_id:
            raise ValueError("SFT 父模型缺少 producing training run，无法证明同一模型身份")
    if profile.parent_model_rule == "registered_sft_parent":
        if model.stage == "base":
            if model.kind != "base" or model.produced_by_run_id:
                raise ValueError("Base 父模型必须是没有训练 Run 的已登记外部根节点")
            resolved = _resolve_registered_base_path(root, model)
            _validate_sft_model_config(resolved)
            return {"model_id": model.model_id, "stage": model.stage, "kind": model.kind,
                    "digest": model.digest.value, "status": "valid",
                    "checks": [_check("parent_model:registry", True, "已登记 Base 模型身份有效"),
                               _check("parent_model:weights", True, "本地 Transformers 权重和配置可用"),
                               _check("parent_model:digest", True, "Base 模型内容摘要与登记一致"),
                               _check("parent_model:lineage", True, "Base 模型是无 producing Run 的外部根节点")]}
    if not model.path:
        raise ValueError("父模型 artifact 没有本地路径")
    supplied = Path(model.path)
    candidate = supplied if supplied.is_absolute() else root / supplied
    if _has_symlink_between(candidate, root):
        raise ValueError("父模型路径不能是符号链接")
    resolved = candidate.resolve(strict=True)
    if profile.parent_model_rule != "registered_sft_parent":
        resolved.relative_to(root)
    if not (resolved.is_file() or resolved.is_dir()):
        raise ValueError("父模型路径不是文件或目录")
    if profile.parent_model_rule == "registered_sft_parent":
        _validate_sft_model_config(resolved)
    observed_digest = sha256_directory(resolved) if resolved.is_dir() else sha256_file(resolved)
    if model.digest.status != "available" or model.digest.value != observed_digest:
        raise ValueError("父模型内容摘要与 artifact 登记不一致")
    validate_model_producing_run(root, registry_path, model)
    lineage_detail = ("父模型 producing run lineage 与输出身份有效" if model.produced_by_run_id
                      else "Base 模型登记为外部根节点，无 producing run")
    return {"model_id": model.model_id, "stage": model.stage, "kind": model.kind,
            "digest": observed_digest, "status": "valid",
            "checks": [_check("parent_model:registry", True, "model artifact 唯一且阶段匹配"),
                       _check("parent_model:digest", True, "模型内容摘要与登记一致"),
                       _check("parent_model:lineage", True, lineage_detail)]}


def _validate_sft_model_config(path: Path) -> None:
    """Fail closed on malformed or unsupported local Transformers/ms-swift models."""
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ValueError("Base 模型缺少本地 Transformers config.json")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("Base 模型 config.json 无法解析") from exc
    if not isinstance(config, dict) or not isinstance(config.get("model_type"), str):
        raise ValueError("Base 模型 config.json 缺少有效 model_type")
    architectures = config.get("architectures")
    if architectures is not None and (not isinstance(architectures, list)
                                      or not all(isinstance(item, str) for item in architectures)):
        raise ValueError("Base 模型 architectures 字段格式无效")
    tokenizer_files = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "spiece.model",
                       "vocab.json", "vocab.txt")
    if not any((path / name).is_file() for name in tokenizer_files):
        raise ValueError("Base 模型缺少可识别的本地 tokenizer 文件")
    try:
        from swift.model.model_meta import get_model_info_meta
        model_info, model_meta = get_model_info_meta(str(path), use_hf=True, download_model=False)
    except Exception as exc:
        raise ValueError(f"当前 ms-swift 无法解析该本地模型架构：{type(exc).__name__}") from exc
    if not getattr(model_info, "model_type", None) or not getattr(model_meta, "model_type", None):
        raise ValueError("当前 ms-swift 未登记该模型架构的训练支持")


def resolve_registered_sft_parent_path(workspace: str | Path, model_id: str) -> tuple[Path, ModelLineage]:
    """Resolve the exact, verified Base/CPT parent selected for registered SFT."""
    root = Path(workspace).resolve(strict=True)
    registry_path, model = _find_model(root, model_id)
    if model.stage == "base":
        path = _resolve_registered_base_path(root, model)
        _validate_sft_model_config(path)
        return path, model
    if model.stage != "cpt" or model.kind != "merged":
        raise ValueError("SFT 父模型必须是已登记 Base 或合并后的 CPT 模型")
    _inspect_parent(root, (registry_path, model), next(
        profile for profile in stage_profile_catalog(root).profiles if profile.parent_model_rule == "registered_sft_parent"))
    candidate = Path(model.path or "")
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve(strict=True), model


def _resolve_registered_base_path(root: Path, model: ModelLineage) -> Path:
    entry = next((item for item in ModelPlaygroundCatalog(root).list_models()
                  if item.get("model_id") == model.model_id and item.get("kind") == "base"), None)
    if not entry or not entry.get("available"):
        reason = entry.get("unavailable_reason") if entry else "模型未出现在已核验的本地目录"
        raise ValueError(f"Base 模型权重不可用或不受支持：{reason}")
    path = Path(str(entry.get("_resolved_path") or ""))
    if not path.is_dir():
        raise ValueError("Base 模型权重目录不存在")
    actual = sha256_directory(path)
    if actual != model.digest.value or entry.get("digest") != model.digest.value:
        raise ValueError("Base 模型内容摘要与登记不一致")
    if model.model_id != f"model-{actual[:20]}":
        raise ValueError("Base 模型 ID 与其内容摘要不匹配，不能写入可追溯训练 lineage")
    return path


def _workspace_file(root: Path, relative: str) -> Path | None:
    path = root / relative
    if _has_symlink_between(path, root) or not path.is_file() or not path.resolve().is_relative_to(root):
        return None
    return path.resolve()


def _has_symlink_between(path: Path, root: Path) -> bool:
    """Reject links in every path component, not only the final file."""
    try:
        relative = path.absolute().relative_to(root.absolute())
    except ValueError:
        return True
    current = root
    if current.is_symlink():
        return True
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False
