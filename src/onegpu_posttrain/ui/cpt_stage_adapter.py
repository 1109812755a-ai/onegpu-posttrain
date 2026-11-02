"""Allowlisted CPT CLI adapter for a registered CPT training dataset."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..data.cpt_training import inspect_local_base_model, inspect_registered_cpt_dataset
from ..data.manifest import load_manifest, sha256_file
from .launcher import _gpu_snapshot
from ..subprocess_utils import console_command_available
from .stage_profiles import stage_profile_catalog
from .training_preflight import _find_manifest

_OUTPUT_ID = re.compile(r"^runs/cpt/cpt-smoke-v1/run-[a-z0-9][a-z0-9-]{0,119}$")


class CPTPreflightRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1, max_length=200)


class CPTPreflightCheck(BaseModel):
    """One independently actionable, read-only CPT readiness check."""

    model_config = ConfigDict(extra="forbid")

    check_id: str
    status: str = Field(pattern="^(pass|warning|fail)$")
    detail: str
    repair_hint: str | None = None


class CPTPreflightResponse(BaseModel):
    """Stable API contract for CPT readiness diagnostics; never starts work."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    read_only: bool
    stage: str
    algorithm: str
    profile_id: str
    status: str
    dataset: dict[str, Any]
    parent_model: dict[str, Any]
    approved_recipe: dict[str, Any]
    runtime: dict[str, Any]
    checks: list[CPTPreflightCheck]
    blockers: list[str]
    warnings: list[str]
    execution_started: bool


class CPTStageAdapter:
    """Resolve registered IDs and build only the approved existing CPT CLI command."""

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve(strict=True)

    def preflight(self, dataset_id: str) -> dict[str, Any]:
        if not isinstance(dataset_id, str) or not dataset_id.strip() or len(dataset_id) > 200:
            raise ValueError("dataset_id 无效")
        profile = next((item for item in stage_profile_catalog(self.workspace).profiles
                        if item.profile_id == "cpt-smoke-v1"), None)
        if profile is None:
            raise ValueError("CPT 阶段 profile 不可用")
        blockers: list[str] = []
        warnings: list[str] = []
        recipe = _safe_workspace_file(self.workspace, profile.approved_recipe)
        if recipe is None:
            raise ValueError("批准的 CPT recipe 不存在或越界")
        recipe_raw = yaml.safe_load(recipe.read_text(encoding="utf-8"))
        if (not isinstance(recipe_raw, dict) or recipe_raw.get("stage") != "peft_cpt"
                or recipe_raw.get("training", {}).get("tuner_type") != "lora"):
            raise ValueError("批准 CPT recipe 内容不符合 PEFT-CPT allowlist")
        model_config_path = recipe_raw.get("model_config")
        model_config_error = None
        model_ref = revision = None
        if not isinstance(model_config_path, str):
            model_config_error = "CPT recipe 缺少固定 model_config"
        else:
            model_config_file = _safe_workspace_file(self.workspace, model_config_path)
            if model_config_file is None:
                model_config_error = "Base model 配置不存在或越界"
            else:
                try:
                    parsed_model_config = yaml.safe_load(model_config_file.read_text(encoding="utf-8"))
                    parsed_model_config = (parsed_model_config.get("model", parsed_model_config)
                                           if isinstance(parsed_model_config, dict) else {})
                    model_ref = (parsed_model_config.get("repo") or parsed_model_config.get("model_id")
                                 or parsed_model_config.get("model"))
                    revision = parsed_model_config.get("revision")
                    if not isinstance(model_ref, str) or not model_ref or not isinstance(revision, str) or not revision:
                        model_config_error = "CPT Base model 必须配置固定模型 ID 与 pinned revision"
                except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
                    model_config_error = f"Base model 配置无法读取：{str(exc)[:300]}"

        manifest_path = _find_manifest(self.workspace, dataset_id)
        manifest = load_manifest(manifest_path)
        checks: list[dict[str, str]] = []
        dataset_evidence = None
        try:
            dataset_evidence = inspect_registered_cpt_dataset(manifest_path)
            checks.append(_check("dataset:registered_cpt", True, "CPT 登记证据、canonical 摘要和 train_seed 角色通过"))
        except (OSError, ValueError, TypeError) as exc:
            detail = str(exc)[:500]
            blockers.append(f"数据集/登记证据：{detail}")
            checks.append(_check("dataset:registered_cpt", False, detail,
                                 "重新构建 CPT 数据，或仅对符合条件的旧 FineMath 构建使用 repair-cpt-roles"))

        model_evidence = None
        try:
            if model_config_error:
                raise ValueError(model_config_error)
            model_evidence = inspect_local_base_model(model_ref, revision)
            checks.append(_check("parent_model:local_base", True, "固定 Base 模型本地配置和权重可用"))
        except (OSError, ValueError, TypeError) as exc:
            detail = str(exc)[:500]
            blockers.append(f"父模型本地文件：{detail}")
            model_label = f"{model_ref}@{revision}" if model_ref and revision else "recipe 固定的 Base 模型"
            checks.append(_check("parent_model:local_base", False, detail,
                                 f"检查 recipe 的 model_config，并在本机准备 {model_label} 的完整配置和权重；预检不会联网下载"))

        swift_available = console_command_available("swift")
        if not swift_available:
            detail = "当前项目 Python 对应的 scripts/bin 路径中未发现 swift 命令"
            blockers.append(detail)
            checks.append(_check("runtime:ms_swift", False, detail,
                                 "在启动 UI 的同一个项目虚拟环境安装项目 train 依赖，然后重启 UI 服务"))
        else:
            checks.append(_check("runtime:ms_swift", True, "项目 Python 对应环境中的 swift 命令可用"))

        gpu = _gpu_snapshot()
        if gpu.get("available"):
            checks.append(_check("hardware:gpu", True, str(gpu.get("name") or "已检测到 NVIDIA GPU")))
        else:
            warnings.append("GPU 状态暂不可确认；请检查本机 CUDA/NVIDIA 驱动后再启动训练")
            checks.append(_check("hardware:gpu", "warning", str(gpu.get("reason") or "GPU 状态未知"),
                                 "确认 GPU 驱动可用；此项为只读状态探测，未启动训练"))
        if not profile.execution_supported:
            detail = profile.execution_blocker or "CPT 阶段 Launcher adapter 尚未就绪"
            blockers.append(detail)
            checks.append(_check("launcher:execution_adapter", False, detail))
        else:
            checks.append(_check("launcher:execution_adapter", True, "CPT 受控执行 adapter 可用"))

        quality_status = manifest.metadata.get("quality_status", "not_provided")
        license_status = manifest.origin.license_status if manifest.origin else "not_provided"
        if manifest.quality_report is None:
            warnings.append("数据集没有质量报告；不因样本数量缺口阻断探索性 CPT")
        elif str(quality_status).upper() not in {"PASS", "PASSED", "READY"}:
            warnings.append(f"质量状态为 {quality_status}；训练前请查看数据质量报告")
        if license_status != "verified":
            warnings.append(f"许可状态为 {license_status}；请自行确认来源数据的使用权限")

        return {
            "schema_version": "1.0", "read_only": True, "stage": "cpt",
            "algorithm": "cpt", "profile_id": profile.profile_id,
            "status": "blocked" if blockers else "ready_for_cli",
            "dataset": {"dataset_id": manifest.dataset_id, "stage": manifest.stage,
                        "record_count": manifest.sample_count,
                        "manifest_path": manifest_path.relative_to(self.workspace).as_posix(),
                        "manifest_sha256": sha256_file(manifest_path),
                        "content_sha256": (dataset_evidence or {}).get("dataset_sha256"),
                        "quality_status": quality_status, "license_status": license_status,
                        "role": "train_seed",
                        "validation_status": "pass" if dataset_evidence else "blocked"},
            "parent_model": {"model_ref": model_ref, "revision": revision,
                             "config_sha256": (model_evidence or {}).get("config_sha256"),
                             "local_weights_available": model_evidence is not None,
                             "validation_status": "pass" if model_evidence else "blocked",
                             "downloads_performed": False},
            "approved_recipe": {"path": profile.approved_recipe, "sha256": sha256_file(recipe)},
            "runtime": {"swift_available": swift_available, "gpu": gpu},
            "checks": checks,
            "blockers": blockers, "warnings": warnings, "execution_started": False,
        }

    def build_command(self, dataset_id: str, output_location: str, *, config_path: str | None = None) -> list[str]:
        """Build a shell-free CLI argv; output is fixed under this workspace's CPT run root."""
        if not isinstance(output_location, str) or not _OUTPUT_ID.fullmatch(output_location):
            raise ValueError("CPT 输出位置必须是 runs/cpt/cpt-smoke-v1/<run_id>")
        output = self.workspace / output_location
        if output.exists() or output.is_symlink():
            raise FileExistsError("CPT 输出位置已存在；历史 Run 不覆盖")
        for parent in output.parents:
            if parent == self.workspace.parent:
                break
            if parent.is_symlink():
                raise ValueError("CPT 输出路径不能经过符号链接")
        summary = self.preflight(dataset_id)
        if summary["status"] != "ready_for_cli":
            raise ValueError("CPT 预检阻断：" + "; ".join(summary["blockers"]))
        config = config_path or summary["approved_recipe"]["path"]
        if config_path is not None:
            expected = f".onegpu-posttrain/launcher/configs/{output_location.rsplit('/', 1)[-1]}.yaml"
            if config_path != expected or _safe_workspace_file(self.workspace, config_path) is None:
                raise ValueError("CPT resolved recipe 必须属于当前已确认 Run")
        return [
            sys.executable, "-m", "onegpu_posttrain", "run-registered-cpt",
            "--workspace", str(self.workspace), "--manifest", summary["dataset"]["manifest_path"],
            "--config", config, "--model", summary["parent_model"]["model_ref"],
            "--output", output_location, "--run-id", output_location.rsplit("/", 1)[-1], "--merge",
        ]


def _safe_workspace_file(root: Path, relative: str) -> Path | None:
    path = root / relative
    try:
        resolved = path.resolve(strict=True)
        if path.is_symlink() or not resolved.is_relative_to(root) or not resolved.is_file():
            return None
        current = path
        while current != root:
            if current.is_symlink():
                return None
            current = current.parent
        return resolved
    except (OSError, RuntimeError):
        return None


def _check(check_id: str, passed: bool | str, detail: str, repair_hint: str | None = None) -> dict[str, str]:
    status = passed if isinstance(passed, str) else ("pass" if passed else "fail")
    check = {"check_id": check_id, "status": status, "detail": detail}
    if repair_hint:
        check["repair_hint"] = repair_hint
    return check
