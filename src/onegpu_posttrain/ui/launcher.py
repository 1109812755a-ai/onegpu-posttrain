"""Narrow, opt-in local supervisor for the reviewed Mini SFT recipe."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..adapters.registered_dataset import adapt_registered_dataset
from ..data.manifest import load_manifest, validate_manifest_evidence
from ..lineage import (
    ModelLineage,
    new_run_id,
    sha256_directory,
    sha256_file,
    sha256_json,
)
from ..subprocess_utils import child_process_env, console_command_available
from .model_artifact_validation import validate_model_producing_run
from .gpu_lease import GpuLeaseBusy, GpuLeaseHandle, WorkspaceGpuLease
from .run_supervisor import LocalRunSupervisor
from .stage_profiles import resolve_training_memory_mode, resolve_training_parameters, stage_profile_catalog
from .training_memory import inspect_training_memory_mode
from .training_validation import (
    TrainingValidationConfig, apply_native_partition, inspect_registered_validation_dataset,
    plan_random_split, read_jsonl_records,
)

RECIPE = "recipes/mini/23-sft-1024-fallback.yaml"
EVAL_RECIPE = "recipes/mini/30-eval-math-v1.yaml"
EVAL_CONFIG = "configs/evaluation/math-v1-launcher-mini.yaml"
MODEL_ID = "model-93611b3dbc4e10161349"
DATASET_ID = "ds-sft-derived-1024-690ad3799b30c06d8d6f"
MAX_LOG_BYTES = 1024 * 1024
MAX_LOG_TAIL = 64 * 1024
TOKEN_TTL_SECONDS = 300
_RUN_ID_RE = re.compile(r"^run-[a-z0-9][a-z0-9-]{0,119}$")
WINDOWS_PATH_BUDGET = 260
WINDOWS_PATH_SAFETY_MARGIN = 20


def _sft_validation_summary(workspace: Path, inputs: dict[str, Any], config: TrainingValidationConfig) -> dict[str, Any]:
    """Resolve an SFT validation identity without mutating registered source data."""
    if config.mode == "none":
        return {"mode": "none", "status": "ready"}
    if config.mode == "registered":
        profile = next(item for item in stage_profile_catalog(workspace).profiles
                       if item.profile_id == "sft-mini-1024-v1")
        value = inspect_registered_validation_dataset(
            workspace, config.dataset_id or "", "sft", profile.validation_dataset_roles,
        )
        if value["record_count"] < profile.validation_min_samples:
            raise ValueError(f"SFT 独立验证集至少需要 {profile.validation_min_samples} 条样本")
        train_rows = read_jsonl_records(inputs["canonical_path"])
        from .training_preflight import _find_manifest
        from ..data.manifest import load_manifest, resolve_canonical_path
        val_manifest_path = _find_manifest(workspace, config.dataset_id or "")
        val_manifest = load_manifest(val_manifest_path)
        val_rows = read_jsonl_records(resolve_canonical_path(val_manifest_path, val_manifest))
        train_ids = {str(row.get("id")) for row in train_rows if row.get("id") is not None}
        val_ids = {str(row.get("id")) for row in val_rows if row.get("id") is not None}
        if inputs["canonical_sha256"] == value["content_sha256"] or train_ids & val_ids:
            raise ValueError("SFT 训练集与独立验证集存在完全相同内容或重复 record ID")
        return {key: item for key, item in value.items() if key != "canonical_path"} | {"status": "ready"}
    rows = read_jsonl_records(inputs["canonical_path"])
    plan = plan_random_split(rows, ratio=float(config.ratio), seed=int(config.seed))
    if min(plan["train_count"], plan["validation_count"]) < 2:
        raise ValueError("SFT 随机切分后训练集和验证集至少各需 2 条")
    return {key: item for key, item in plan.items()
            if key not in {"train_indices", "validation_indices", "train_record_ids", "validation_record_ids"}} | {
                "source_dataset_id": inputs["dataset_id"], "source_manifest_sha256": inputs["manifest_sha256"],
                "source_content_sha256": inputs["canonical_sha256"], "status": "ready"}


def _evaluation_path_budget(workspace: Path, eval_run_id: str, *, windows: bool | None = None) -> dict[str, Any]:
    """Estimate EvalScope's longest review-cache path without creating it."""
    use_windows_rules = os.name == "nt" if windows is None else windows
    review_leaf = "math_500.jsonl.rerun-" + ("f" * 32)
    expected = workspace / "runs" / "evaluations" / eval_run_id / "evalscope" / "reviews" / "M2-SFT" / review_leaf
    length = len(str(expected))
    safe_limit = WINDOWS_PATH_BUDGET - WINDOWS_PATH_SAFETY_MARGIN
    return {"supported": (not use_windows_rules or length <= safe_limit),
            "path_length": length, "safe_limit": safe_limit,
            "windows_rules": use_windows_rules}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _contained_file(root: Path, relative: str) -> Path:
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f"固定输入不可用或越界：{relative}")
    return path.resolve()


def _gpu_snapshot() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "status": "unknown", "reason": "nvidia-smi 不可用"}
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {"available": False, "status": "unknown", "reason": "无法读取 GPU 状态"}
        name, total, used = [part.strip() for part in result.stdout.splitlines()[0].split(",", 2)]
        total_mib, used_mib = int(total), int(used)
        return {"available": True, "status": "advisory", "name": name,
                "total_mib": total_mib, "used_mib": used_mib,
                "free_mib": max(0, total_mib - used_mib), "observed_at": _utc_now()}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {"available": False, "status": "unknown", "reason": "GPU 检查失败；该检查不影响只读预检"}


def _sft_runtime_available() -> bool:
    return console_command_available("swift")


class SFTLauncher:
    """One in-memory confirmation token and one supervised local run."""

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.state_root = self.workspace / ".onegpu-posttrain" / "launcher"
        self.records_root = self.state_root / "runs"
        self._lock = threading.RLock()
        self._state_write_lock = threading.Lock()
        self._confirmations: dict[str, dict[str, Any]] = {}
        self._process: subprocess.Popen[str] | None = None
        self._active_run_id: str | None = None
        self._active_parent_run_id: str | None = None
        self._monitor_thread: threading.Thread | None = None
        self._shutdown_run_ids: set[str] = set()
        self._stop = threading.Event()
        self._gpu_lease = WorkspaceGpuLease(self.workspace)
        self._gpu_lease_handle: GpuLeaseHandle | None = None
        self._mark_stale_runs_interrupted()
        self._run_supervisor: LocalRunSupervisor | None = None
        self._stage_cleanup_paths: dict[str, list[Path]] = {}

    def _claim_gpu(self, task_type: str, run_id: str) -> GpuLeaseHandle:
        handle = self._gpu_lease.acquire(task_type=task_type, task_id=run_id, run_id=run_id)
        self._gpu_lease_handle = handle
        return handle

    def _assert_gpu_available(self) -> None:
        state = self._gpu_lease.inspect()
        if state["status"] != "available":
            owner = state.get("owner") or {"status": state["status"], "reason": state.get("reason")}
            owner["stale"] = state.get("stale", False)
            raise GpuLeaseBusy(owner)

    def _release_gpu(self) -> None:
        with self._lock:
            handle, self._gpu_lease_handle = self._gpu_lease_handle, None
            if handle is not None:
                handle.release()

    def _safe_state_root(self) -> None:
        parent = self.workspace / ".onegpu-posttrain"
        if parent.is_symlink():
            raise ValueError("Launcher 状态目录不能经过符号链接")
        parent.mkdir(exist_ok=True)
        if not parent.resolve().is_relative_to(self.workspace):
            raise ValueError("Launcher 状态目录越过工作区边界")
        if self.state_root.is_symlink():
            raise ValueError("Launcher 状态目录不能经过符号链接")
        self.records_root.mkdir(parents=True, exist_ok=True)
        if not self.records_root.resolve().is_relative_to(self.workspace):
            raise ValueError("Launcher 运行记录目录越过工作区边界")

    def _state_path(self, run_id: str) -> Path:
        if not _RUN_ID_RE.fullmatch(run_id):
            raise ValueError("无效 run_id")
        self._safe_state_root()
        return self.records_root / f"{run_id}.json"

    def _write_state(self, state: dict[str, Any]) -> None:
        path = self._state_path(str(state["run_id"]))
        temp = path.with_name(f"{path.stem}.{secrets.token_hex(6)}.tmp")
        with self._state_write_lock:
            try:
                state["updated_at"] = _utc_now()
                temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
                for attempt in range(5):
                    try:
                        os.replace(temp, path)
                        break
                    except PermissionError:
                        if attempt == 4:
                            raise
                        time.sleep(0.02 * (attempt + 1))
            finally:
                temp.unlink(missing_ok=True)

    def _read_state(self, run_id: str) -> dict[str, Any] | None:
        path = self._state_path(run_id)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 128 * 1024:
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) and value.get("run_id") == run_id else None
        except (OSError, ValueError, UnicodeError):
            return None

    def _mark_stale_runs_interrupted(self) -> None:
        try:
            if self.records_root.is_symlink() or not self.records_root.is_dir():
                return
            for path in self.records_root.glob("*.json"):
                if path.is_symlink() or path.stat().st_size > 128 * 1024:
                    continue
                state = json.loads(path.read_text(encoding="utf-8"))
                if state.get("run_kind") == "evaluation" and self._recover_finished_evaluation(state):
                    self._write_state(state)
                    if state.get("parent_run_id"):
                        parent = self._read_state(state["parent_run_id"])
                        if parent:
                            parent.update(evaluation_status=state["status"],
                                          evaluation_run_id=state["run_id"],
                                          evaluation_detail=state.get("detail"),
                                          evaluation_lineage_path=state.get("evaluation_lineage_path"))
                            self._write_state(parent)
                    continue
                stale_active = state.get("status") in {"queued", "starting", "running"}
                stale_evaluation = (state.get("run_kind") == "evaluation"
                                    and state.get("status") == "interrupted"
                                    and (state.get("evaluation_status") == "running"
                                         or not state.get("completed_at")))
                if stale_active or stale_evaluation:
                    if stale_active:
                        state["status"] = "interrupted"
                    if state.get("run_kind") == "evaluation":
                        state["evaluation_status"] = "interrupted"
                    state["completed_at"] = state.get("completed_at") or _utc_now()
                    state["detail"] = state.get("detail") or "UI 服务重启前未确认任务完成；不会自动续训"
                    self._write_state(state)
                    if state.get("run_kind") == "evaluation" and state.get("parent_run_id"):
                        parent = self._read_state(state["parent_run_id"])
                        if parent and parent.get("evaluation_status") in {"running", "starting"}:
                            parent.update(evaluation_status="interrupted", evaluation_detail=state["detail"])
                            self._write_state(parent)
        except (OSError, ValueError, TypeError):
            return

    def _recover_finished_evaluation(self, state: dict[str, Any]) -> bool:
        """Recover a terminal CLI evaluation result if the UI supervisor was restarted."""
        run_id = state.get("run_id")
        relative_output = state.get("output")
        if not isinstance(run_id, str) or not isinstance(relative_output, str):
            return False
        unresolved_output = self.workspace / relative_output
        if unresolved_output.is_symlink():
            return False
        output = unresolved_output.resolve(strict=False)
        if not output.is_relative_to(self.workspace):
            return False
        result_path = output / "evaluation_run.json"
        lineage_path = output / "evaluation_lineage.json"
        try:
            if (result_path.is_symlink() or lineage_path.is_symlink()
                    or not result_path.is_file() or not lineage_path.is_file()):
                return False
            result = json.loads(result_path.read_text(encoding="utf-8"))
            lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            return False
        terminal_status = result.get("status")
        if (result.get("run_id") != run_id
                or result.get("run_type") != "evaluate"
                or terminal_status not in {"completed", "partial", "failed", "oom"}
                or lineage.get("produced_by_run_id") != run_id):
            return False
        state.update(
            status=terminal_status,
            evaluation_status=terminal_status,
            completed_at=result.get("completed_at") or state.get("completed_at") or _utc_now(),
            evaluation_lineage_path=lineage_path.relative_to(self.workspace).as_posix(),
            outputs=[p.relative_to(self.workspace).as_posix() for p in output.glob("*.json")
                     if p.is_file() and not p.is_symlink()],
            detail=(None if terminal_status in {"completed", "partial"}
                    else "评测失败；请查看评测日志与 evaluation_run.json"),
        )
        return True

    def capabilities(self) -> dict[str, Any]:
        recipe_path = self.workspace / RECIPE
        return {"enabled": True, "recipe_key": "mini-sft-1024", "recipe_ref": RECIPE,
                "available": recipe_path.is_file() and not recipe_path.is_symlink(),
                "stage": "sft", "training_enabled": True, "evaluation_enabled": True,
                "parent_model_id": MODEL_ID,
                "default_dataset_id": DATASET_ID, "datasets": self.registered_sft_datasets()}

    def _merged_model_for_run(self, run_id: str) -> dict[str, Any]:
        state = self._read_state(run_id)
        if not state or state.get("status") != "completed" or not self._has_cli_success(state):
            raise ValueError("只有已成功完成且 lineage 完整的训练 run 才能评测")
        output_unresolved = self.workspace / str(state.get("output", ""))
        if output_unresolved.is_symlink():
            raise ValueError("训练输出目录不能是符号链接")
        output = output_unresolved.resolve()
        if not output.is_relative_to(self.workspace):
            raise ValueError("训练输出目录越过工作区边界")
        merged = output / "merged" / "M2-SFT"
        if merged.is_symlink() or not merged.is_dir() or not merged.resolve().is_relative_to(output):
            raise ValueError("未找到已合并的 M2-SFT 模型")
        artifacts = output / "model_artifacts"
        matches = []
        for path in artifacts.glob("model-*.json") if artifacts.is_dir() and not artifacts.is_symlink() else []:
            if path.is_symlink() or path.stat().st_size > 64 * 1024:
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if record.get("kind") == "merged" and Path(str(record.get("path", ""))).resolve() == merged.resolve():
                matches.append(record)
        if len(matches) != 1:
            raise ValueError("M2-SFT 必须有且仅有一条模型登记记录")
        record = matches[0]
        digest = sha256_directory(merged)
        if record.get("digest", {}).get("status") != "available" or record.get("digest", {}).get("value") != digest:
            raise ValueError("M2-SFT 内容摘要与模型登记记录不匹配")
        expected_id = f"model-{digest[:20]}"
        if record.get("model_id") != expected_id:
            raise ValueError("M2-SFT model_id 与内容摘要不匹配")
        return {"model_id": expected_id, "path": merged, "sha256": digest,
                "artifact": record, "training_run_id": run_id,
                "merge_run_id": record.get("produced_by_run_id"),
                "targeted_plan_id": state.get("targeted_stage_plan_id"),
                "targeted_plan_version": state.get("targeted_stage_plan_version")}

    def evaluation_preflight(self, training_run_id: str) -> dict[str, Any]:
        try:
            if not _RUN_ID_RE.fullmatch(training_run_id):
                raise ValueError("无效训练 run_id")
            model = self._merged_model_for_run(training_run_id)
            recipe_path = _contained_file(self.workspace, EVAL_RECIPE)
            config_path = _contained_file(self.workspace, EVAL_CONFIG)
            recipe_raw, config_raw = recipe_path.read_bytes(), config_path.read_bytes()
            recipe, config = yaml.safe_load(recipe_raw), yaml.safe_load(config_raw)
            if (recipe.get("stage") != "sft_eval" or recipe.get("evaluation_config") != EVAL_CONFIG
                    or recipe.get("seed") != 42 or config.get("protocol_version") != "math-eval-v1"
                    or config.get("benchmarks") != ["gsm8k", "math_500"]
                    or config.get("benchmark_execution") != [
                        {"dataset": "gsm8k", "limit": 5},
                        {"dataset": "math_500", "limit": 1, "dataset_args": {
                            "subset_list": ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]}},
                    ]):
                raise ValueError("固定 math-eval-v1 配置与白名单不符")
            eval_env = child_process_env()
            if not shutil.which("evalscope", path=eval_env.get("PATH")):
                raise ValueError("当前项目 Python 环境未安装 EvalScope；预检不会联网或自动安装")
            eval_run_id = new_run_id("evaluate", label="launcher-mini-math-eval")
            path_budget = _evaluation_path_budget(self.workspace, eval_run_id)
            if not path_budget["supported"]:
                raise ValueError(f"当前工作区路径过长，评测 review 缓存预计 {path_budget['path_length']} 字符；安全上限为 {path_budget['safe_limit']}。请将仓库放到更短路径后重试。")
            eval_output = self.workspace / "runs" / "evaluations" / eval_run_id
            if eval_output.exists():
                raise ValueError("评测输出目录冲突，请重新预检")
            fingerprint = sha256_json({"training_run_id": training_run_id, "model_id": model["model_id"],
                                       "model_sha256": model["sha256"], "recipe_sha256": hashlib.sha256(recipe_raw).hexdigest(),
                                       "config_sha256": hashlib.sha256(config_raw).hexdigest(), "eval_run_id": eval_run_id,
                                       "targeted_plan_id": model.get("targeted_plan_id"),
                                       "targeted_plan_version": model.get("targeted_plan_version")})
            token = secrets.token_urlsafe(32)
            with self._lock:
                self._confirmations[token] = {"kind": "evaluation", "expires": time.time() + TOKEN_TTL_SECONDS,
                                              "fingerprint": fingerprint, "run_id": eval_run_id,
                                              "training_run_id": training_run_id,
                                              "targeted_plan_id": model.get("targeted_plan_id"),
                                              "targeted_plan_version": model.get("targeted_plan_version"),
                                              "output": eval_output.relative_to(self.workspace).as_posix()}
            return {"ok": True, "blockers": [], "warnings": ["这是探索性评测：GSM8K 取 5 条，MATH-500 五个难度各取 1 条；不代表正式 benchmark 结果。",
                          "评测会占用 GPU；本地数据/模型缓存缺失时，EvalScope 运行期间可能尝试下载；预检本身不会下载。"],
                    "confirmation_token": token, "expires_in_seconds": TOKEN_TTL_SECONDS,
                    "training_run_id": training_run_id, "evaluation_run_id": eval_run_id,
                    "protocol_version": "math-eval-v1", "benchmarks": [{"name": "GSM8K", "samples": 5}, {"name": "MATH-500", "samples": 5,
                        "sampling": "Level 1–5 each 1"}],
                    "model": {"model_id": model["model_id"], "sha256": model["sha256"],
                              "produced_by_run_id": model["merge_run_id"], "path": model["path"].relative_to(self.workspace).as_posix()},
                    "targeted_stage_plan": ({"plan_id": model["targeted_plan_id"],
                                             "plan_version": model["targeted_plan_version"]}
                                            if model.get("targeted_plan_id") else None),
                    "output": eval_output.relative_to(self.workspace).as_posix(),
                    "path_budget": path_budget,
                    "gpu": _gpu_snapshot(), "fingerprint": fingerprint}
        except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError, json.JSONDecodeError) as exc:
            return {"ok": False, "blockers": [str(exc)], "warnings": [], "gpu": _gpu_snapshot()}

    def start_evaluation(self, token: str) -> dict[str, Any]:
        if not isinstance(token, str) or len(token) > 128:
            raise ValueError("预检确认令牌无效")
        with self._lock:
            confirmation = self._confirmations.pop(token, None)
            if confirmation is None or confirmation.get("kind") != "evaluation" or confirmation["expires"] <= time.time():
                raise ValueError("评测预检已过期或已使用，请重新预检")
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError(f"已有任务正在运行：{self._active_run_id}")
            self._assert_gpu_available()
            model = self._merged_model_for_run(confirmation["training_run_id"])
            output = self.workspace / confirmation["output"]
            if output.exists() or not output.resolve().is_relative_to(self.workspace):
                raise ValueError("评测输出目录冲突或越界")
            output.mkdir(parents=True, exist_ok=False)
            run_id = confirmation["run_id"]
            state = {"run_id": run_id, "run_kind": "evaluation", "parent_run_id": confirmation["training_run_id"],
                     "status": "starting", "started_at": _utc_now(), "completed_at": None, "elapsed_seconds": 0,
                     "progress": None, "progress_status": "unknown", "log_path": (output / "launcher.log").relative_to(self.workspace).as_posix(),
                     "output": output.relative_to(self.workspace).as_posix(), "evaluation_status": "running",
                     "model_id": model["model_id"], "model_sha256": model["sha256"], "exit_code": None,
                     "targeted_stage_plan_id": confirmation.get("targeted_plan_id"),
                     "targeted_stage_plan_version": confirmation.get("targeted_plan_version"),
                     "evaluation_lineage_path": None, "detail": None}
            self._write_state(state)
            training = self._read_state(confirmation["training_run_id"])
            training.update(evaluation_status="running", evaluation_run_id=run_id)
            self._write_state(training)
            argv = [sys.executable, "-m", "onegpu_posttrain", "run-model-eval", "--config", EVAL_RECIPE,
                    "--output", str(output), "--model", str(model["path"]), "--run-id", run_id,
                    "--run-label", "launcher-mini-math-eval-v1", "--stage", "sft", "--limit", "5", "--stream-output"]
            if confirmation.get("targeted_plan_id") is not None:
                argv.extend(["--targeted-plan-id", confirmation["targeted_plan_id"],
                             "--targeted-plan-version", str(confirmation["targeted_plan_version"])])
            lease = self._claim_gpu("evaluation", run_id)
            try:
                with (output / "launcher.log").open("w", encoding="utf-8") as log_stream:
                    process = subprocess.Popen(argv, cwd=self.workspace, shell=False,
                                               stdout=log_stream, stderr=subprocess.STDOUT,
                                               **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}))
            except OSError as exc:
                self._release_gpu()
                state.update(status="failed", evaluation_status="failed", completed_at=_utc_now(), detail=f"评测子进程无法启动：{exc.strerror or type(exc).__name__}")
                self._write_state(state)
                training.update(evaluation_status="failed", evaluation_detail=state["detail"])
                self._write_state(training)
                return {"run_id": run_id, "status": "failed"}
            lease.set_owner_pid(getattr(process, "pid", os.getpid()))
            self._process, self._active_run_id, self._active_parent_run_id = process, run_id, confirmation["training_run_id"]
            state.update(status="running")
            self._write_state(state)
            thread = threading.Thread(target=self._monitor_evaluation, args=(process, state, output / "launcher.log"), daemon=True)
            self._monitor_thread = thread
            thread.start()
            return {"run_id": run_id, "status": "running", "output": state["output"]}

    def stage_evaluation_preflight(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Prepare a one-use confirmation for a registered stage model evaluation."""
        from .stage_evaluation import StageEvaluationAdapter, StageEvaluationRequest

        request = StageEvaluationRequest.model_validate(payload)
        summary = StageEvaluationAdapter(self.workspace).preflight(request)
        if not summary.get("ok"):
            return summary
        fingerprint = sha256_json({
            "training_run_id": request.training_run_id,
            "model_id": request.model_id,
            "profile_id": request.profile_id,
            "dataset_id": request.dataset_id,
            "scoring_strategy": request.scoring_strategy,
            "dataset": summary.get("dataset"),
            "model": summary["model"],
            "evaluation_run_id": summary["evaluation_run_id"],
        })
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._confirmations[token] = {
                "kind": "stage_evaluation", "expires": time.time() + TOKEN_TTL_SECONDS,
                "fingerprint": fingerprint, "run_id": summary["evaluation_run_id"],
                "training_run_id": request.training_run_id, "request": request.model_dump(mode="json"),
                "summary": summary,
            }
        return {**summary, "confirmation_token": token, "expires_in_seconds": TOKEN_TTL_SECONDS,
                "fingerprint": fingerprint}

    def start_stage_evaluation(self, token: str) -> dict[str, Any]:
        from .stage_evaluation import StageEvaluationAdapter, StageEvaluationRequest

        if not isinstance(token, str) or len(token) > 128:
            raise ValueError("预检确认令牌无效")
        with self._lock:
            confirmation = self._confirmations.pop(token, None)
            if (confirmation is None or confirmation.get("kind") != "stage_evaluation"
                    or confirmation["expires"] <= time.time()):
                raise ValueError("阶段评测预检已过期或已使用，请重新预检")
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError(f"已有本地 GPU 任务正在运行：{self._active_run_id}")
            self._assert_gpu_available()
            request = StageEvaluationRequest.model_validate(confirmation["request"])
            adapter = StageEvaluationAdapter(self.workspace)
            current = adapter.preflight(request)
            if not current.get("ok"):
                raise ValueError("阶段评测输入已变化：" + "; ".join(current.get("blockers", [])))
            observed_fingerprint = sha256_json({
                "training_run_id": request.training_run_id, "model_id": request.model_id, "profile_id": request.profile_id,
                "dataset_id": request.dataset_id, "scoring_strategy": request.scoring_strategy,
                "dataset": current.get("dataset"), "model": current["model"],
                "evaluation_run_id": confirmation["run_id"],
            })
            if observed_fingerprint != confirmation["fingerprint"]:
                raise ValueError("阶段评测模型/数据身份已变化，请重新预检")
            run_id = confirmation["run_id"]
            output = self.workspace / "runs" / "evaluations" / run_id
            if output.exists() or output.is_symlink() or not output.resolve().is_relative_to(self.workspace):
                raise ValueError("评测输出目录冲突或越界")
            command = adapter.build_command(request, run_id)
            # The evaluator owns its output directory and requires it not to exist.
            # Keep the launcher stream in its own namespace so supervision does not
            # pre-create the path passed to the CLI.
            log_dir = self.workspace / ".onegpu-posttrain" / "launcher" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            if log_dir.is_symlink() or not log_dir.resolve().is_relative_to(self.workspace):
                raise ValueError("评测日志目录越界或不可用")
            log_path = log_dir / f"{run_id}.eval.log"
            state = {
                "run_id": run_id, "run_kind": "evaluation", "stage": current["model"]["stage"],
                "profile_id": request.profile_id, "parent_run_id": request.training_run_id,
                "scoring_strategy": request.scoring_strategy,
                "evaluation_sample_count": (current.get("dataset") or {}).get("record_count"),
                "base_model_evaluation": request.model_id is not None,
                "status": "starting", "evaluation_status": "starting", "started_at": _utc_now(),
                "completed_at": None, "elapsed_seconds": 0, "progress": None,
                "progress_status": "unknown", "log_path": log_path.relative_to(self.workspace).as_posix(),
                "output": output.relative_to(self.workspace).as_posix(), "model_id": current["model"]["model_id"],
                "dataset_id": request.dataset_id, "evaluation_lineage_path": None, "detail": None,
            }
            self._write_state(state)
            parent = self._read_state(request.training_run_id) if request.training_run_id else None
            if parent:
                parent.update(evaluation_status="starting", evaluation_run_id=run_id,
                              evaluation_detail=None)
                self._write_state(parent)
            lease = self._claim_gpu("evaluation", run_id)
            try:
                with log_path.open("w", encoding="utf-8") as log_stream:
                    process = subprocess.Popen(
                        command, cwd=self.workspace, shell=False, stdout=log_stream,
                        stderr=subprocess.STDOUT,
                        **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                           else {"start_new_session": True}),
                    )
            except OSError as exc:
                self._release_gpu()
                state.update(status="failed", evaluation_status="failed", completed_at=_utc_now(),
                             detail=f"评测子进程无法启动：{exc.strerror or type(exc).__name__}")
                self._write_state(state)
                if parent:
                    parent.update(evaluation_status="failed", evaluation_run_id=run_id,
                                  evaluation_detail=state["detail"])
                    self._write_state(parent)
                return {"run_id": run_id, "status": "failed", "detail": state["detail"]}
            lease.set_owner_pid(getattr(process, "pid", os.getpid()))
            self._process, self._active_run_id, self._active_parent_run_id = process, run_id, request.training_run_id
            state.update(status="running", evaluation_status="running")
            self._write_state(state)
            if parent:
                parent.update(evaluation_status="running", evaluation_run_id=run_id,
                              evaluation_detail=None)
                self._write_state(parent)
            thread = threading.Thread(
                target=self._monitor_stage_evaluation, args=(process, state, log_path), daemon=True,
            )
            self._monitor_thread = thread
            thread.start()
            return {"run_id": run_id, "status": "running", "output": state["output"]}

    def _monitor_stage_evaluation(self, process: subprocess.Popen[str], state: dict[str, Any], log_path: Path) -> None:
        started = time.monotonic()
        try:
            code = process.wait()
            self._trim_evaluation_log(log_path)
            state["exit_code"] = code
            result_path = self.workspace / state["output"] / "evaluation_run.json"
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                cli_status = result.get("status")
            except (OSError, ValueError):
                cli_status = None
            state["status"] = cli_status if cli_status in {"completed", "partial", "failed"} else ("failed" if code else "completed")
            if state["status"] == "failed" and self._logs_say_oom(log_path):
                state["status"] = "oom"
            if state["run_id"] in self._shutdown_run_ids and state["status"] not in {"completed", "partial"}:
                state["status"] = "interrupted"
                state["detail"] = "UI 服务关闭时请求停止评测"
            state["evaluation_status"] = state["status"]
            lineage = self.workspace / state["output"] / "evaluation_lineage.json"
            state["evaluation_lineage_path"] = lineage.relative_to(self.workspace).as_posix() if lineage.is_file() else None
            state["outputs"] = [p.relative_to(self.workspace).as_posix() for p in (self.workspace / state["output"]).glob("*.json")
                                 if p.is_file() and not p.is_symlink()]
            state["detail"] = None if state["status"] in {"completed", "partial"} else "评测失败；请查看评测日志与 evaluation_run.json"
            self._write_state(state)
        except Exception as exc:  # noqa: BLE001 - preserve truthful terminal state
            state.update(status="failed", evaluation_status="failed", detail=f"评测监督器错误：{type(exc).__name__}")
            self._write_state(state)
        finally:
            state["completed_at"] = _utc_now()
            state["elapsed_seconds"] = int(time.monotonic() - started)
            self._write_state(state)
            parent = self._read_state(state.get("parent_run_id")) if state.get("parent_run_id") else None
            if parent:
                parent.update(evaluation_status=state["status"], evaluation_run_id=state["run_id"],
                              evaluation_detail=state.get("detail"),
                              evaluation_lineage_path=state.get("evaluation_lineage_path"))
                self._write_state(parent)
            with self._lock:
                self._process = None; self._active_run_id = None; self._active_parent_run_id = None; self._monitor_thread = None
            self._release_gpu()

    def _monitor_evaluation(self, process: subprocess.Popen[str], state: dict[str, Any], log_path: Path) -> None:
        started = time.monotonic()
        try:
            code = process.wait()
            self._trim_evaluation_log(log_path)
            state["exit_code"] = code
            eval_run_path = self.workspace / state["output"] / "evaluation_run.json"
            try:
                eval_run = json.loads(eval_run_path.read_text(encoding="utf-8"))
                cli_status = eval_run.get("status")
            except (OSError, ValueError):
                cli_status = None
            state["status"] = cli_status if cli_status in {"completed", "partial", "failed"} else ("failed" if code else "completed")
            if state["status"] == "failed" and self._logs_say_oom(log_path):
                state["status"] = "oom"
            if state["run_id"] in self._shutdown_run_ids and state["status"] not in {"completed", "partial"}:
                state["status"] = "interrupted"
                state["detail"] = "UI 服务关闭时请求停止评测"
            state["evaluation_status"] = state["status"]
            state["evaluation_lineage_path"] = (Path(state["output"]) / "evaluation_lineage.json").as_posix() if (self.workspace / state["output"] / "evaluation_lineage.json").is_file() else None
            state["detail"] = None if state["status"] in {"completed", "partial"} else "评测失败；请查看评测日志与 evaluation_run.json"
            state["outputs"] = [p.relative_to(self.workspace).as_posix() for p in (self.workspace / state["output"]).glob("evaluation_*.json") if p.is_file() and not p.is_symlink()]
            parent = self._read_state(state["parent_run_id"])
            if parent:
                parent.update(evaluation_status=state["status"], evaluation_run_id=state["run_id"],
                              evaluation_detail=state.get("detail"), evaluation_lineage_path=state.get("evaluation_lineage_path"))
                self._write_state(parent)
        except Exception as exc:  # noqa: BLE001 - preserve terminal evidence for any supervisor failure
            state.update(status="failed", evaluation_status="failed", detail=f"评测监督器错误：{type(exc).__name__}")
            self._write_state(state)
        finally:
            state["completed_at"] = _utc_now(); state["elapsed_seconds"] = int(time.monotonic() - started)
            self._write_state(state)
            with self._lock:
                self._process = None; self._active_run_id = None; self._active_parent_run_id = None; self._monitor_thread = None
            self._release_gpu()

    @staticmethod
    def _trim_evaluation_log(log_path: Path) -> None:
        try:
            if log_path.is_symlink() or not log_path.is_file() or log_path.stat().st_size <= MAX_LOG_BYTES:
                return
            data = log_path.read_bytes()[-(MAX_LOG_BYTES // 2):]
            log_path.write_text("[日志较长，已保留尾部]\n" + data.decode("utf-8", errors="replace"), encoding="utf-8")
        except OSError:
            return

    def _registered_sft_manifest(self, dataset_id: str) -> Path:
        root = self.workspace / "data" / "processed"
        if root.is_symlink() or not root.is_dir():
            raise ValueError("找不到本地登记数据集目录")
        matches = []
        for path in root.glob("**/manifests/*.yaml"):
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(self.workspace):
                continue
            try:
                manifest = load_manifest(path)
            except (OSError, ValueError, TypeError):
                continue
            if manifest.dataset_id == dataset_id:
                matches.append((path, manifest))
        if len(matches) != 1:
            raise ValueError("dataset_id 必须唯一匹配一个本地登记 manifest")
        manifest_path, manifest = matches[0]
        if manifest.stage != "sft":
            raise ValueError("Mini SFT Launcher 只接受 SFT 阶段数据集")
        errors = validate_manifest_evidence(manifest_path)
        if errors:
            raise ValueError("登记数据集证据无效：" + "; ".join(errors[:3]))
        return manifest_path

    def registered_sft_datasets(self) -> list[dict[str, Any]]:
        root = self.workspace / "data" / "processed"
        if root.is_symlink() or not root.is_dir():
            return []
        result = []
        for path in list(root.glob("**/manifests/*.yaml"))[:500]:
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(self.workspace):
                continue
            try:
                manifest = load_manifest(path)
                if manifest.stage != "sft" or validate_manifest_evidence(path):
                    continue
                result.append({"dataset_id": manifest.dataset_id, "stage": manifest.stage,
                    "count": manifest.sample_count, "quality_status": manifest.metadata.get("quality_status", "unknown"),
                    "license_status": (manifest.origin.license_status if manifest.origin else "unknown"),
                    "source_name": (manifest.origin.name if manifest.origin else "未知来源")})
            except (OSError, ValueError, TypeError):
                continue
        return sorted(result, key=lambda row: row["dataset_id"])

    def _resolve_fixed_inputs(self, dataset_id: str = DATASET_ID) -> dict[str, Any]:
        recipe_path = _contained_file(self.workspace, RECIPE)
        recipe_raw = recipe_path.read_bytes()
        recipe = yaml.safe_load(recipe_raw)
        if not isinstance(recipe, dict) or recipe.get("stage") != "sft":
            raise ValueError("固定 recipe 格式或阶段不匹配")
        if recipe.get("parent_model_id") != MODEL_ID or recipe.get("dataset", {}).get("dataset_id") != DATASET_ID:
            raise ValueError("固定 recipe 的模型/数据身份与白名单不一致")
        if recipe.get("training", {}).get("max_length") != 1024 or recipe.get("seed") != 42:
            raise ValueError("固定 recipe 的 max_length/seed 与白名单不一致")
        model_rel = recipe.get("parent_model_path")
        if not isinstance(model_rel, str):
            raise TypeError("固定 recipe 缺少父模型路径")
        model_candidate = self.workspace / model_rel
        if model_candidate.is_symlink():
            raise ValueError("固定 Mini M1 路径不能是符号链接")
        model_path = model_candidate.resolve()
        if not model_path.is_relative_to(self.workspace) or not model_path.is_dir():
            raise ValueError("固定 Mini M1 模型不存在或越界；不会自动下载")
        model_registry = self.workspace / "runs/mini/cpt-finemath-5m-s42-v3/model_artifacts" / f"{MODEL_ID}.json"
        if model_registry.is_symlink() or not model_registry.is_file():
            raise ValueError("找不到固定父模型登记记录")
        model_record = json.loads(model_registry.read_text(encoding="utf-8"))
        if model_record.get("model_id") != MODEL_ID or model_record.get("kind") != "merged":
            raise ValueError("父模型登记身份不匹配")
        if model_record.get("produced_by_run_id"):
            validate_model_producing_run(self.workspace, model_registry, ModelLineage.model_validate(model_record))
        model_digest = sha256_directory(model_path)
        if model_record.get("digest", {}).get("value") != model_digest:
            raise ValueError("父模型目录摘要与登记记录不一致")

        dataset = recipe["dataset"]
        if dataset_id == DATASET_ID:
            manifest_path = _contained_file(self.workspace, dataset["manifest"])
            native_path = _contained_file(self.workspace, dataset["native"])
            canonical_path = _contained_file(self.workspace, dataset["canonical"])
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("dataset_id") != DATASET_ID or manifest.get("stage") != "sft"
                    or manifest.get("sample_count") != 307
                    or manifest.get("hashes", {}).get("canonical_records_sha256") != sha256_file(canonical_path)
                    or manifest.get("metadata", {}).get("native_sha256") != sha256_file(native_path)):
                raise ValueError("SFT 数据 manifest、样本数或内容摘要不匹配")
            if dataset.get("expected_count") != 307 or dataset.get("usage") != "exploratory":
                raise ValueError("固定数据集计数或 exploratory 标签不匹配")
            quality_status = manifest.get("metadata", {}).get("quality_status", "unknown")
            count = 307
            source_license = "verified"
            source_name = "DeepMath Mini SFT 子集"
        else:
            manifest_path = self._registered_sft_manifest(dataset_id)
            registered = load_manifest(manifest_path)
            artifact_root = manifest_path.parent.parent.resolve(strict=True)
            canonical_rel = registered.metadata.get("canonical_path")
            if not isinstance(canonical_rel, str):
                raise ValueError("登记数据集没有 canonical_path")
            canonical_unresolved = artifact_root / canonical_rel
            if canonical_unresolved.is_symlink():
                raise ValueError("登记 canonical 数据不能是符号链接")
            canonical_path = canonical_unresolved.resolve(strict=True)
            canonical_path.relative_to(artifact_root)
            if sha256_file(canonical_path) != registered.hashes.get("canonical_records_sha256"):
                raise ValueError("登记 canonical 数据摘要不匹配")
            native_path = canonical_path  # Preflight/launch adapts this exact registered content.
            manifest = registered.model_dump(mode="json")
            count = registered.sample_count
            quality_status = registered.metadata.get("quality_status", "unknown")
            source_license = registered.origin.license_status if registered.origin else "unknown"
            source_name = registered.origin.name if registered.origin else "未知来源"
        targeted_plan_id = manifest.get("metadata", {}).get("targeted_stage_plan_id") if isinstance(manifest, dict) else None
        targeted_plan_version = manifest.get("metadata", {}).get("targeted_stage_plan_version") if isinstance(manifest, dict) else None
        if targeted_plan_id is None and targeted_plan_version is not None:
            raise ValueError("SFT manifest 只有 targeted plan version，没有 plan ID")
        if targeted_plan_id is not None:
            if (not isinstance(targeted_plan_id, str)
                    or not re.fullmatch(r"plan-targeted-[a-f0-9]{20}", targeted_plan_id)
                    or not isinstance(targeted_plan_version, int) or targeted_plan_version < 1):
                raise ValueError("SFT manifest targeted plan 身份格式无效")
            from ..data.targeted_stage_plan import load_registered_targeted_stage_plan
            stage_plan = load_registered_targeted_stage_plan(self.workspace, targeted_plan_id, targeted_plan_version)
            if stage_plan.target.stage != "sft":
                raise ValueError("登记 SFT 数据引用的 targeted plan 不是 SFT 阶段")
        return {"recipe": recipe, "recipe_path": recipe_path, "recipe_sha256": hashlib.sha256(recipe_raw).hexdigest(),
                "model_path": model_path, "model_sha256": model_digest, "model_id": MODEL_ID,
                "manifest_path": manifest_path, "manifest_sha256": sha256_file(manifest_path),
                "canonical_sha256": sha256_file(canonical_path), "native_path": native_path,
                "native_sha256": sha256_file(native_path) if dataset_id == DATASET_ID else None,
                "dataset_id": dataset_id, "dataset_count": count, "quality_status": quality_status,
                "license_status": source_license, "source_name": source_name, "canonical_path": canonical_path,
                "targeted_stage_plan_id": targeted_plan_id, "targeted_stage_plan_version": targeted_plan_version}

    def preflight(self, dataset_id: str = DATASET_ID,
                  parameter_overrides: dict[str, int | float] | None = None,
                  memory_mode_id: str | None = None,
                  validation: dict[str, Any] | None = None,
                  checkpoint_policy: str = "final") -> dict[str, Any]:
        try:
            if checkpoint_policy not in {"final", "best"}:
                raise ValueError("checkpoint_policy 仅允许 final 或 best")
            if not isinstance(dataset_id, str) or len(dataset_id) > 200:
                raise ValueError("dataset_id 无效")
            inputs = self._resolve_fixed_inputs(dataset_id)
            validation_config = TrainingValidationConfig.model_validate(validation or {"mode": "none"})
            validation_summary = _sft_validation_summary(self.workspace, inputs, validation_config)
            checkpoint_selection = {"policy": checkpoint_policy,
                                    "metric": "loss" if checkpoint_policy == "best" else None,
                                    "greater_is_better": False if checkpoint_policy == "best" else None}
            if checkpoint_policy == "best" and (validation_config.mode == "none" or validation_summary.get("status") != "ready"):
                raise ValueError("选择 best 必须启用有效的训练期验证数据")
            profile = next(item for item in stage_profile_catalog(self.workspace).profiles
                           if item.profile_id == "sft-mini-1024-v1")
            effective_parameters = resolve_training_parameters(profile, parameter_overrides or {})
            memory_mode = resolve_training_memory_mode(profile, memory_mode_id)
            compatibility = inspect_training_memory_mode(memory_mode.mode_id)
            if not compatibility["ok"]:
                raise ValueError(" ".join(compatibility["blockers"]))
            recipe_data = yaml.safe_load(inputs["recipe_path"].read_text(encoding="utf-8"))
            if not isinstance(recipe_data, dict) or not isinstance(recipe_data.get("training"), dict):
                raise TypeError("批准的 SFT recipe 缺少 training 配置")
            training = {**recipe_data["training"], **effective_parameters, **memory_mode.settings}
            if checkpoint_policy == "best":
                training.update({"load_best_model_at_end": True, "metric_for_best_model": "loss",
                                 "greater_is_better": False})
            resolved_recipe = {**recipe_data, "training": training}
            training_config_sha256 = sha256_json({"profile_id": profile.profile_id,
                                                  "recipe_sha256": inputs["recipe_sha256"],
                                                  "training": resolved_recipe["training"]})
            effective_config_sha256 = sha256_json({"training_config_sha256": training_config_sha256,
                                                   "validation": validation_summary,
                                                   "checkpoint_selection": checkpoint_selection})
            resolved_recipe["_onegpu_ui"] = {
                "profile_id": profile.profile_id,
                "parameter_overrides": parameter_overrides or {},
                "training_memory_mode": memory_mode.model_dump(mode="json"),
                "validation": validation_summary,
                "checkpoint_selection": checkpoint_selection,
                "training_config_sha256": training_config_sha256,
                "effective_config_sha256": effective_config_sha256,
            }
            inputs.setdefault("dataset_count", 307)
            inputs.setdefault("quality_status", "unknown")
            inputs.setdefault("license_status", "verified")
            inputs.setdefault("source_name", None)
            inputs.setdefault("targeted_stage_plan_id", None)
            inputs.setdefault("targeted_stage_plan_version", None)
            run_id = new_run_id("ui-sft", label="mini-sft-1024")
            output = self.workspace / "runs" / "mini" / "ui-launcher" / run_id
            if output.exists():
                raise ValueError("本次输出目录已存在；请重新执行预检")
            # Exercise the same CLI plan/resolution contract in an isolated temp directory.
            import contextlib
            import io
            import tempfile
            from argparse import Namespace

            from ..cli import cmd_plan_sft
            with tempfile.TemporaryDirectory(prefix="onegpu-sft-preflight-") as temp:
                if dataset_id != DATASET_ID:
                    adapted = adapt_registered_dataset(inputs["manifest_path"], Path(temp) / "adapter")
                    inputs["native_path"] = Path(temp) / "adapter" / adapted["native_path"]
                    inputs["native_sha256"] = adapted["native_sha256"]
                resolved_recipe_path = Path(temp) / "resolved_recipe.yaml"
                resolved_recipe_path.write_text(yaml.safe_dump(resolved_recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")
                args = Namespace(config=str(resolved_recipe_path), model=str(inputs["model_path"]),
                                 dataset=str(inputs["native_path"]), output=temp,
                                 validation_dataset=(str(Path(temp) / "validation.jsonl")
                                                    if checkpoint_policy == "best" else None))
                with contextlib.redirect_stdout(io.StringIO()):
                    cmd_plan_sft(args)
                plan = json.loads((Path(temp) / "train_plan.json").read_text(encoding="utf-8"))
            if not _sft_runtime_available():
                raise ValueError("当前 Python 环境未安装 ms-swift 的 swift 命令；请安装项目 train 依赖后重启 UI")
            fingerprint = sha256_json({key: inputs[key] for key in (
                "recipe_sha256", "model_sha256", "manifest_sha256", "canonical_sha256", "native_sha256",
                "model_id", "dataset_id", "dataset_count", "quality_status", "license_status",
                "targeted_stage_plan_id", "targeted_stage_plan_version") } | {
                    "output": output.relative_to(self.workspace).as_posix(), "run_id": run_id,
                    "parameter_overrides": parameter_overrides or {},
                    "memory_mode_id": memory_mode.mode_id,
                    "effective_config_sha256": effective_config_sha256,
                    "validation": validation_summary})
            token = secrets.token_urlsafe(32)
            warnings = []
            if dataset_id == DATASET_ID:
                warnings.append("此 307 条数据是 exploratory 子集，语义质量未重新审核；结果不能代表正式 Mini 验收。")
            if inputs["license_status"] != "verified":
                warnings.append("该数据集来源或许可状态未知/未核验；启动前需要明确确认仅按本地声明使用。")
            warnings.extend(["父模型、seed、长度与 LoRA 配方保持固定；此处只切换已登记 SFT 数据集。",
                             "历史 RTX 4060 记录峰值约 7,939 / 8,188 MiB，显存余量很小；当前 GPU 读数仅供参考。",
                             "训练完成后统一评测仍需通过 CLI 显式执行。"])
            if memory_mode.mode_id == "qlora-4bit-nf4":
                warnings.append("QLoRA 4-bit NF4 可降低基座权重显存占用，但激活值仍随序列长度增长；不能保证长序列不会 OOM。")
            result = {"ok": True, "blockers": [], "warnings": warnings, "confirmation_token": token,
                      "expires_in_seconds": TOKEN_TTL_SECONDS, "run_id": run_id,
                      "recipe": RECIPE, "recipe_sha256": inputs["recipe_sha256"],
                      "resolved_config": plan, "training_parameters": {
                          "defaults": {item.key: item.default for item in profile.training_parameters},
                          "overrides": parameter_overrides or {}, "effective": effective_parameters,
                          "training_config_sha256": training_config_sha256,
                          "config_sha256": effective_config_sha256, "validation": validation_summary,
                          "checkpoint_selection": checkpoint_selection},
                      "validation": validation_summary,
                      "training_memory_mode": {**memory_mode.model_dump(mode="json"),
                                                "compatibility": compatibility},
                      "model": {"model_id": MODEL_ID, "path": model_path_relative(self.workspace, inputs["model_path"]), "sha256": inputs["model_sha256"]},
                      "dataset": {"dataset_id": dataset_id, "count": inputs["dataset_count"], "usage": "exploratory",
                                  "manifest_sha256": inputs["manifest_sha256"], "canonical_sha256": inputs["canonical_sha256"],
                                  "native_sha256": inputs["native_sha256"], "quality_status": inputs["quality_status"],
                                  "license_status": inputs["license_status"], "source_name": inputs.get("source_name")},
                      "output": output.relative_to(self.workspace).as_posix(), "gpu": _gpu_snapshot(),
                      "fingerprint": fingerprint}
            with self._lock:
                # A fresh configuration preflight supersedes prior pending consent tokens.
                self._confirmations.clear()
                self._confirmations[token] = {"expires": time.time() + TOKEN_TTL_SECONDS, "fingerprint": fingerprint,
                                              "run_id": run_id, "output": output.relative_to(self.workspace).as_posix(),
                                              "dataset_id": dataset_id, "requires_ack": inputs["license_status"] != "verified",
                                              "parameter_overrides": parameter_overrides or {},
                                              "memory_mode_id": memory_mode.mode_id,
                                              "effective_config_sha256": effective_config_sha256,
                                              "validation_config": validation_config.model_dump(mode="json"),
                      "checkpoint_policy": checkpoint_policy,
                                              "validation_summary": validation_summary}
            return result
        except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError, json.JSONDecodeError) as exc:
            return {"ok": False, "blockers": [str(exc)], "warnings": [], "gpu": _gpu_snapshot()}

    def start(self, token: str, acknowledged_dataset_id: str | None = None) -> dict[str, Any]:
        if not isinstance(token, str) or len(token) > 128:
            raise ValueError("预检确认令牌无效")
        with self._lock:
            confirmation = self._confirmations.get(token)
            if confirmation is None or confirmation["expires"] <= time.time():
                raise ValueError("预检已过期或已使用，请重新预检")
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError(f"已有 SFT 正在运行：{self._active_run_id}")
            self._assert_gpu_available()
            dataset_id = confirmation["dataset_id"]
            if confirmation["requires_ack"] and acknowledged_dataset_id != dataset_id:
                raise ValueError("来源/许可未核验；必须明确确认并提交完全匹配的 dataset_id")
            if not confirmation["requires_ack"] and acknowledged_dataset_id not in {None, dataset_id}:
                raise ValueError("确认的 dataset_id 与预检数据不匹配")
            self._confirmations.pop(token, None)
            inputs = self._resolve_fixed_inputs(dataset_id)
            validation_config = TrainingValidationConfig.model_validate(confirmation["validation_config"])
            validation_summary = _sft_validation_summary(self.workspace, inputs, validation_config)
            if sha256_json(validation_summary) != sha256_json(confirmation["validation_summary"]):
                raise ValueError("训练期验证数据或成员摘要已变化，请重新预检")
            inputs.setdefault("dataset_count", 307)
            inputs.setdefault("quality_status", "unknown")
            inputs.setdefault("license_status", "verified")
            inputs.setdefault("source_name", None)
            inputs.setdefault("targeted_stage_plan_id", None)
            inputs.setdefault("targeted_stage_plan_version", None)
            profile = next(item for item in stage_profile_catalog(self.workspace).profiles
                           if item.profile_id == "sft-mini-1024-v1")
            effective_parameters = resolve_training_parameters(profile, confirmation["parameter_overrides"])
            memory_mode = resolve_training_memory_mode(profile, confirmation["memory_mode_id"])
            compatibility = inspect_training_memory_mode(memory_mode.mode_id)
            if not compatibility["ok"]:
                raise ValueError("QLoRA 依赖在确认后发生变化，请修复项目环境并重新预检")
            recipe_data = yaml.safe_load(inputs["recipe_path"].read_text(encoding="utf-8"))
            if not isinstance(recipe_data, dict) or not isinstance(recipe_data.get("training"), dict):
                raise TypeError("批准的 SFT recipe 缺少 training 配置")
            checkpoint_policy = confirmation.get("checkpoint_policy", "final")
            checkpoint_selection = {"policy": checkpoint_policy,
                                    "metric": "loss" if checkpoint_policy == "best" else None,
                                    "greater_is_better": False if checkpoint_policy == "best" else None}
            training = {**recipe_data["training"], **effective_parameters, **memory_mode.settings}
            if checkpoint_policy == "best":
                if validation_config.mode == "none" or validation_summary.get("status") != "ready":
                    raise ValueError("选择 best 必须启用有效的训练期验证数据")
                training.update({"load_best_model_at_end": True, "metric_for_best_model": "loss",
                                 "greater_is_better": False})
            resolved_recipe = {**recipe_data, "training": training}
            training_config_sha256 = sha256_json({"profile_id": profile.profile_id,
                                                  "recipe_sha256": inputs["recipe_sha256"],
                                                  "training": resolved_recipe["training"]})
            effective_config_sha256 = sha256_json({"training_config_sha256": training_config_sha256,
                                                   "validation": validation_summary,
                                                   "checkpoint_selection": checkpoint_selection})
            resolved_recipe["_onegpu_ui"] = {
                "profile_id": profile.profile_id,
                "parameter_overrides": confirmation["parameter_overrides"],
                "training_memory_mode": memory_mode.model_dump(mode="json"),
                "validation": validation_summary,
                "checkpoint_selection": checkpoint_selection,
                "training_config_sha256": training_config_sha256,
                "effective_config_sha256": effective_config_sha256,
            }
            if effective_config_sha256 != confirmation["effective_config_sha256"]:
                raise ValueError("有效训练配置已变化，请重新预检")
            if dataset_id != DATASET_ID:
                import tempfile
                with tempfile.TemporaryDirectory(prefix="onegpu-sft-start-check-") as temp:
                    adapted = adapt_registered_dataset(inputs["manifest_path"], Path(temp) / "adapter")
                    inputs["native_sha256"] = adapted["native_sha256"]
            current = sha256_json({key: inputs[key] for key in (
                "recipe_sha256", "model_sha256", "manifest_sha256", "canonical_sha256", "native_sha256",
                "model_id", "dataset_id", "dataset_count", "quality_status", "license_status",
                "targeted_stage_plan_id", "targeted_stage_plan_version") } | {
                    "output": confirmation["output"], "run_id": confirmation["run_id"],
                    "parameter_overrides": confirmation["parameter_overrides"],
                    "memory_mode_id": memory_mode.mode_id,
                    "effective_config_sha256": effective_config_sha256,
                    "validation": validation_summary})
            if current != confirmation["fingerprint"]:
                raise ValueError("recipe 或输入摘要已变化，请重新预检")
            run_id = confirmation["run_id"]
            output = self.workspace / confirmation["output"]
            if output.exists():
                raise ValueError("输出目录冲突；请重新预检")
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.parent.is_symlink() or not output.parent.resolve().is_relative_to(self.workspace):
                raise ValueError("输出目录越过工作区边界")
            output.mkdir(exist_ok=False)
            resolved_recipe_path = output / "resolved_recipe.yaml"
            resolved_recipe_path.write_text(yaml.safe_dump(resolved_recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")
            native_path = inputs["native_path"]
            if dataset_id != DATASET_ID:
                adapted = adapt_registered_dataset(inputs["manifest_path"], output / "dataset-adapter")
                native_path = output / "dataset-adapter" / adapted["native_path"]
                if adapted["native_sha256"] != inputs["native_sha256"]:
                    raise ValueError("已登记数据在启动前发生变化；请重新预检")
            validation_native_path = None
            if validation_config.mode == "random_split":
                plan = plan_random_split(read_jsonl_records(inputs["canonical_path"]),
                                         ratio=float(validation_config.ratio), seed=int(validation_config.seed))
                if plan["split_identity_sha256"] != validation_summary.get("split_identity_sha256"):
                    raise ValueError("SFT 随机验证成员与已确认摘要不一致")
                split = apply_native_partition(native_path, plan, output / "validation-split")
                if split["membership_manifest_sha256"] != validation_summary.get("membership_manifest_sha256"):
                    raise ValueError("SFT 随机验证成员清单与已确认摘要不一致")
                native_path = split["train_path"]
                validation_native_path = split["validation_path"]
                validation_summary = {**validation_summary, "train_native_sha256": split["train_native_sha256"],
                                      "validation_native_sha256": split["validation_native_sha256"],
                                      "membership_manifest_path": split["membership_manifest_path"],
                                      "membership_manifest_sha256": split["membership_manifest_sha256"]}
            elif validation_config.mode == "registered":
                val_id = validation_config.dataset_id or ""
                val_identity = inspect_registered_validation_dataset(
                    self.workspace, val_id, "sft", profile.validation_dataset_roles,
                )
                from .training_preflight import _find_manifest
                val_manifest = _find_manifest(self.workspace, val_id)
                if (val_identity["content_sha256"] != validation_summary.get("content_sha256")
                        or val_identity["manifest_sha256"] != validation_summary.get("manifest_sha256")):
                    raise ValueError("SFT 独立验证集在启动前发生变化")
                adapted_val = adapt_registered_dataset(val_manifest, output / "validation-dataset")
                validation_native_path = output / "validation-dataset" / adapted_val["native_path"]
            state = {"run_id": run_id, "status": "starting", "started_at": _utc_now(), "completed_at": None,
                     "elapsed_seconds": 0, "progress": None, "progress_status": "unknown", "log_path": None,
                     "log_bytes": 0, "model_id": MODEL_ID, "dataset_id": dataset_id,
                     "targeted_stage_plan_id": inputs["targeted_stage_plan_id"],
                     "targeted_stage_plan_version": inputs["targeted_stage_plan_version"],
                     "quality_status": inputs["quality_status"], "license_status": inputs["license_status"],
                     "dataset_usage_acknowledged": confirmation["requires_ack"],
                     "parameter_overrides": confirmation["parameter_overrides"],
                     "training_memory_mode": {**memory_mode.model_dump(mode="json"),
                                               "compatibility": compatibility},
                     "training_parameters": {
                         "defaults": {item.key: item.default for item in profile.training_parameters},
                         "overrides": confirmation["parameter_overrides"],
                         "effective": effective_parameters,
                         "config_sha256": effective_config_sha256, "validation": validation_summary,
                         "checkpoint_selection": checkpoint_selection,
                     },
                     "validation": validation_summary,
                     "effective_config_sha256": effective_config_sha256,
                     "output": output.relative_to(self.workspace).as_posix(), "exit_code": None,
                     "outputs": [], "model_ids": [], "lineage_path": None, "detail": None}
            self._write_state(state)
            argv = [sys.executable, "-m", "onegpu_posttrain", "run-sft", "--config",
                    resolved_recipe_path.relative_to(self.workspace).as_posix(),
                    "--model", str(inputs["model_path"]), "--dataset", str(native_path),
                    "--output", str(output), "--run-id", run_id]
            if validation_native_path is not None:
                argv.extend(["--validation-dataset", str(validation_native_path)])
            if memory_mode.mode_id == "qlora-4bit-nf4":
                # The downstream SFT workflow consumes a registered merged model;
                # QLoRA's adapter is not itself an inference-ready checkpoint.
                argv.append("--merge")
            if inputs["targeted_stage_plan_id"] is not None:
                argv.extend(["--targeted-stage-plan-id", inputs["targeted_stage_plan_id"],
                             "--targeted-stage-plan-version", str(inputs["targeted_stage_plan_version"])])
            log_path = output / "launcher.log"
            if self._run_supervisor is None:
                self._run_supervisor = LocalRunSupervisor(self.workspace)
            self._run_supervisor.create(
                run_id=run_id, stage="sft", profile_id="sft-mini-1024-v1",
                output_location=output.relative_to(self.workspace).as_posix(),
            )
            self._run_supervisor.transition(run_id, "starting")
            lease = self._claim_gpu("training", run_id)
            try:
                process = subprocess.Popen(argv, cwd=self.workspace, shell=False,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                           encoding="utf-8", errors="replace", bufsize=1,
                                           **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}))
            except OSError as exc:
                self._release_gpu()
                state.update(status="failed", completed_at=_utc_now(),
                             detail=f"训练子进程无法启动：{exc.strerror or type(exc).__name__}")
                self._write_state(state)
                self._run_supervisor.transition(run_id, "failed", detail=state["detail"])
                return {"run_id": run_id, "status": "failed", "output": state["output"]}
            lease.set_owner_pid(getattr(process, "pid", os.getpid()))
            self._process, self._active_run_id = process, run_id
            self._run_supervisor.transition(run_id, "running")
            state.update(status="running", log_path=log_path.relative_to(self.workspace).as_posix())
            self._write_state(state)
            thread = threading.Thread(target=self._monitor, args=(process, state, log_path), daemon=True)
            self._monitor_thread = thread
            thread.start()
            return {"run_id": run_id, "status": "running", "output": state["output"]}

    def start_stage_cli(self, *, run_id: str, stage: str, profile_id: str, output_location: str,
                        argv: list[str], dataset_id: str, parent_model_id: str,
                        acknowledged_dataset_id: str | None = None,
                        license_status: str = "unknown",
                        training_parameters: dict[str, Any] | None = None,
                        cleanup_paths: list[str] | None = None) -> dict[str, Any]:
        """Run an allowlisted registered-stage CLI command under the shared local supervisor."""
        if not _RUN_ID_RE.fullmatch(run_id):
            raise ValueError("无效阶段训练 run_id")
        relative_output = Path(output_location)
        if (relative_output.is_absolute() or ".." in relative_output.parts or not relative_output.parts
                or not relative_output.as_posix().startswith(f"runs/{stage}/{profile_id}/")):
            raise ValueError("阶段训练输出必须是对应阶段/profile 下的工作区相对路径")
        output = self.workspace / relative_output
        if output.exists() or output.is_symlink() or not output.resolve(strict=False).is_relative_to(self.workspace):
            raise ValueError("阶段训练输出位置已存在或越界")
        if acknowledged_dataset_id not in {None, dataset_id}:
            raise ValueError("许可确认的数据集 ID 与训练输入不匹配")
        if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or "\x00" in arg for arg in argv):
            raise ValueError("阶段 adapter 命令无效")
        resolved_cleanup: list[Path] = []
        for value in cleanup_paths or []:
            relative = Path(value)
            expected = Path(".onegpu-posttrain") / "launcher" / "adapted" / run_id
            if relative.is_absolute() or relative != expected:
                raise ValueError("阶段临时数据清理路径不符合本次 Run allowlist")
            candidate = self.workspace / relative
            if candidate.is_symlink() or not candidate.resolve(strict=True).is_relative_to(self.workspace):
                raise ValueError("阶段临时数据清理路径越界或经过符号链接")
            resolved_cleanup.append(candidate)
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError(f"已有本地训练任务正在运行：{self._active_run_id}")
            self._assert_gpu_available()
            if self._run_supervisor is None:
                self._run_supervisor = LocalRunSupervisor(self.workspace)
            self._run_supervisor.create(run_id=run_id, stage=stage, profile_id=profile_id,
                                        output_location=relative_output.as_posix())
            self._run_supervisor.transition(run_id, "starting")
            started = _utc_now()
            state = {"run_id": run_id, "run_kind": "stage-training", "stage": stage,
                     "profile_id": profile_id, "status": "starting", "started_at": started,
                     "completed_at": None, "elapsed_seconds": 0, "progress": None,
                     "progress_status": "unknown", "dataset_id": dataset_id,
                     "parent_model_id": parent_model_id, "license_status": license_status,
                     "dataset_usage_acknowledged": acknowledged_dataset_id == dataset_id,
                     "training_parameters": training_parameters or {},
                     "validation": (training_parameters or {}).get("validation", {"mode": "none"}),
                     "effective_config_sha256": (training_parameters or {}).get("config_sha256"),
                     "output": relative_output.as_posix(),
                     "log_path": (Path(".onegpu-posttrain/launcher/logs") / f"{run_id}.log").as_posix(),
                     "exit_code": None, "outputs": [], "model_ids": [], "lineage_path": None}
            if resolved_cleanup:
                self._stage_cleanup_paths[run_id] = resolved_cleanup
            self._write_state(state)
            lease = self._claim_gpu("training", run_id)
            try:
                process = subprocess.Popen(
                    argv, cwd=self.workspace, shell=False, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
                    **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                       else {"start_new_session": True}),
                )
            except OSError as exc:
                self._release_gpu()
                detail = f"阶段训练子进程无法启动：{exc.strerror or type(exc).__name__}"
                state.update(status="failed", completed_at=_utc_now(), detail=detail)
                self._write_state(state)
                self._run_supervisor.transition(run_id, "failed", detail=detail)
                return {"run_id": run_id, "status": "failed", "output": relative_output.as_posix(), "detail": detail}
            lease.set_owner_pid(getattr(process, "pid", os.getpid()))
            self._process, self._active_run_id = process, run_id
            self._run_supervisor.transition(run_id, "running")
            state["status"] = "running"
            self._write_state(state)
            thread = threading.Thread(target=self._monitor, args=(process, state, self.workspace / state["log_path"]), daemon=True)
            self._monitor_thread = thread
            thread.start()
            return {"run_id": run_id, "status": "running", "output": relative_output.as_posix()}

    def _monitor(self, process: subprocess.Popen[str], state: dict[str, Any], log_path: Path) -> None:
        started = time.monotonic()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        supervisor_buffer: list[str] = []
        supervisor_buffer_bytes = 0
        supervisor_flush_at = started

        def flush_supervisor_log() -> None:
            nonlocal supervisor_buffer_bytes, supervisor_flush_at
            if supervisor_buffer:
                self._run_supervisor.append_log(state["run_id"], "".join(supervisor_buffer))
                supervisor_buffer.clear()
                supervisor_buffer_bytes = 0
                supervisor_flush_at = time.monotonic()

        try:
            with log_path.open("w", encoding="utf-8") as log:
                assert process.stdout is not None
                while True:
                    line = process.stdout.readline(8192)
                    if not line:
                        break
                    log.write(line)
                    log.flush()
                    supervisor_buffer.append(line)
                    supervisor_buffer_bytes += len(line.encode("utf-8", errors="replace"))
                    if supervisor_buffer_bytes >= 16 * 1024 or time.monotonic() - supervisor_flush_at >= 0.5:
                        flush_supervisor_log()
                    current_size = log_path.stat().st_size
                    if current_size > MAX_LOG_BYTES:
                        data = log_path.read_bytes()[-(MAX_LOG_BYTES // 2):]
                        log.seek(0)
                        log.truncate()
                        log.write("[日志较长，已保留尾部]\n")
                        log.write(data.decode("utf-8", errors="replace"))
                        log.flush()
                    match = re.search(
                        r"(?:global_step|step)\s*[=:]\s*['\"]?(\d+)(?:\s*/\s*(\d+))?['\"]?",
                        line, re.IGNORECASE,
                    )
                    if match:
                        state["progress"] = {"step": int(match.group(1)), "total": int(match.group(2)) if match.group(2) else None}
                        state["progress_status"] = "measured"
                        state["elapsed_seconds"] = int(time.monotonic() - started)
                        self._run_supervisor.update_progress(state["run_id"], line)
                        self._write_state(state)
            code = process.wait()
            flush_supervisor_log()
            state["exit_code"] = code
            state["status"] = "completed" if code == 0 and self._has_cli_success(state) else "oom" if self._logs_say_oom(log_path) else "failed"
            state["detail"] = None if state["status"] == "completed" else "训练未生成可确认的成功 run/lineage 记录"
            self._discover_outputs(state)
            self._record_dataset_usage_ack(state)
        except Exception as exc:  # noqa: BLE001 - supervisor evidence must survive child/log errors
            state.update(status="failed", detail=f"监督器错误：{type(exc).__name__}")
        finally:
            try:
                flush_supervisor_log()
                lifecycle = self._run_supervisor.get(state["run_id"])
                if lifecycle and lifecycle["status"] in {"queued", "starting", "running"}:
                    self._run_supervisor.transition(
                        state["run_id"], state.get("status", "failed"),
                        detail=state.get("detail"), exit_code=state.get("exit_code"),
                    )
            except (OSError, RuntimeError, ValueError):
                pass
            state["completed_at"] = _utc_now()
            state["elapsed_seconds"] = int(time.monotonic() - started)
            state["log_bytes"] = log_path.stat().st_size if log_path.exists() else 0
            if (state["run_id"] in self._shutdown_run_ids
                    and (state.get("status") != "completed" or not self._has_cli_success(state))):
                state.update(status="interrupted", detail="UI 服务关闭时请求停止训练")
            self._write_state(state)
            with self._lock:
                self._process = None
                self._active_run_id = None
                self._monitor_thread = None
            self._release_gpu()
            for path in self._stage_cleanup_paths.pop(str(state["run_id"]), []):
                if path.is_symlink() or not path.resolve(strict=False).is_relative_to(self.workspace):
                    continue
                shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _logs_say_oom(path: Path) -> bool:
        try:
            return "out of memory" in path.read_text(encoding="utf-8", errors="replace")[-MAX_LOG_BYTES:].lower()
        except OSError:
            return False

    def _has_cli_success(self, state: dict[str, Any]) -> bool:
        output = self.workspace / str(state["output"])
        try:
            record = json.loads((output / "lineage.json").read_text(encoding="utf-8"))
            run = json.loads((output / "run.json").read_text(encoding="utf-8"))
            return record.get("status") == "completed" and run.get("run_id") == state["run_id"]
        except (OSError, ValueError, TypeError):
            return False

    def _discover_outputs(self, state: dict[str, Any]) -> None:
        output = self.workspace / str(state["output"])
        state.setdefault("outputs", [])
        state.setdefault("model_ids", [])
        for relative in ("run.json", "lineage.json", "resolved_config.json", "merged/M2-SFT", "train"):
            path = output / relative
            artifact_path = path.relative_to(self.workspace).as_posix()
            if path.exists() and artifact_path not in state["outputs"]:
                state["outputs"].append(artifact_path)
        if (output / "lineage.json").is_file():
            state["lineage_path"] = (output / "lineage.json").relative_to(self.workspace).as_posix()
        run_record = output / "run.json"
        if run_record.is_file() and not run_record.is_symlink():
            try:
                recorded = json.loads(run_record.read_text(encoding="utf-8"))
                if isinstance(recorded.get("checkpoint_selection"), dict):
                    state["checkpoint_selection"] = recorded["checkpoint_selection"]
            except (OSError, ValueError, TypeError):
                pass
        artifacts = output / "model_artifacts"
        if artifacts.is_dir() and not artifacts.is_symlink():
            state.setdefault("model_artifacts", [])
            for path in list(artifacts.glob("model-*.json"))[:32]:
                if path.is_symlink() or path.stat().st_size > 64 * 1024:
                    continue
                try:
                    artifact = json.loads(path.read_text(encoding="utf-8"))
                    model_id = artifact.get("model_id")
                    if re.fullmatch(r"model-[a-f0-9]{20}", model_id or "") and model_id not in state["model_ids"]:
                        state["model_ids"].append(model_id)
                    if re.fullmatch(r"model-[a-f0-9]{20}", model_id or ""):
                        state["model_artifacts"].append({"model_id": model_id, "kind": artifact.get("kind"),
                            "path": artifact.get("path"), "digest": artifact.get("digest", {}).get("value"),
                            "produced_by_run_id": artifact.get("produced_by_run_id"), "stage": artifact.get("stage")})
                except (OSError, ValueError, TypeError):
                    continue

    def _record_dataset_usage_ack(self, state: dict[str, Any]) -> None:
        if not state.get("dataset_usage_acknowledged"):
            return
        lineage_path = self.workspace / str(state.get("output", "")) / "lineage.json"
        try:
            if lineage_path.is_symlink() or not lineage_path.is_file() or not lineage_path.resolve().is_relative_to(self.workspace):
                return
            lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
            warnings = lineage.get("lineage_warnings")
            if not isinstance(warnings, list):
                warnings = []
            note = (f"User explicitly acknowledged local use while source/license remains unverified; "
                    f"dataset_id={state.get('dataset_id')}; license_status={state.get('license_status')}.")
            if note not in warnings:
                warnings.append(note)
            lineage["lineage_warnings"] = warnings
            lineage_path.write_text(json.dumps(lineage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except (OSError, ValueError, TypeError):
            state.setdefault("lineage_warnings", []).append("Dataset usage acknowledgement could not be recorded in run lineage")

    def get_run(self, run_id: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            run_id = run_id or self._active_parent_run_id or self._active_run_id
            if not run_id:
                return self.latest_run()
            state = self._read_state(run_id)
            if state and state.get("run_kind") == "evaluation" and self._recover_finished_evaluation(state):
                self._write_state(state)
                parent_id = state.get("parent_run_id")
                parent = self._read_state(parent_id) if isinstance(parent_id, str) else None
                if parent:
                    parent.update(evaluation_status=state["status"], evaluation_run_id=state["run_id"],
                                  evaluation_detail=state.get("detail"),
                                  evaluation_lineage_path=state.get("evaluation_lineage_path"))
                    self._write_state(parent)
            if state and state.get("status") in {"starting", "running"} and self._process is None:
                # The durable supervisor owns the lifecycle across Windows
                # console-script wrappers. A missing in-memory Popen handle is
                # not enough evidence to mark its still-active child interrupted.
                lifecycle = self._run_supervisor.get(run_id) if self._run_supervisor else None
                if not lifecycle or lifecycle.get("status") not in {"queued", "starting", "running"}:
                    state.update(status="interrupted", detail="监督进程不可用；不会自动续训")
                    self._write_state(state)
            elif (state and state.get("status") == "interrupted"
                  and state.get("detail") == "监督进程不可用；不会自动续训"
                  and self._process is None):
                # A prior read may have raced with Windows' nested Python
                # launcher. Restore only when the durable record still says
                # the same run is active; supervisor startup marks stale runs
                # interrupted, so this cannot resume work after a restart.
                lifecycle = self._run_supervisor.get(run_id) if self._run_supervisor else None
                if lifecycle and lifecycle.get("status") in {"queued", "starting", "running"}:
                    state.update(status=lifecycle["status"], detail=lifecycle.get("detail"), completed_at=None)
                    self._write_state(state)
            if state:
                # Upgrade persisted completed SFT runs created before model artifact
                # summaries were added, so the evaluation CTA appears without retraining.
                if state.get("run_kind") != "evaluation" and state.get("status") == "completed" and "model_artifacts" not in state:
                    state["model_artifacts"] = []
                    self._discover_outputs(state)
                    self._write_state(state)
                if state.get("status") in {"starting", "running"} and state.get("started_at"):
                    try:
                        started = datetime.fromisoformat(state["started_at"])
                        state["elapsed_seconds"] = max(0, int((datetime.now(started.tzinfo) - started).total_seconds()))
                    except (TypeError, ValueError):
                        pass
                stage_logs = self._stage_log_tail(state)
                if stage_logs:
                    state["log_tail"] = stage_logs["text"]
                    state["log_sources"] = stage_logs["sources"]
                    state["log_truncated"] = stage_logs["truncated"]
                    parsed = self._parse_training_progress(stage_logs["text"])
                    if parsed:
                        state["progress"] = parsed
                        state["progress_status"] = "measured"
                    metrics = self._parse_training_metrics(stage_logs["text"])
                    if metrics:
                        state["training_metrics"] = metrics
                    validation_metrics = self._parse_validation_metrics(stage_logs["text"])
                    if validation_metrics:
                        state["validation_metrics"] = validation_metrics
                else:
                    # Supervisor output is a fallback only; it contains CLI summaries,
                    # not necessarily the trainer's live progress stream.
                    log_path = state.get("log_path")
                    if log_path:
                        path = self.workspace / log_path
                        try:
                            if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(self.workspace):
                                state["log_tail"] = path.read_bytes()[-MAX_LOG_TAIL:].decode("utf-8", errors="replace")
                                state["log_truncated"] = path.stat().st_size > MAX_LOG_TAIL
                                state["log_sources"] = ["supervisor"]
                        except OSError:
                            state["log_tail"] = ""
                state["evaluation_status"] = (state.get("status") if state.get("run_kind") == "evaluation"
                                               else "pending" if state.get("status") == "completed" else "not_applicable")
                if state.get("run_kind") != "evaluation" and state.get("evaluation_run_id"):
                    evaluation = self._read_state(state["evaluation_run_id"])
                    if evaluation and self._recover_finished_evaluation(evaluation):
                        self._write_state(evaluation)
                        state.update(evaluation_status=evaluation["status"],
                                     evaluation_detail=evaluation.get("detail"),
                                     evaluation_lineage_path=evaluation.get("evaluation_lineage_path"))
                        self._write_state(state)
                    if evaluation:
                        state["evaluation_status"] = evaluation.get("status", state.get("evaluation_status"))
                        state["evaluation_run_id"] = evaluation["run_id"]
                        state["evaluation_detail"] = evaluation.get("detail")
                        state["evaluation_log_tail"] = self._bounded_log(evaluation.get("log_path"))
                        if evaluation.get("scoring_strategy") == "bailian_llm_judge":
                            state["evaluation_progress"] = self._choice_judge_progress(
                                state["evaluation_log_tail"], evaluation,
                            )
                        state["evaluation_outputs"] = evaluation.get("outputs", [])
                        state["evaluation_lineage_path"] = evaluation.get("evaluation_lineage_path")
                        state["evaluation_metrics"] = self._read_evaluation_metrics(evaluation)
                elif state.get("run_kind") == "evaluation" and state.get("scoring_strategy") == "bailian_llm_judge":
                    state["choice_judge_progress"] = self._choice_judge_progress(state.get("log_tail", ""), state)
            return state

    @staticmethod
    def _choice_judge_progress(log_text: str, state: dict[str, Any]) -> dict[str, Any]:
        from ..evaluation.choice_judge import parse_choice_judge_progress

        return parse_choice_judge_progress(
            log_text, total=state.get("evaluation_sample_count"),
            status=state.get("evaluation_status") or state.get("status") or "running",
        )

    def _bounded_log(self, relative: str | None) -> str:
        if not relative:
            return ""
        path = self.workspace / relative
        try:
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(self.workspace):
                return ""
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - MAX_LOG_TAIL))
                return stream.read(MAX_LOG_TAIL).decode("utf-8", errors="replace")
        except OSError:
            return ""

    def _read_evaluation_metrics(self, evaluation: dict[str, Any]) -> dict[str, Any] | None:
        path = self.workspace / str(evaluation.get("output", "")) / "eval.json"
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024 or not path.resolve().is_relative_to(self.workspace):
                return None
            result = json.loads(path.read_text(encoding="utf-8"))
            metric_statuses = result.get("metric_statuses")
            if isinstance(metric_statuses, dict):
                metrics = []
                for name, item in metric_statuses.items():
                    if not isinstance(name, str) or not isinstance(item, dict):
                        continue
                    value = item.get("value")
                    metrics.append({"name": name, "metric": "value",
                                    "value": value if isinstance(value, (int, float)) and not isinstance(value, bool) else None,
                                    "denominator": item.get("denominator") if isinstance(item.get("denominator"), int) else None,
                                    "denominator_unit": item.get("denominator_unit"),
                                    "status": item.get("status", "unknown"), "reason": item.get("reason"),
                                    "parse_failures": item.get("parse_failures")
                                    if isinstance(item.get("parse_failures"), int) else None,
                                    "comparability_key": item.get("comparability_key")})
                return {"protocol_version": result.get("protocol_version") or result.get("resolved_evaluation", {}).get("protocol_version", "unknown"),
                        "status": evaluation.get("status"), "metrics": metrics,
                        "evaluation_suite": result.get("evaluation_suite"),
                        "comparability": result.get("comparability"),
                        "model_id": result.get("model_id") or evaluation.get("model_id"),
                        "run_id": evaluation.get("run_id")}
            reports = result.get("evalscope_reports")
            if not isinstance(reports, dict):
                return None
            expected = {"gsm8k": "GSM8K", "math_500": "MATH-500"}
            metrics = []
            for key, label in expected.items():
                report = next((value for name, value in reports.items() if key in str(name).lower()), None)
                if isinstance(report, dict) and isinstance(report.get("score"), (int, float)):
                    denominator = report.get("samples")
                    metrics.append({"name": label, "metric": "accuracy@1", "value": report["score"],
                                    "denominator": denominator if isinstance(denominator, int) else None})
            return {"protocol_version": result.get("resolved_evaluation", {}).get("protocol_version", "math-eval-v1"),
                    "status": evaluation.get("status"), "metrics": metrics,
                    "model_id": evaluation.get("model_id"), "run_id": evaluation.get("run_id")}
        except (OSError, ValueError, TypeError):
            return None

    def _stage_log_tail(self, state: dict[str, Any]) -> dict[str, Any] | None:
        """Read only known run-local training/merge logs, with a strict bounded tail."""
        output_rel = state.get("output")
        if not isinstance(output_rel, str):
            return None
        try:
            output_unresolved = self.workspace / output_rel
            if output_unresolved.is_symlink():
                return None
            output = output_unresolved.resolve(strict=True)
            if not output.is_relative_to(self.workspace) or not output.is_dir():
                return None
        except (OSError, ValueError):
            return None
        # Prefer the trainer's own stream; include stderr because ms-swift commonly
        # emits progress there. Merge output is useful after SFT's merge phase.
        candidates = [
            ("train/stdout", output / "logs" / "train" / "stdout.log"),
            ("train/stderr", output / "logs" / "train" / "stderr.log"),
            ("train/events", output / "train"),
            ("merge/stdout", output / "logs" / "merge" / "stdout.log"),
            ("merge/stderr", output / "logs" / "merge" / "stderr.log"),
        ]
        chunks: list[tuple[str, bytes, int]] = []
        for source, path in candidates:
            if source == "train/events":
                try:
                    paths = sorted(path.glob("*/logging.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True)[:1]
                except OSError:
                    paths = []
                path = paths[0] if paths else path / "__missing__"
            try:
                if (path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != output and parent.is_relative_to(output))
                        or not path.is_file() or not path.resolve().is_relative_to(output)):
                    continue
                size = path.stat().st_size
                with path.open("rb") as stream:
                    stream.seek(max(0, size - MAX_LOG_TAIL))
                    data = stream.read(MAX_LOG_TAIL)
                if data:
                    chunks.append((source, data, size))
            except OSError:
                continue
        if not chunks:
            return None
        # Keep total response bounded even when there are several stage files.
        combined = b"\n".join(b"--- " + name.encode() + b" ---\n" + data for name, data, _ in chunks)
        return {
            "text": combined[-MAX_LOG_TAIL:].decode("utf-8", errors="replace"),
            "sources": [name for name, _, _ in chunks],
            "truncated": any(size > MAX_LOG_TAIL for _, _, size in chunks) or len(combined) > MAX_LOG_TAIL,
        }

    @staticmethod
    def _parse_training_progress(text: str) -> dict[str, Any] | None:
        patterns = (
            r"global_step/max_steps\s*['\"]?\s*:\s*['\"]?(\d+)\s*/\s*(\d+)['\"]?",
            r"(?:global_step|step)\s*[=:]\s*['\"]?(\d+)(?:\s*/\s*(\d+))?['\"]?",
        )
        found = None
        for line in text.splitlines():
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    found = {"step": int(match.group(1)), "total": int(match.group(2)) if match.group(2) else None}
        return found

    @staticmethod
    def _parse_training_metrics(text: str) -> dict[str, Any]:
        allowed = {"loss", "train_loss", "learning_rate", "token_acc", "epoch", "elapsed_time", "remaining_time", "memory(GiB)", "train_speed(s/it)"}
        result: dict[str, Any] = {}
        for line in text.splitlines():
            start = line.find("{")
            if start < 0:
                continue
            try:
                record = ast.literal_eval(line[start:])
            except (ValueError, SyntaxError):
                continue
            if not isinstance(record, dict):
                continue
            for key in allowed:
                value = record.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool) or key in {"elapsed_time", "remaining_time"} and isinstance(value, str):
                    result[key] = value
        return result

    @staticmethod
    def _parse_validation_metrics(text: str) -> dict[str, int | float]:
        """Keep ms-swift eval_* metrics separate from optimization metrics."""
        result: dict[str, int | float] = {}
        for line in text.splitlines():
            start = line.find("{")
            if start < 0:
                continue
            try:
                record = ast.literal_eval(line[start:])
            except (ValueError, SyntaxError):
                continue
            if not isinstance(record, dict):
                continue
            for key, value in record.items():
                if (isinstance(key, str) and re.fullmatch(r"eval_[a-z0-9_]+", key)
                        and isinstance(value, (int, float)) and not isinstance(value, bool)):
                    result[key] = value
        return result

    def latest_run(self) -> dict[str, Any] | None:
        try:
            files = sorted(self.records_root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for path in files:
                if not path.is_symlink() and path.stat().st_size <= 128 * 1024:
                    state = self._read_state(path.stem)
                    if state and state.get("run_kind") != "evaluation":
                        return self.get_run(path.stem)
        except OSError:
            return None
        return None

    def shutdown(self, grace_seconds: float = 5.0) -> None:
        with self._lock:
            process, run_id = self._process, self._active_run_id
            monitor_thread = self._monitor_thread
        if process is None:
            return
        if process.poll() is not None:
            if monitor_thread is not None:
                monitor_thread.join(timeout=grace_seconds)
            if run_id:
                state = self._read_state(run_id)
                if state and state.get("status") in {"starting", "running"}:
                    state.update(status="interrupted", completed_at=_utc_now(),
                                 detail="UI 服务关闭时监督器未能确认训练结果")
                    self._write_state(state)
            return
        with self._lock:
            if run_id:
                self._shutdown_run_ids.add(run_id)
                state = self._read_state(run_id)
                if state and state.get("status") in {"starting", "running"}:
                    state.update(status="interrupted", completed_at=_utc_now(), detail="UI 服务关闭时请求停止任务")
                    self._write_state(state)
                    if state.get("run_kind") == "evaluation" and state.get("parent_run_id"):
                        parent = self._read_state(state["parent_run_id"])
                        if parent:
                            parent.update(evaluation_status="interrupted", evaluation_detail=state["detail"])
                            self._write_state(parent)
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=grace_seconds)
        except (OSError, subprocess.TimeoutExpired):
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=3, check=False)
                else:
                    os.killpg(process.pid, signal.SIGKILL)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
        finally:
            if monitor_thread is not None and monitor_thread is not threading.current_thread():
                monitor_thread.join(timeout=grace_seconds)


def model_path_relative(workspace: Path, path: Path) -> str:
    return path.relative_to(workspace).as_posix()
