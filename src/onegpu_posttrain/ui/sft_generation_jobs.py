"""Persisted, single-worker supervisor for explicit local SFT generation jobs."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..data.generated_sft_registration import (
    preview_generated_sft_dataset,
    register_generated_sft_dataset,
)
from ..synthetic.ordinary_bundle import load_ordinary_sft_bundle
from ..synthetic.ordinary_workflow import (
    execute_ordinary_sft_generation,
    preflight_ordinary_sft_generation,
)
from ..synthetic.prompt_only import load_prompt_only_bundle
from ..synthetic.prompt_only_workflow import (
    execute_prompt_only_sft_generation,
    preflight_prompt_only_sft_generation,
)
from ..synthetic.sft_workflow import execute_sft_generation, preflight_sft_generation
from ..synthetic.workflow_bundle import load_sft_workflow_bundle
from ..synthetic.request_guard import acknowledge_uncertain_attempt, ensure_no_uncertain_requests, unresolved_attempts
from ..data.targeted_sft_bridge import validate_targeted_sft_bridge
from ..data.targeted_stage_plan import load_registered_targeted_stage_plan

_JOB_ID = re.compile(r"^sftgen-[a-f0-9]{20}$")
_TERMINAL = {"completed", "partial", "cancelled", "failed", "interrupted"}
_MAX_EVENTS_BYTES = 1024 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SFTGenerationJobs:
    """One active local generation at a time; reload reads persisted state only."""

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve(strict=True)
        self.root = self.workspace / "runs" / "synthetic-workflow"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._active_id: str | None = None
        self._thread: threading.Thread | None = None
        self._stop_events: dict[str, threading.Event] = {}
        self._recover_interrupted()

    def _job_dir(self, execution_id: str) -> Path:
        if not _JOB_ID.fullmatch(execution_id):
            raise ValueError("execution_id 无效")
        path = self.root / execution_id
        if path.is_symlink() or not path.resolve(strict=False).is_relative_to(self.root):
            raise ValueError("execution 路径不可用")
        return path

    def request_review(self, bundle_id: str, *, attempt_id: str | None = None) -> list[dict[str, Any]]:
        """Read or explicitly acknowledge unknown provider outcomes for a validated bundle."""
        if bundle_id.startswith("plan-sft-prompt-"):
            load_prompt_only_bundle(self.workspace, bundle_id)
        elif bundle_id.startswith("plan-sft-ordinary-"):
            load_ordinary_sft_bundle(self.workspace, bundle_id)
        else:
            raise ValueError("该方案不支持此请求审查")
        bundle_dir = self.workspace / "artifacts" / "synthetic" / bundle_id
        with self._lock:
            if attempt_id is not None:
                if self._active_id is not None:
                    state = self._read(self._active_id)
                    if state and state.get("status") not in _TERMINAL:
                        raise RuntimeError("生成任务运行中，无法处理未知请求")
                acknowledge_uncertain_attempt(bundle_dir, attempt_id)
            return unresolved_attempts(bundle_dir)

    def _read(self, execution_id: str) -> dict[str, Any] | None:
        path = self._job_dir(execution_id) / "state.json"
        if path.is_symlink() or not path.is_file():
            return None
        for attempt in range(5):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
                return state if isinstance(state, dict) and state.get("execution_id") == execution_id else None
            except OSError:
                if attempt < 4:
                    time.sleep(0.005)
                    continue
                return None
            except (UnicodeError, json.JSONDecodeError):
                return None
        return None

    def _write(self, state: dict[str, Any]) -> None:
        with self._lock:
            path = self._job_dir(state["execution_id"]) / "state.json"
            temporary = path.with_name(f"state-{uuid4().hex}.tmp")
            temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            try:
                for attempt in range(10):
                    try:
                        os.replace(temporary, path)
                        break
                    except PermissionError:
                        if attempt == 9:
                            raise
                        time.sleep(0.01)
            finally:
                temporary.unlink(missing_ok=True)

    def _event(self, execution_id: str, event: dict[str, Any]) -> None:
        path = self._job_dir(execution_id) / "events.jsonl"
        if path.exists() and path.stat().st_size > _MAX_EVENTS_BYTES:
            return
        # Only caller-built numeric/status summaries are accepted; never persist raw provider content.
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")

    def _shared_plans_for_bundle(self, bundle_id: str) -> list[dict[str, Any]]:
        plan_root = self.workspace / "artifacts" / "targeted-plans"
        if not plan_root.is_dir() or plan_root.is_symlink():
            return []
        matches = []
        for path in plan_root.glob("plan-targeted-*/v*/plan.json"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                plan_id, version = payload.get("plan_id"), payload.get("plan_version")
                plan = load_registered_targeted_stage_plan(self.workspace, plan_id, version)
                bridge = validate_targeted_sft_bridge(self.workspace, stage_plan=plan, bundle_id=bundle_id)
                matches.append({"plan_id": plan.plan_id, "version": plan.version,
                                "scenario_id": plan.target.scenario_id, "bridge": bridge})
            except (OSError, TypeError, ValueError):
                continue
        return matches

    def _recover_interrupted(self) -> None:
        for state_path in self.root.glob("sftgen-*/state.json"):
            if state_path.is_symlink() or not state_path.is_file():
                continue
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if isinstance(state, dict) and state.get("status") in {"queued", "running"}:
                state.update(status="interrupted", detail="服务重启时任务仍在运行；未自动重启 provider 请求", updated_at=_now())
                self._write(state)

    def bundles(self) -> list[dict[str, Any]]:
        root = self.workspace / "artifacts" / "synthetic"
        if root.is_symlink() or not root.is_dir():
            return []
        items = []
        for path in root.glob("plan-targeted-*/workflow_bundle.json"):
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
                continue
            try:
                bundle_id = path.parent.name
                bundle, inputs = load_sft_workflow_bundle(self.workspace, bundle_id)
                import yaml

                from ..data.targeted_plan import load_targeted_plan
                plan = load_targeted_plan(inputs["plan"])
                profile = yaml.safe_load(inputs["profile"].read_text(encoding="utf-8"))
                items.append({
                    "bundle_id": bundle.bundle_id,
                    "source_dataset_id": plan.training_seed.dataset_id,
                    "scenario_id": plan.target_scenario_id,
                    "provider_profile_id": profile.get("profile_id"),
                    "model": profile.get("model"),
                    "max_requests": plan.budget.max_requests,
                    "max_candidates": plan.budget.target_candidates,
                    "shared_stage_plans": self._shared_plans_for_bundle(bundle.bundle_id),
                })
            except (OSError, ValueError, TypeError):
                continue
        for path in root.glob("plan-sft-ordinary-*/ordinary_workflow_bundle.json"):
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
                continue
            try:
                bundle_id = path.parent.name
                bundle, files = load_ordinary_sft_bundle(self.workspace, bundle_id)
                plan = json.loads(files["plan"].read_text(encoding="utf-8"))
                import yaml
                profile = yaml.safe_load(files["profile"].read_text(encoding="utf-8"))
                items.append({
                    "bundle_id": bundle.bundle_id,
                    "mode": "ordinary_sft",
                    "source_dataset_id": plan["source_dataset_id"],
                    "provider_profile_id": plan["provider_profile_id"],
                    "model": profile.get("model"),
                    "target_count": plan["target_count"],
                    "max_requests": plan["request_limit"],
                    "max_candidates": plan["target_count"],
                })
            except (OSError, ValueError, TypeError):
                continue
        for path in root.glob("plan-sft-prompt-*/prompt_only_workflow_bundle.json"):
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
                continue
            try:
                bundle_id = path.parent.name
                plan, files = load_prompt_only_bundle(self.workspace, bundle_id)
                import yaml
                profile = yaml.safe_load(files["profile"].read_text(encoding="utf-8"))
                items.append({"bundle_id": bundle_id, "mode": "prompt_only_sft",
                    "provider_profile_id": plan.provider_profile_id, "model": profile.get("model"),
                    "target_count": plan.target_count, "max_requests": plan.request_limit,
                    "max_candidates": plan.target_count})
            except (OSError, ValueError, TypeError):
                continue
        return sorted(items, key=lambda item: item["bundle_id"])

    def _validate_shared_plan(self, bundle_id: str, stage_plan_id: str | None, plan_version: int | None):
        if stage_plan_id is None and plan_version is None:
            return None
        if not isinstance(stage_plan_id, str) or not isinstance(plan_version, int) or isinstance(plan_version, bool):
            raise ValueError("shared targeted stage plan requires both plan_id and version")
        plan = load_registered_targeted_stage_plan(self.workspace, stage_plan_id, plan_version)
        bridge = validate_targeted_sft_bridge(self.workspace, stage_plan=plan, bundle_id=bundle_id)
        return {"plan_id": plan.plan_id, "plan_version": plan.version, "bridge": bridge}

    def preflight(self, bundle_id: str, *, usable_target: int, max_requests: int, target_candidates: int,
                  stage_plan_id: str | None = None, plan_version: int | None = None):
        shared_plan = self._validate_shared_plan(bundle_id, stage_plan_id, plan_version)
        if bundle_id.startswith("plan-sft-prompt-"):
            prepared = preflight_prompt_only_sft_generation(self.workspace, bundle_id)
            ensure_no_uncertain_requests(self.workspace / "artifacts" / "synthetic" / bundle_id)
            if usable_target != prepared["target_count"]:
                raise ValueError("prompt-only SFT target is immutable; prepare a new plan")
            if target_candidates != prepared["target_count"] or max_requests > prepared["request_limit"]:
                raise ValueError("request budget cannot exceed immutable prompt-only plan")
            prepared.update({"candidate_budget": prepared["target_count"], "plan_request_limit": prepared["request_limit"],
                             "request_limit": max_requests, "max_requests": max_requests,
                             "cost_status": "unknown", "estimated_output_cost_upper_bound_usd": None})
            return prepared | ({"targeted_stage_plan": shared_plan} if shared_plan else {})
        if bundle_id.startswith("plan-sft-ordinary-"):
            prepared = preflight_ordinary_sft_generation(self.workspace, bundle_id)
            ensure_no_uncertain_requests(self.workspace / "artifacts" / "synthetic" / bundle_id)
            if usable_target != prepared["target_count"]:
                raise ValueError("普通 SFT 目标数在方案登记后不可变；请重新准备新方案")
            if target_candidates != prepared["target_count"] or max_requests > prepared["request_limit"]:
                raise ValueError("本次预算不能超过不可变普通 SFT 方案目标")
            prepared.update({"desired_usable_records": prepared["target_count"],
                             "candidate_budget": prepared["target_count"],
                             "plan_request_limit": prepared["request_limit"], "request_limit": max_requests,
                             "max_requests": max_requests,
                             "max_output_tokens_per_request": prepared["max_output_tokens_per_request"],
                             "cost_status": "unknown", "estimated_output_cost_upper_bound_usd": None})
            return prepared | ({"targeted_stage_plan": shared_plan} if shared_plan else {})
        prepared = preflight_sft_generation(
            self.workspace, bundle_id, usable_target=usable_target,
            max_requests=max_requests, target_candidates=target_candidates,
        )
        return prepared | ({"targeted_stage_plan": shared_plan} if shared_plan else {})

    def start(self, bundle_id: str, *, usable_target: int, max_requests: int, target_candidates: int,
              stage_plan_id: str | None = None, plan_version: int | None = None) -> dict[str, Any]:
        self.preflight(bundle_id, usable_target=usable_target, max_requests=max_requests,
                       target_candidates=target_candidates, stage_plan_id=stage_plan_id, plan_version=plan_version)
        with self._lock:
            if self._active_id is not None:
                state = self._read(self._active_id)
                if state and state.get("status") not in _TERMINAL:
                    raise RuntimeError("已有 SFT 数据生成任务运行中")
            execution_id = f"sftgen-{uuid4().hex[:20]}"
            job_dir = self._job_dir(execution_id)
            job_dir.mkdir(parents=True, exist_ok=False)
            state = {
                "execution_id": execution_id, "bundle_id": bundle_id, "status": "queued",
                "stage": "sft",
                "source_mode": "targeted" if stage_plan_id is not None or bundle_id.startswith("plan-targeted-") else "ordinary",
                "mode": ("prompt_only_sft" if bundle_id.startswith("plan-sft-prompt-") else
                         "ordinary_sft" if bundle_id.startswith("plan-sft-ordinary-") else "targeted"),
                "phase": "queued", "created_at": _now(), "updated_at": _now(),
                "usable_target": usable_target, "candidate_budget": target_candidates,
                "max_requests": max_requests, "progress": None, "result": None,
                "stage_plan_id": stage_plan_id, "plan_version": plan_version,
                "detail": None,
            }
            self._write(state)
            stop = threading.Event()
            self._stop_events[execution_id] = stop
            self._active_id = execution_id
            thread = threading.Thread(target=self._run, args=(state, stop), daemon=True,
                                      name=f"onegpu-{execution_id}")
            self._thread = thread
            thread.start()
            return state

    def _run(self, state: dict[str, Any], stop: threading.Event) -> None:
        execution_id = state["execution_id"]
        state.update(status="running", phase="generation", updated_at=_now())
        self._write(state)
        self._event(execution_id, {"at": _now(), "event": "generation_started"})

        def report(progress: dict[str, Any]) -> None:
            safe = {key: value for key, value in progress.items()
                    if key in {"completed_candidates", "attempted_requests", "candidate_target", "request_budget"}
                    and isinstance(value, int)}
            state.update(progress=safe, updated_at=_now())
            self._write(state)
            self._event(execution_id, {"at": _now(), "event": "generation_progress", **safe})

        try:
            if state.get("mode") == "prompt_only_sft":
                result = execute_prompt_only_sft_generation(
                    self.workspace, state["bundle_id"], execution_id=execution_id,
                    max_requests=state["max_requests"],
                    progress_callback=report, stop_requested=stop.is_set,
                )
            elif state.get("mode") == "ordinary_sft":
                result = execute_ordinary_sft_generation(
                    self.workspace, state["bundle_id"], execution_id=execution_id,
                    max_requests=state["max_requests"],
                    progress_callback=report, stop_requested=stop.is_set,
                )
            else:
                result = execute_sft_generation(
                    self.workspace, state["bundle_id"], execution_id=execution_id,
                    usable_target=state["usable_target"], max_requests=state["max_requests"],
                    target_candidates=state["candidate_budget"], progress_callback=report,
                    stop_requested=stop.is_set,
                )
            if state.get("stage_plan_id"):
                result["targeted_stage_plan"] = {"plan_id": state["stage_plan_id"],
                                                   "plan_version": state["plan_version"]}
            state.update(status=result["status"], phase="review", result=result,
                         detail=None, updated_at=_now())
            self._event(execution_id, {"at": _now(), "event": "generation_finished",
                                       "status": result["status"],
                                       "candidate_count": result.get("candidate_count", 0),
                                       "usable_records": result.get("usable_records", 0),
                                       "request_count": result.get("request_count", 0)})
        except Exception as exc:  # noqa: BLE001 - provider errors may include secrets; persist only exception type
            state.update(status="failed", phase="failed", result=None,
                         detail=f"生成失败（{type(exc).__name__}）；请检查服务端本地日志", updated_at=_now())
            self._event(execution_id, {"at": _now(), "event": "generation_failed", "error_type": type(exc).__name__})
        finally:
            self._write(state)
            with self._lock:
                self._stop_events.pop(execution_id, None)
                if self._active_id == execution_id:
                    self._active_id = None
                self._thread = None

    def cancel(self, execution_id: str) -> dict[str, Any]:
        state = self._read(execution_id)
        if state is None:
            raise KeyError(execution_id)
        stop = self._stop_events.get(execution_id)
        if not stop or state.get("status") in _TERMINAL:
            raise RuntimeError("任务当前不可取消")
        stop.set()
        self._event(execution_id, {"at": _now(), "event": "cancel_requested",
                                   "note": "正在等待当前 provider 请求结束"})
        return state | {"cancel_requested": True}

    def get(self, execution_id: str) -> dict[str, Any] | None:
        state = self._read(execution_id)
        if state is None:
            return None
        events_path = self._job_dir(execution_id) / "events.jsonl"
        events = []
        if events_path.is_file() and not events_path.is_symlink():
            try:
                lines = events_path.read_text(encoding="utf-8").splitlines()[-200:]
                events = [json.loads(line) for line in lines]
            except (OSError, UnicodeError, json.JSONDecodeError):
                events = []
        return state | {"events": events}

    def register(self, execution_id: str) -> dict[str, Any]:
        """Register only a terminal execution after the user reviewed its output."""
        with self._lock:
            state = self._read(execution_id)
            if state is None:
                raise KeyError(execution_id)
            if state.get("status") not in {"completed", "partial", "cancelled"}:
                raise RuntimeError("生成任务尚未结束，不能登记数据集")
            registration = register_generated_sft_dataset(self.workspace, state)
            state["registered_dataset_id"] = registration["dataset_id"]
            state["registration"] = registration
            state["updated_at"] = _now()
            self._write(state)
            self._event(execution_id, {"at": _now(), "event": "dataset_registered",
                                       "dataset_id": registration["dataset_id"],
                                       "count": registration.get("count"),
                                       "reused": registration.get("reused", False)})
            return state | {"registration": registration}

    def preview(self, execution_id: str, *, limit: int = 3) -> dict[str, Any]:
        state = self._read(execution_id)
        if state is None:
            raise KeyError(execution_id)
        return preview_generated_sft_dataset(self.workspace, state, limit=limit)

    def shutdown(self) -> None:
        with self._lock:
            for stop in self._stop_events.values():
                stop.set()
