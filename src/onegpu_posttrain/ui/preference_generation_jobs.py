"""Persisted, explicit Provider workflow for ordinary/targeted Preference data."""

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

import yaml

from ..data.manifest import load_manifest, resolve_canonical_path, sha256_file
from ..data.ordinary_preference import (
    build_human_reviewed_preference_pairs,
    generate_ordinary_preference_candidates,
    register_human_reviewed_preference_pairs,
)
from ..data.targeted_preference import (
    build_targeted_preference_pairs,
    generate_targeted_preference_candidates,
    register_targeted_preference_pairs,
)
from ..data.targeted_stage_plan import (
    PreferenceTarget,
    load_registered_targeted_stage_plan,
)
from ..synthetic.providers import (
    OpenAICompatibleProvider,
    ProviderError,
    ProviderProfile,
)
from ..synthetic.request_guard import RequestGuard, ensure_no_uncertain_requests
from ..synthetic.secrets import load_local_provider_credential
from .sft_generation_setup import _credential_available

_BUNDLE = re.compile(r"^plan-preference-(ordinary|targeted)-[a-f0-9]{20}$")
_EXECUTION = re.compile(r"^prefgen-[a-f0-9]{20}$")
_TERMINAL = {"completed", "partial", "cancelled", "failed", "interrupted"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _GuardedProvider:
    def __init__(self, guard: RequestGuard):
        self.guard = guard

    def generate(self, profile, request):
        return self.guard.generate(profile, request)


class PreferenceGenerationJobs:
    """Server-owned immutable plans and candidate artifacts; reload never replays calls."""

    def __init__(self, workspace: str | Path, *, provider_factory=None):
        self.workspace = Path(workspace).resolve(strict=True)
        self.root = self.workspace / "artifacts" / "synthetic"
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs_root = self.workspace / "runs" / "preference-generation"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._allow_fake_provider = provider_factory is not None
        self.provider_factory = provider_factory or OpenAICompatibleProvider
        self._lock = threading.RLock()
        self._active: str | None = None
        self._stops: dict[str, threading.Event] = {}
        self._recover()

    def _bundle_dir(self, bundle_id: str) -> Path:
        if not _BUNDLE.fullmatch(bundle_id):
            raise ValueError("Preference bundle ID 无效")
        path = self.root / bundle_id
        if path.is_symlink() or not path.resolve(strict=False).is_relative_to(self.root.resolve()):
            raise ValueError("Preference bundle 路径不可用")
        return path

    def _job_dir(self, execution_id: str) -> Path:
        if not _EXECUTION.fullmatch(execution_id):
            raise ValueError("Preference execution ID 无效")
        path = self.jobs_root / execution_id
        if path.is_symlink() or not path.resolve(strict=False).is_relative_to(self.jobs_root.resolve()):
            raise ValueError("Preference execution 路径不可用")
        return path

    def _read_state(self, execution_id: str) -> dict[str, Any] | None:
        path = self._job_dir(execution_id) / "state.json"
        if path.is_symlink():
            return None
        for attempt in range(6):
            if not path.is_file():
                if attempt < 5:
                    time.sleep(0.005)
                    continue
                return None
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                return value if isinstance(value, dict) and value.get("execution_id") == execution_id else None
            except (OSError, UnicodeError, json.JSONDecodeError):
                if attempt < 5:
                    time.sleep(0.005)
                    continue
                return None
        return None

    def _write_state(self, state: dict[str, Any]) -> None:
        path = self._job_dir(state["execution_id"]) / "state.json"
        temp = path.with_name(f"state-{uuid4().hex}.tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        try:
            for attempt in range(10):
                try:
                    os.replace(temp, path)
                    break
                except PermissionError:
                    if attempt == 9:
                        raise
                    time.sleep(0.01)
        finally:
            temp.unlink(missing_ok=True)

    def _recover(self) -> None:
        for path in self.jobs_root.glob("prefgen-*/state.json"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if isinstance(state, dict) and state.get("status") in {"queued", "running"}:
                state.update(status="interrupted", phase="interrupted",
                             detail="服务重启时生成任务仍在运行；Provider 请求不会自动重放", updated_at=_now())
                try:
                    self._write_state(state)
                except (OSError, ValueError):
                    continue

    def _profile(self, profile_id: str) -> tuple[ProviderProfile, Path]:
        matches = []
        for path in (self.workspace / "configs" / "synthetic").rglob("provider*.yaml"):
            if path.is_symlink() or not path.resolve().is_relative_to(self.workspace):
                continue
            try:
                profile = ProviderProfile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
            except (OSError, TypeError, ValueError, yaml.YAMLError):
                continue
            permitted_adapter = profile.adapter == "openai_compatible" or (self._allow_fake_provider and profile.adapter == "fake")
            if profile.profile_id == profile_id and permitted_adapter and (profile.adapter == "fake" or "example" not in path.name):
                matches.append((profile, path))
        if len(matches) != 1:
            raise ValueError("请从已配置的真实 Provider 中选择一个 profile")
        profile, path = matches[0]
        if profile.adapter == "openai_compatible" and not _credential_available(self.workspace, profile.api_key_env):
            raise ValueError("Provider 凭据未配置")
        return profile, path

    def _targeted_inputs(self, plan_id: str, version: int):
        plan = load_registered_targeted_stage_plan(self.workspace, plan_id, version)
        if plan.target.stage != "preference" or not isinstance(plan.target, PreferenceTarget):
            raise ValueError("定向计划不是 Preference 计划")
        # Resolve by the immutable dataset identity and canonical digest. Only
        # train_seed rows leave this boundary; diagnostic/holdout bodies are not read.
        for candidate in (self.workspace / "data" / "processed").rglob("manifests/*.yaml"):
            if candidate.is_symlink() or not candidate.is_file():
                continue
            try:
                manifest = load_manifest(candidate)
                if manifest.dataset_id != plan.training_seed.dataset_id:
                    continue
                digest = manifest.hashes.get("canonical_records_sha256")
                if digest != plan.training_seed.content_sha256:
                    raise ValueError("定向计划训练种子 dataset hash 与 manifest 不一致")
                canonical = resolve_canonical_path(candidate, manifest)
                rows = [json.loads(line) for line in canonical.read_text(encoding="utf-8").splitlines() if line.strip()]
                seeds = [row for row in rows
                         if (row.get("metadata") or {}).get("data_role", row.get("data_role")) == "train_seed"]
                if not seeds:
                    raise ValueError("定向计划的训练种子数据没有 train_seed 记录")
                return plan, seeds, {"dataset_id": manifest.dataset_id, "content_sha256": digest,
                                     "role": "train_seed"}
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        raise ValueError("找不到与定向计划完全匹配且 hash 校验通过的 train_seed 数据集")

    def prepare(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = payload.get("mode")
        profile_id = payload.get("profile_id")
        if mode not in {"ordinary", "targeted"} or not isinstance(profile_id, str):
            raise ValueError("需明确 Preference 模式及 Provider profile")
        profile, profile_path = self._profile(profile_id)
        budgets = {key: payload.get(key) for key in ("request_budget", "candidate_budget", "max_output_tokens")}
        timeout_seconds = payload.get("timeout_seconds", 120)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in budgets.values()):
            raise ValueError("请求、候选与输出预算须为正整数")
        if budgets["request_budget"] > 1000 or budgets["candidate_budget"] > 500 or budgets["max_output_tokens"] > 4096:
            raise ValueError("Preference 预算超过 Web 工作台安全上限")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not 1 <= timeout_seconds <= 600:
            raise ValueError("Provider 单次超时必须在 1 到 600 秒之间")
        plan_id = payload.get("stage_plan_id")
        version = payload.get("plan_version")
        prompts: list[str] = []
        parent_ref = None
        if mode == "ordinary":
            if set(payload) - {"mode", "profile_id", "prompts", *budgets.keys(), "timeout_seconds"} or not {
                    "mode", "profile_id", "prompts", *budgets.keys()}.issubset(payload):
                raise ValueError("普通 Preference 方案字段不完整或包含不支持的字段")
            prompts = payload["prompts"]
            if not isinstance(prompts, list) or not 1 <= len(prompts) <= 100:
                raise ValueError("普通模式请提供 1..100 道用户提供的问题")
            if any(not isinstance(row, str) or not row.strip() or len(row) > 16000 for row in prompts):
                raise ValueError("问题必须是 1..16000 字符的非空文本")
            if sum(len(row.encode("utf-8")) for row in prompts) > 64 * 1024:
                raise ValueError("普通 Preference 题目总大小不能超过 64 KiB")
            if budgets["candidate_budget"] > budgets["request_budget"]:
                raise ValueError("候选预算不能大于请求预算")
        else:
            if set(payload) - {"mode", "profile_id", "stage_plan_id", "plan_version", *budgets.keys(),
                               "timeout_seconds"} or not {
                    "mode", "profile_id", "stage_plan_id", "plan_version", *budgets.keys()}.issubset(payload):
                raise ValueError("定向 Preference 方案字段不完整或包含不支持的字段")
            if not isinstance(plan_id, str) or isinstance(version, bool) or not isinstance(version, int):
                raise ValueError("定向方案必须提供已登记 plan ID 与 version")
            plan, _, parent_ref = self._targeted_inputs(plan_id, version)
            if (budgets["request_budget"] > plan.budget.max_requests
                    or budgets["candidate_budget"] > plan.budget.max_candidates
                    or budgets["max_output_tokens"] > plan.budget.max_output_tokens
                    or timeout_seconds > plan.budget.timeout_seconds):
                raise ValueError("生成预算不能超过不可变定向计划预算")
        bundle_id = f"plan-preference-{mode}-" + uuid4().hex[:20]
        directory = self._bundle_dir(bundle_id)
        directory.mkdir(parents=True, exist_ok=False)
        plan_payload = {
            "schema_version": "1.0", "bundle_id": bundle_id, "stage": "preference", "mode": mode,
            "profile_id": profile.profile_id, "profile_path": profile_path.relative_to(self.workspace).as_posix(),
            "profile_sha256": sha256_file(profile_path), **budgets,
            "timeout_seconds": float(timeout_seconds),
            "stage_plan_id": plan_id, "plan_version": version, "parent_dataset_ref": parent_ref,
            "prompts": prompts if mode == "ordinary" else None,
            "created_at": _now(),
        }
        (directory / "plan.json").write_text(json.dumps(plan_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {key: value for key, value in plan_payload.items() if key != "prompts"} | {
            "prompt_count": len(prompts) if prompts else parent_ref and len(self._targeted_inputs(plan_id, version)[1]),
            "provider_calls": 0, "status": "prepared_not_executed",
        }

    def _plan(self, bundle_id: str) -> tuple[dict[str, Any], Path, ProviderProfile]:
        directory = self._bundle_dir(bundle_id)
        path = directory / "plan.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("Preference 方案不存在")
        payload = json.loads(path.read_text(encoding="utf-8"))
        profile_path = self.workspace / payload["profile_path"]
        if (profile_path.is_symlink() or not profile_path.is_file()
                or sha256_file(profile_path) != payload.get("profile_sha256")):
            raise ValueError("Provider 配置已变化，请重新准备方案")
        profile = ProviderProfile.model_validate(yaml.safe_load(profile_path.read_text(encoding="utf-8")))
        if (profile.profile_id != payload["profile_id"]
                or (profile.adapter == "openai_compatible" and not _credential_available(self.workspace, profile.api_key_env))):
            raise ValueError("Provider 凭据或 profile 身份不可用")
        return payload, directory, profile

    def preflight(self, bundle_id: str) -> dict[str, Any]:
        plan, directory, profile = self._plan(bundle_id)
        ensure_no_uncertain_requests(directory)
        parent = plan.get("parent_dataset_ref")
        seed_count = None
        if plan["mode"] == "targeted":
            stage_plan, rows, actual_ref = self._targeted_inputs(plan["stage_plan_id"], plan["plan_version"])
            if actual_ref != parent:
                raise ValueError("定向计划、训练种子或其 hash 与已保存方案不匹配")
            seed_count = len(rows)
        return {"status": "preflight_ready", "read_only": True, "provider_calls": 0,
                "bundle_id": bundle_id, "stage": "preference", "mode": plan["mode"],
                "provider_profile_id": profile.profile_id, "provider_model": profile.model,
                "provider_revision": profile.revision, "request_budget": plan["request_budget"],
                "candidate_budget": plan["candidate_budget"], "timeout_seconds": plan.get("timeout_seconds", 120),
                "training_seed_count": seed_count,
                "parent_dataset_ref": parent, "unresolved_provider_requests": []}

    def start(self, bundle_id: str, *, confirm_start: bool) -> dict[str, Any]:
        if confirm_start is not True:
            raise ValueError("Provider 调用前必须明确 confirm_start=true")
        self.preflight(bundle_id)
        plan, directory, _ = self._plan(bundle_id)
        with self._lock:
            if self._active:
                active = self._read_state(self._active)
                if active and active.get("status") not in _TERMINAL:
                    raise RuntimeError("已有 Preference 生成任务正在运行")
            execution_id = "prefgen-" + uuid4().hex[:20]
            self._job_dir(execution_id).mkdir(parents=True, exist_ok=False)
            state = {"execution_id": execution_id, "bundle_id": bundle_id, "stage": "preference",
                     "mode": plan["mode"], "status": "queued", "phase": "queued", "progress": {},
                     "created_at": _now(), "updated_at": _now(), "registration": None}
            self._write_state(state)
            stop = threading.Event()
            self._stops[execution_id] = stop
            self._active = execution_id
            threading.Thread(target=self._run, args=(state, stop), daemon=True, name=execution_id).start()
            return state

    def _run(self, state: dict[str, Any], stop: threading.Event) -> None:
        execution_id, bundle_id = state["execution_id"], state["bundle_id"]
        state.update(status="running", phase="generation", updated_at=_now())
        self._write_state(state)
        request_guard = None
        try:
            plan_payload, bundle_dir, profile = self._plan(bundle_id)
            if profile.adapter == "openai_compatible":
                load_local_provider_credential(profile.api_key_env or "", self.workspace / ".env.local")
            request_guard = RequestGuard(bundle_dir, execution_id, self.provider_factory())
            guarded = _GuardedProvider(request_guard)
            if plan_payload["mode"] == "ordinary":
                report = generate_ordinary_preference_candidates(
                    plan_payload["prompts"], profile=profile, provider=guarded,
                    request_budget=plan_payload["request_budget"], candidate_budget=plan_payload["candidate_budget"],
                    max_output_tokens=plan_payload["max_output_tokens"],
                    timeout_seconds=plan_payload.get("timeout_seconds", 120),
                )
            else:
                stage_plan, seeds, _ = self._targeted_inputs(plan_payload["stage_plan_id"], plan_payload["plan_version"])
                report = generate_targeted_preference_candidates(
                    seeds, plan=stage_plan, profile=profile, provider=guarded,
                    max_requests=plan_payload["request_budget"], max_candidates=plan_payload["candidate_budget"],
                    max_output_tokens=plan_payload["max_output_tokens"],
                )
                report["pair_report"] = build_targeted_preference_pairs(report["candidates"], plan=stage_plan)
            result_path = self._job_dir(execution_id) / "result.json"
            result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            summary = {key: report.get(key) for key in (
                "status", "candidate_count", "request_count", "usable_pairs", "stop_reason", "request_errors",
                "timeout_seconds",
            )}
            if request_guard.provider_diagnostics():
                summary["provider_diagnostics"] = request_guard.provider_diagnostics()
            state.update(status=report.get("status", "partial"), phase="review", result=summary, updated_at=_now())
        except Exception as exc:
            state.update(status="failed", phase="failed", detail=f"Preference 生成失败（{type(exc).__name__}）；请检查本地日志", updated_at=_now())
            if isinstance(exc, ProviderError):
                state["provider_error"] = {"category": exc.category, "status_code": exc.status_code,
                                            "provider_request_id": exc.provider_request_id,
                                            "timeout_seconds": exc.timeout_seconds}
                if exc.category == "timeout":
                    seconds = f"{exc.timeout_seconds:g}" if exc.timeout_seconds is not None else "配置时限"
                    state["detail"] = (f"Preference Provider 请求超过 {seconds} 秒未返回；本次结果未知，可能已到达服务端。"
                                        "请核查 Provider 调用记录；系统未自动重试。")
        finally:
            self._write_state(state)
            with self._lock:
                self._stops.pop(execution_id, None)
                if self._active == execution_id:
                    self._active = None

    def get(self, execution_id: str) -> dict[str, Any] | None:
        return self._read_state(execution_id)

    def plans(self) -> list[dict[str, Any]]:
        rows = []
        for path in self.root.glob("plan-preference-*/plan.json"):
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(self.root.resolve()):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows.append({key: payload.get(key) for key in (
                    "bundle_id", "stage", "mode", "profile_id", "stage_plan_id", "plan_version",
                    "request_budget", "candidate_budget", "created_at", "parent_dataset_ref",
                )})
            except (OSError, TypeError, UnicodeError, json.JSONDecodeError):
                continue
        return sorted(rows, key=lambda row: row.get("created_at") or "", reverse=True)[:100]

    def targeted_plans(self) -> list[dict[str, Any]]:
        rows = []
        base = self.workspace / "artifacts" / "targeted-plans"
        if base.is_symlink() or not base.is_dir():
            return rows
        for path in base.glob("plan-targeted-*/v*/plan.json"):
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(base.resolve()):
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                plan_id, version = raw.get("plan_id"), raw.get("plan_version")
                plan = load_registered_targeted_stage_plan(self.workspace, plan_id, version)
                if plan.target.stage == "preference":
                    rows.append({"plan_id": plan.plan_id, "plan_version": plan.version,
                                 "scenario_id": plan.target.scenario_id,
                                 "training_seed_dataset_id": plan.training_seed.dataset_id,
                                 "training_seed_sha256": plan.training_seed.content_sha256,
                                 "request_budget": plan.budget.max_requests,
                                 "candidate_budget": plan.budget.max_candidates,
                                  "max_output_tokens": plan.budget.max_output_tokens,
                                  "timeout_seconds": plan.budget.timeout_seconds})
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return sorted(rows, key=lambda row: (row["plan_id"], row["plan_version"]))[:200]

    def preview(self, execution_id: str, *, limit: int = 5) -> dict[str, Any]:
        state = self._read_state(execution_id)
        if state is None:
            raise KeyError(execution_id)
        path = self._job_dir(execution_id) / "result.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("任务尚无候选结果")
        report = json.loads(path.read_text(encoding="utf-8"))
        candidates = report.get("candidates", [])
        return {"execution_id": execution_id, "stage": "preference", "mode": state["mode"],
                "candidate_count": len(candidates), "ordering_evidence_status": report.get("ordering_evidence_status", "not_independently_verified"),
                "trainable": False, "candidates": candidates[:limit],
                "pair_report": report.get("pair_report")}

    def review(self, execution_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        state = self._read_state(execution_id)
        if state is None:
            raise KeyError(execution_id)
        if state["mode"] != "ordinary" or state["status"] not in {"completed", "partial"}:
            raise ValueError("只有已完成的普通 Preference 候选支持人工排序")
        if set(payload) != {"reviewer_id", "reviews"} or not isinstance(payload.get("reviews"), list):
            raise ValueError("审核仅接受审阅者标识和 chosen/rejected 候选 ID 列表")
        result_path = self._job_dir(execution_id) / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        report = build_human_reviewed_preference_pairs(result.get("candidates", []), payload["reviews"], reviewer_id=payload["reviewer_id"])
        (self._job_dir(execution_id) / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        state["review"] = {key: report.get(key) for key in ("status", "candidate_count", "review_count", "usable_pairs", "rejections")}
        state["updated_at"] = _now()
        self._write_state(state)
        return state

    def register(self, execution_id: str) -> dict[str, Any]:
        state = self._read_state(execution_id)
        if state is None:
            raise KeyError(execution_id)
        if state.get("registration"):
            return state
        if state["status"] not in {"completed", "partial"}:
            raise ValueError("生成任务尚未完成")
        plan_payload, _, profile = self._plan(state["bundle_id"])
        if state["mode"] == "ordinary":
            review_path = self._job_dir(execution_id) / "review.json"
            if not review_path.is_file() or review_path.is_symlink():
                raise ValueError("普通 Preference 候选须先审核排序")
            report = json.loads(review_path.read_text(encoding="utf-8"))
            registration = register_human_reviewed_preference_pairs(
                report, workspace=self.workspace, output_root=self.workspace / "data" / "processed" / "generated",
                provider_profile_id=profile.profile_id, provider_model=profile.model, provider_revision=profile.revision,
                provider_adapter=profile.adapter,
                generation_bundle_id=state["bundle_id"], generation_execution_id=execution_id,
            )
        else:
            result = json.loads((self._job_dir(execution_id) / "result.json").read_text(encoding="utf-8"))
            plan, _, parent_ref = self._targeted_inputs(plan_payload["stage_plan_id"], plan_payload["plan_version"])
            report = result["pair_report"]
            registration = register_targeted_preference_pairs(
                report, plan=plan, workspace=self.workspace,
                output_root=self.workspace / "data" / "processed" / "generated", provider_profile_id=profile.profile_id,
                provider_model=profile.model, provider_revision=profile.revision,
                provider_adapter=profile.adapter,
                generation_bundle_id=state["bundle_id"], generation_execution_id=execution_id,
            )
            if registration.get("parent_dataset_refs") != [parent_ref]:
                raise ValueError("登记结果的父数据集 ID/hash 与定向计划不一致")
        state["registration"] = registration
        state["updated_at"] = _now()
        self._write_state(state)
        return state

    def shutdown(self) -> None:
        for stop in self._stops.values():
            stop.set()
