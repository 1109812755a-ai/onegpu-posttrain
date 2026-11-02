"""Persisted GRPO prompt-only candidate generation jobs.

These jobs intentionally stop before dataset registration: the model output has
no trusted reference and cannot satisfy the GRPO training contract by itself.
"""

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

from ..data.manifest import (
    load_manifest,
    resolve_canonical_path,
    sha256_file,
    validate_manifest_evidence,
)
from ..data.targeted_grpo import ExecutableRewardContract
from ..data.targeted_stage_plan import GRPOTarget, load_registered_targeted_stage_plan
from ..synthetic.grpo_generation import generate_grpo_prompt_candidates
from ..synthetic.providers import (
    OpenAICompatibleProvider,
    ProviderError,
    ProviderProfile,
)
from ..synthetic.request_guard import RequestGuard, ensure_no_uncertain_requests
from ..synthetic.secrets import load_local_provider_credential
from .sft_generation_setup import _credential_available

_BUNDLE = re.compile(r"^plan-grpo-(ordinary|targeted)-[a-f0-9]{20}$")
_EXECUTION = re.compile(r"^grpo-gen-[a-f0-9]{20}$")
_TERMINAL = {"completed", "partial", "failed", "interrupted", "cancelled"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _GuardedProvider:
    def __init__(self, guard: RequestGuard):
        self.guard = guard

    def generate(self, profile, request):
        return self.guard.generate(profile, request)


class GRPOGenerationJobs:
    """Server-owned plans and resumable read-only status for prompt candidates."""

    def __init__(self, workspace: str | Path, *, provider_factory=None):
        self.workspace = Path(workspace).resolve(strict=True)
        self.root = self.workspace / "artifacts" / "synthetic"
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs_root = self.workspace / "runs" / "grpo-generation"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._allow_fake = provider_factory is not None
        self.provider_factory = provider_factory or OpenAICompatibleProvider
        self._lock = threading.RLock()
        self._active: str | None = None
        self._stops: dict[str, threading.Event] = {}
        self._recover()

    def _bundle_dir(self, bundle_id: str) -> Path:
        if not isinstance(bundle_id, str) or not _BUNDLE.fullmatch(bundle_id):
            raise ValueError("GRPO bundle ID 无效")
        path = self.root / bundle_id
        if path.is_symlink() or not path.resolve(strict=False).is_relative_to(self.root.resolve()):
            raise ValueError("GRPO bundle 路径不可用")
        return path

    def _job_dir(self, execution_id: str) -> Path:
        if not isinstance(execution_id, str) or not _EXECUTION.fullmatch(execution_id):
            raise ValueError("GRPO execution ID 无效")
        path = self.jobs_root / execution_id
        if path.is_symlink() or not path.resolve(strict=False).is_relative_to(self.jobs_root.resolve()):
            raise ValueError("GRPO execution 路径不可用")
        return path

    def _read_state(self, execution_id: str) -> dict[str, Any] | None:
        path = self._job_dir(execution_id) / "state.json"
        if path.is_symlink() or not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) and value.get("execution_id") == execution_id else None
        except (OSError, UnicodeError, json.JSONDecodeError):
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
        for path in self.jobs_root.glob("grpo-gen-*/state.json"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(state, dict) and state.get("status") in {"queued", "running"}:
                    state.update(status="interrupted", phase="interrupted",
                                 detail="服务重启时任务仍在运行；未完成的 Provider 请求不会自动重放",
                                 updated_at=_now())
                    self._write_state(state)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
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
            permitted = profile.adapter == "openai_compatible" or (self._allow_fake and profile.adapter == "fake")
            if profile.profile_id == profile_id and permitted and (profile.adapter == "fake" or "example" not in path.name):
                matches.append((profile, path))
        if len(matches) != 1:
            raise ValueError("请选择唯一且可用的 Provider profile")
        profile, path = matches[0]
        if profile.adapter == "openai_compatible" and not _credential_available(self.workspace, profile.api_key_env):
            raise ValueError("Provider 凭据未配置")
        return profile, path

    def targeted_plans(self) -> list[dict[str, Any]]:
        base = self.workspace / "artifacts" / "targeted-plans"
        if base.is_symlink() or not base.is_dir():
            return []
        rows = []
        for path in base.glob("plan-targeted-*/v*/plan.json"):
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(base.resolve()):
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                plan = load_registered_targeted_stage_plan(self.workspace, raw["plan_id"], raw["plan_version"])
                if plan.target.stage == "grpo" and isinstance(plan.target, GRPOTarget):
                    rows.append({"plan_id": plan.plan_id, "plan_version": plan.version,
                                 "scenario_id": plan.target.scenario_id,
                                 "training_seed_dataset_id": plan.training_seed.dataset_id,
                                 "training_seed_sha256": plan.training_seed.content_sha256,
                                 "max_requests": plan.budget.max_requests,
                                 "max_candidates": plan.budget.max_candidates,
                                 "max_output_tokens": plan.budget.max_output_tokens,
                                 "timeout_seconds": plan.budget.timeout_seconds})
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return sorted(rows, key=lambda row: (row["plan_id"], row["plan_version"]))[:200]

    def _targeted_inputs(self, plan_id: str, version: int):
        plan = load_registered_targeted_stage_plan(self.workspace, plan_id, version)
        if plan.target.stage != "grpo" or not isinstance(plan.target, GRPOTarget):
            raise ValueError("定向方案不是 GRPO 计划")
        for path in (self.workspace / "data" / "processed").rglob("manifests/*.yaml"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                manifest = load_manifest(path)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if manifest.dataset_id != plan.training_seed.dataset_id:
                continue
            if manifest.lineage is None or manifest.processing_report is None:
                raise ValueError("GRPO 训练种子必须有登记 lineage 与 processing report")
            evidence_errors = validate_manifest_evidence(path)
            if evidence_errors:
                raise ValueError("GRPO 训练种子登记证据无效：" + "; ".join(evidence_errors[:5]))
            try:
                digest = manifest.hashes.get("canonical_records_sha256")
                if digest != plan.training_seed.content_sha256:
                    raise ValueError("GRPO 训练种子 hash 与计划不一致")
                canonical = resolve_canonical_path(path, manifest)
                if sha256_file(canonical) != digest:
                    raise ValueError("GRPO 训练种子 canonical hash 无效")
                seeds = []
                for line in canonical.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    metadata = row.get("metadata") if isinstance(row, dict) else None
                    if isinstance(metadata, dict) and metadata.get("data_role") == "train_seed":
                        seeds.append(row)
                if not seeds:
                    raise ValueError("计划训练种子中没有 train_seed 记录")
                return plan, seeds, {"dataset_id": manifest.dataset_id, "content_sha256": digest,
                                     "role": "train_seed"}
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"GRPO 训练种子数据无效：{exc}") from exc
        raise ValueError("找不到与 GRPO 定向计划 ID/hash 完全匹配的 train_seed 数据集")

    def prepare(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode, profile_id = payload.get("mode"), payload.get("profile_id")
        if mode not in {"ordinary", "targeted"} or not isinstance(profile_id, str):
            raise ValueError("需明确普通/定向模式及 Provider")
        profile, profile_path = self._profile(profile_id)
        request_budget, candidate_budget, max_tokens = (payload.get(key) for key in
                                                         ("request_budget", "candidate_budget", "max_output_tokens"))
        timeout_seconds = payload.get("timeout_seconds", 120)
        for name, value in (("request_budget", request_budget), ("candidate_budget", candidate_budget),
                            ("max_output_tokens", max_tokens)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} 必须是正整数")
        if request_budget > 100 or candidate_budget > 100 or max_tokens > 2048 or candidate_budget > request_budget:
            raise ValueError("GRPO Web 候选预算超限或候选数大于请求数")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not 1 <= timeout_seconds <= 600:
            raise ValueError("Provider 单次超时必须在 1 到 600 秒之间")
        plan_id = payload.get("stage_plan_id")
        version = payload.get("plan_version")
        parent_ref = None
        seeds = []
        prompts = []
        if mode == "ordinary":
            if (set(payload) - {"mode", "profile_id", "prompts", "request_budget", "candidate_budget",
                                "max_output_tokens", "timeout_seconds"}
                    or not {"mode", "profile_id", "prompts", "request_budget", "candidate_budget",
                            "max_output_tokens"}.issubset(payload)):
                raise ValueError("普通 GRPO 方案字段不完整")
            prompts = payload["prompts"]
            if not isinstance(prompts, list) or not 1 <= len(prompts) <= 50 or any(
                    not isinstance(value, str) or not value.strip() or len(value) > 8000 for value in prompts):
                raise ValueError("普通模式需提供 1 到 50 个非空数学主题/训练种子提示")
            if sum(len(value.encode("utf-8")) for value in prompts) > 32 * 1024:
                raise ValueError("输入提示总大小不能超过 32 KiB")
            seeds = [{"id": f"ordinary-topic-{i + 1}", "problem": prompt,
                      "metadata": {"problem_id": f"ordinary-topic-{i + 1}"}}
                     for i, prompt in enumerate(prompts)]
        else:
            if (set(payload) - {"mode", "profile_id", "stage_plan_id", "plan_version", "request_budget",
                                "candidate_budget", "max_output_tokens", "timeout_seconds"}
                    or not {"mode", "profile_id", "stage_plan_id", "plan_version", "request_budget",
                            "candidate_budget", "max_output_tokens"}.issubset(payload)):
                raise ValueError("定向 GRPO 方案字段不完整")
            if not isinstance(plan_id, str) or isinstance(version, bool) or not isinstance(version, int):
                raise ValueError("定向方案需提供 plan ID/version")
            plan, seeds, parent_ref = self._targeted_inputs(plan_id, version)
            contract = ExecutableRewardContract.create()
            evidence = plan.target.reward_contract
            if not evidence or evidence.evidence_id != contract.contract_id or evidence.content_sha256 != contract.protocol_sha256:
                raise ValueError("定向计划未绑定当前可执行 reward contract")
            if (request_budget > plan.budget.max_requests or candidate_budget > plan.budget.max_candidates
                    or max_tokens > plan.budget.max_output_tokens or timeout_seconds > plan.budget.timeout_seconds):
                raise ValueError("候选预算不能超过不可变 GRPO 定向计划")
        bundle_id = f"plan-grpo-{mode}-" + uuid4().hex[:20]
        directory = self._bundle_dir(bundle_id)
        directory.mkdir(parents=True, exist_ok=False)
        plan_payload = {"schema_version": "1.0", "bundle_id": bundle_id, "stage": "grpo", "mode": mode,
                        "profile_id": profile.profile_id, "profile_path": profile_path.relative_to(self.workspace).as_posix(),
                        "profile_sha256": sha256_file(profile_path), "request_budget": request_budget,
                        "candidate_budget": candidate_budget, "max_output_tokens": max_tokens,
                        "timeout_seconds": float(timeout_seconds),
                        "stage_plan_id": plan_id, "plan_version": version, "parent_dataset_ref": parent_ref,
                        "prompts": prompts if mode == "ordinary" else None, "created_at": _now()}
        (directory / "plan.json").write_text(json.dumps(plan_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {key: value for key, value in plan_payload.items() if key != "prompts"} | {
            "input_count": len(prompts) if prompts else len(seeds), "provider_calls": 0,
            "reference_status": "missing_until_independently_bound", "trainable": False,
            "status": "prepared_not_executed"}

    def _load_plan(self, bundle_id: str):
        directory = self._bundle_dir(bundle_id)
        path = directory / "plan.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("GRPO 生成方案不存在")
        plan = json.loads(path.read_text(encoding="utf-8"))
        profile_path = self.workspace / plan["profile_path"]
        if profile_path.is_symlink() or not profile_path.is_file() or sha256_file(profile_path) != plan["profile_sha256"]:
            raise ValueError("Provider profile 已变化，请重新准备")
        profile = ProviderProfile.model_validate(yaml.safe_load(profile_path.read_text(encoding="utf-8")))
        if profile.profile_id != plan["profile_id"]:
            raise ValueError("Provider profile 身份变化")
        return plan, directory, profile

    def preflight(self, bundle_id: str) -> dict[str, Any]:
        plan, directory, profile = self._load_plan(bundle_id)
        ensure_no_uncertain_requests(directory)
        seed_count = None
        if plan["mode"] == "targeted":
            stage_plan, seeds, parent = self._targeted_inputs(plan["stage_plan_id"], plan["plan_version"])
            if parent != plan["parent_dataset_ref"]:
                raise ValueError("定向计划训练种子 ID/hash 与已准备方案不一致")
            contract = ExecutableRewardContract.create()
            evidence = stage_plan.target.reward_contract
            if not evidence or (evidence.evidence_id, evidence.content_sha256) != (contract.contract_id, contract.protocol_sha256):
                raise ValueError("当前 reward/reference 契约与计划不一致")
            seed_count = len(seeds)
        return {"status": "preflight_ready", "read_only": True, "provider_calls": 0,
                "bundle_id": bundle_id, "stage": "grpo", "mode": plan["mode"],
                "provider_profile_id": profile.profile_id, "provider_model": profile.model,
                "provider_revision": profile.revision, "request_budget": plan["request_budget"],
                "candidate_budget": plan["candidate_budget"], "timeout_seconds": plan.get("timeout_seconds", 120),
                "training_seed_count": seed_count,
                "parent_dataset_ref": plan["parent_dataset_ref"], "trainable": False,
                "blockers": ["模型生成的问题尚无可信 reference；候选预览不会登记为可训练 GRPO 数据"]}

    def start(self, bundle_id: str, *, confirm_start: bool) -> dict[str, Any]:
        if confirm_start is not True:
            raise ValueError("Provider 调用前须明确确认")
        self.preflight(bundle_id)
        plan, _, _ = self._load_plan(bundle_id)
        with self._lock:
            if self._active:
                active = self._read_state(self._active)
                if active and active.get("status") not in _TERMINAL:
                    raise RuntimeError("已有 GRPO 生成任务正在运行")
            execution_id = "grpo-gen-" + uuid4().hex[:20]
            self._job_dir(execution_id).mkdir(parents=True, exist_ok=False)
            state = {"execution_id": execution_id, "bundle_id": bundle_id, "stage": "grpo",
                     "mode": plan["mode"], "status": "queued", "phase": "queued", "progress": {},
                     "provider_profile_id": plan["profile_id"],
                     "targeted_plan_id": plan.get("stage_plan_id"),
                     "targeted_plan_version": plan.get("plan_version"),
                     "source_dataset_id": (plan.get("parent_dataset_ref") or {}).get("dataset_id"),
                     "source_dataset_sha256": (plan.get("parent_dataset_ref") or {}).get("content_sha256"),
                     "created_at": _now(), "updated_at": _now(), "registration": None}
            self._write_state(state)
            self._active = execution_id
            stop = threading.Event()
            self._stops[execution_id] = stop
            threading.Thread(target=self._run, args=(state, stop), daemon=True, name=execution_id).start()
            return state

    def _run(self, state: dict[str, Any], stop: threading.Event) -> None:
        state.update(status="running", phase="generation", updated_at=_now())
        self._write_state(state)
        request_guard = None
        try:
            plan, directory, profile = self._load_plan(state["bundle_id"])
            if profile.adapter == "openai_compatible":
                load_local_provider_credential(profile.api_key_env or "", self.workspace / ".env.local")
            seeds = []
            targeted_plan = None
            if plan["mode"] == "ordinary":
                seeds = [{"id": f"ordinary-topic-{i + 1}", "problem": prompt,
                          "metadata": {"problem_id": f"ordinary-topic-{i + 1}"}}
                         for i, prompt in enumerate(plan["prompts"])]
            else:
                targeted_plan, seeds, parent = self._targeted_inputs(plan["stage_plan_id"], plan["plan_version"])
                if parent != plan["parent_dataset_ref"]:
                    raise ValueError("training_seed_identity_changed")
            contract = ExecutableRewardContract.create()
            request_guard = RequestGuard(directory, state["execution_id"], self.provider_factory())
            report = generate_grpo_prompt_candidates(
                seeds, mode=plan["mode"], profile=profile,
                provider=_GuardedProvider(request_guard),
                contract=contract, request_budget=plan["request_budget"],
                candidate_budget=plan["candidate_budget"], max_output_tokens=plan["max_output_tokens"],
                timeout_seconds=plan.get("timeout_seconds", 120),
                targeted_plan=targeted_plan, stop_requested=stop.is_set,
                progress_callback=lambda value: self._progress(state, value),
            )
            result_path = self._job_dir(state["execution_id"]) / "result.json"
            result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            summary = {key: report.get(key) for key in (
                "status", "candidate_count", "request_count", "request_budget", "request_errors",
                "timeout_seconds",
                "provider_identity_status", "provider_profile_id", "provider_model",
                "provider_revision", "model_identity_status", "targeted_plan_id",
                "targeted_plan_version", "reward_contract_id", "reward_contract_protocol_sha256",
            )}
            summary["generation_artifacts"] = {"candidate_preview": "result.json"}
            if request_guard.provider_diagnostics():
                summary["provider_diagnostics"] = request_guard.provider_diagnostics()
            summary["reference_status"] = "missing_until_independently_bound"
            summary["trainable"] = False
            state.update(status=report["status"], phase="review", result=summary,
                         progress={"candidates": report["candidate_count"], "target": plan["candidate_budget"]},
                         updated_at=_now())
        except Exception as exc:
            state.update(status="failed", phase="failed",
                         detail=f"GRPO 候选生成失败（{type(exc).__name__}）；请检查本地服务日志",
                         updated_at=_now())
            if isinstance(exc, ProviderError):
                state["provider_error"] = {"category": exc.category, "status_code": exc.status_code,
                                            "provider_request_id": exc.provider_request_id,
                                            "timeout_seconds": exc.timeout_seconds}
                if exc.category == "timeout":
                    seconds = f"{exc.timeout_seconds:g}" if exc.timeout_seconds is not None else "配置时限"
                    state["detail"] = (f"GRPO Provider 请求超过 {seconds} 秒未返回；本次结果未知，可能已到达服务端。"
                                        "请核查 Provider 调用记录；系统未自动重试。")
                elif exc.status_code is not None:
                    state["detail"] = f"GRPO Provider 返回 HTTP {exc.status_code}（{exc.category}）；请检查凭据、权限或额度。"
        finally:
            self._write_state(state)
            with self._lock:
                self._stops.pop(state["execution_id"], None)
                if self._active == state["execution_id"]:
                    self._active = None

    def _progress(self, state: dict[str, Any], value: dict[str, Any]) -> None:
        with self._lock:
            state.update(progress={"requests": value["attempted_requests"],
                                  "candidates": value["accepted_candidates"],
                                  "target": value["candidate_budget"]}, updated_at=_now())
            self._write_state(state)

    def cancel(self, execution_id: str) -> dict[str, Any]:
        state = self._read_state(execution_id)
        if state is None:
            raise KeyError(execution_id)
        with self._lock:
            stop = self._stops.get(execution_id)
            if stop is None or state.get("status") not in {"queued", "running"}:
                raise ValueError("GRPO 任务当前不可取消")
            stop.set()
            state.update(cancel_requested=True, updated_at=_now())
            self._write_state(state)
            return state

    def get(self, execution_id: str) -> dict[str, Any] | None:
        return self._read_state(execution_id)

    def preview(self, execution_id: str) -> dict[str, Any]:
        state = self._read_state(execution_id)
        if state is None:
            raise KeyError(execution_id)
        path = self._job_dir(execution_id) / "result.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("任务尚无候选结果")
        result = json.loads(path.read_text(encoding="utf-8"))
        return {"execution_id": execution_id, "stage": "grpo", "mode": state["mode"],
                "status": state["status"], "candidate_count": result["candidate_count"],
                "trainable": False, "registration_allowed": False,
                "reference_status": "missing_until_independently_bound",
                "candidates": result["candidates"][:100],
                "rejections": result["rejections"], "request_errors": result["request_errors"],
                "limitations": result["limitations"]}

    def shutdown(self) -> None:
        # Running tasks are recovered as interrupted on next service start and
        # RequestGuard will prevent unsafe replay of ambiguous remote requests.
        return None
