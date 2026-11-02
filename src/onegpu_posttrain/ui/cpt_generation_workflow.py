"""Read-only preparation and explicit-review registration helpers for CPT generation."""

from __future__ import annotations

import json
import re
import threading
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from ..data.benchmarks import benchmark_index_from_jsonl
from ..data.dataset_import import register_jsonl
from ..data.manifest import load_manifest, resolve_canonical_path, sha256_file
from ..data.registry import DataSource
from ..data.targeted_stage_plan import CPTTarget, load_registered_targeted_stage_plan
from ..synthetic.cpt_generation import generate_cpt_candidates
from ..synthetic.providers import (
    FakeProvider,
    OpenAICompatibleProvider,
    ProviderError,
    ProviderProfile,
)
from ..synthetic.request_guard import (
    RequestGuard,
    acknowledge_uncertain_attempt,
    ensure_no_uncertain_requests,
    unresolved_attempts,
)
from ..synthetic.secrets import load_local_provider_credential
from .sft_generation_setup import _credential_available

_ID = re.compile(r"^cptgen-[a-f0-9]{20}$")


def _profile(root: Path, profile_id: str | None) -> tuple[ProviderProfile, Path]:
    matches: list[tuple[ProviderProfile, Path]] = []
    for path in (root / "configs" / "synthetic").rglob("provider*.yaml"):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            continue
        try:
            profile = ProviderProfile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            continue
        if profile.adapter == "fake" or "example" in path.name:
            continue
        if profile_id is None or profile.profile_id == profile_id:
            matches.append((profile, path))
    if len(matches) != 1:
        raise ValueError("请明确选择一个有效 Provider profile；本地凭据由启动时检查")
    return matches[0]


def _load_plan(root: Path, plan_id: str, version: int):
    plan = load_registered_targeted_stage_plan(root, plan_id, version)
    if plan.target.stage != "cpt" or not isinstance(plan.target, CPTTarget):
        raise ValueError("定向方案不是 CPT 方案")
    return plan


def prepare_cpt_generation(
    workspace: str | Path,
    *,
    mode: str,
    domain: str,
    topic: str,
    target_token_budget: int,
    request_budget: int,
    candidate_budget: int,
    max_output_tokens: int,
    profile_id: str,
    timeout_seconds: float = 120,
    plan_id: str | None = None,
    plan_version: int | None = None,
) -> dict[str, Any]:
    """Persist a bounded immutable plan; preparation never calls the Provider."""
    root = Path(workspace).resolve(strict=True)
    if mode not in {"ordinary", "targeted"}:
        raise ValueError("CPT generation mode must be ordinary or targeted")
    if (not isinstance(domain, str) or not domain.strip() or len(domain) > 120
            or not isinstance(topic, str) or not topic.strip() or len(topic) > 160):
        raise ValueError("CPT 领域最多 120 字符，子主题最多 160 字符，且不能为空")
    for name, value in (("target_token_budget", target_token_budget), ("request_budget", request_budget),
                        ("candidate_budget", candidate_budget), ("max_output_tokens", max_output_tokens)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} 必须是正整数")
    if request_budget > 10_000 or candidate_budget > 10_000 or max_output_tokens > 8192:
        raise ValueError("CPT 请求、候选或输出预算超过安全上限")
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not 1 <= timeout_seconds <= 600):
        raise ValueError("Provider 单次超时必须在 1 到 600 秒之间")
    if mode == "targeted":
        if not isinstance(plan_id, str) or not isinstance(plan_version, int) or isinstance(plan_version, bool):
            raise ValueError("定向 CPT 必须提供已登记计划 ID 和版本")
        plan = _load_plan(root, plan_id, plan_version)
        _targeted_seed_rows(root, plan_id, plan_version)
        if domain.strip() != plan.target.domain or topic.strip() not in plan.target.subtopics:
            raise ValueError("定向主题必须与不可变 CPT 计划中的 domain/subtopics 一致")
        if target_token_budget != plan.budget.expected_usable_count:
            raise ValueError("定向 token 预算必须匹配不可变 CPT 计划")
        if request_budget > plan.budget.max_requests or candidate_budget > plan.budget.max_candidates:
            raise ValueError("请求/候选预算不能超过定向计划上限")
        if max_output_tokens > plan.budget.max_output_tokens:
            raise ValueError("单次输出预算不能超过定向计划上限")
    elif plan_id is not None or plan_version is not None:
        raise ValueError("普通 CPT 生成不能携带定向计划身份")
    profile, profile_path = _profile(root, profile_id)
    execution_plan_id = "cptgen-" + uuid4().hex[:20]
    artifact_root = root / "artifacts" / "synthetic" / execution_plan_id
    artifact_root.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": "1.0", "bundle_id": execution_plan_id, "stage": "cpt", "mode": mode,
        "domain": domain.strip(), "topic": topic.strip(), "target_token_budget": target_token_budget,
        "request_budget": request_budget, "candidate_budget": candidate_budget,
        "max_output_tokens": max_output_tokens, "timeout_seconds": float(timeout_seconds),
        "profile_id": profile.profile_id,
        "profile_sha256": sha256_file(profile_path), "profile_path": str(profile_path.relative_to(root)),
        "targeted_plan_id": plan_id, "targeted_plan_version": plan_version,
        "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    plan_path = artifact_root / "cpt_generation_plan.json"
    plan_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**payload, "provider_calls": 0, "network_used": False, "status": "prepared_not_executed"}


def preflight_cpt_generation(workspace: str | Path, bundle_id: str, *, tokenizer: Any = None) -> dict[str, Any]:
    root = Path(workspace).resolve(strict=True)
    if not _ID.fullmatch(bundle_id):
        raise ValueError("CPT generation plan ID 无效")
    artifact_root = root / "artifacts" / "synthetic" / bundle_id
    if artifact_root.is_symlink() or not artifact_root.resolve(strict=True).is_relative_to(root):
        raise ValueError("CPT 生成方案路径不可用")
    plan_path = artifact_root / "cpt_generation_plan.json"
    if plan_path.is_symlink() or not plan_path.is_file():
        raise ValueError("CPT generation plan 不存在")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    profile_path = root / plan["profile_path"]
    if not profile_path.is_file() or profile_path.is_symlink() or sha256_file(profile_path) != plan["profile_sha256"]:
        raise ValueError("Provider 配置已变化，请重新准备方案")
    profile = ProviderProfile.model_validate(yaml.safe_load(profile_path.read_text(encoding="utf-8")))
    if profile.profile_id != plan["profile_id"]:
        raise ValueError("Provider profile 身份与方案不匹配")
    if profile.adapter != "openai_compatible" or not _credential_available(root, profile.api_key_env):
        raise ValueError("Provider 凭据未配置或 profile 不是受支持的真实 Provider")
    if plan["mode"] == "targeted":
        registered = _load_plan(root, plan["targeted_plan_id"], plan["targeted_plan_version"])
        if (registered.target.domain != plan["domain"] or plan["topic"] not in registered.target.subtopics
                or registered.budget.expected_usable_count != plan["target_token_budget"]):
            raise ValueError("定向计划身份或预算发生变化")
        seed_rows, _ = _targeted_seed_rows(root, plan["targeted_plan_id"], plan["targeted_plan_version"])
    else:
        seed_rows = []
    if tokenizer is None:
        from ..data.tokenization import HuggingFaceTokenizer
        model_cfg = yaml.safe_load((root / "configs" / "model" / "qwen3-0.6b-base.yaml").read_text(encoding="utf-8"))
        model = model_cfg["model"]
        tokenizer = HuggingFaceTokenizer(model["repo"], model.get("revision"), local_files_only=True)
    benchmark_path = root / "data" / "examples" / "policy" / "benchmark.jsonl"
    if not benchmark_path.is_file() or benchmark_path.is_symlink():
        raise ValueError("本地污染筛查索引不存在；请先登记带固定版本和摘要的评测数据")
    benchmark_hash = sha256_file(benchmark_path)
    index = benchmark_index_from_jsonl([benchmark_path])
    uncertain = unresolved_attempts(artifact_root)
    return {
        "status": "blocked" if uncertain else "preflight_ready", "read_only": True, "provider_calls": 0,
        "bundle_id": bundle_id, "stage": "cpt", "mode": plan["mode"],
        "profile_id": profile.profile_id, "model": profile.model,
        "tokenizer_id": tokenizer.name, "benchmark_index_id": "workspace-example-policy-v1",
        "benchmark_index_sha256": benchmark_hash, "benchmark_record_count": 1,
        "contamination_scope": "example_fixture_only",
        "limitations": ["僅對 1 條本地示例基準做精确/近似筛查，不构成正式污染排除证明。",
                        "生成文本事实正确性仍未独立验证。"],
        "target_token_budget": plan["target_token_budget"],
        "eligible_train_seed_count": len(seed_rows),
        "timeout_seconds": plan.get("timeout_seconds", 120),
        "request_budget": plan["request_budget"], "candidate_budget": plan["candidate_budget"],
        "unresolved_provider_attempts": uncertain,
        "blockers": (["存在结果未知的 Provider 请求；需先核查请求记录，不能自动重试"] if uncertain else []),
        "can_start_exploratory_generation": not bool(uncertain),
    }


def _targeted_seed_rows(root: Path, plan_id: str, version: int) -> tuple[list[dict[str, Any]], dict[str, str]]:
    plan = _load_plan(root, plan_id, version)
    # Resolve the exact identity by scanning registered manifests; never select by display name.
    for candidate in (root / "data" / "processed").rglob("*.yaml"):
        if candidate.parent.name != "manifests":
            continue
        if candidate.is_symlink():
            continue
        try:
            manifest = load_manifest(candidate)
            if manifest.dataset_id != plan.training_seed.dataset_id:
                continue
            content_hash = manifest.hashes.get("canonical_records_sha256")
            if content_hash != plan.training_seed.content_sha256:
                raise ValueError("定向计划训练种子 dataset hash 与已登记 manifest 不一致")
            canonical = resolve_canonical_path(candidate, manifest)
            rows = [json.loads(line) for line in canonical.read_text(encoding="utf-8").splitlines() if line.strip()]
            seeds = [row for row in rows if row.get("metadata", {}).get("data_role") == "train_seed"]
            if not seeds:
                raise ValueError("定向 CPT 种子数据中没有 train_seed 记录")
            return seeds, {"dataset_id": manifest.dataset_id, "content_sha256": content_hash}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    raise ValueError("找不到与定向计划 ID 完全匹配且 hash 校验通过的训练种子数据集")


def execute_cpt_generation(
    workspace: str | Path,
    bundle_id: str,
    *,
    execution_id: str,
    provider: Any = None,
    tokenizer: Any = None,
    benchmark_index: Any = None,
    progress_callback=None,
    stop_requested=None,
) -> dict[str, Any]:
    root = Path(workspace).resolve(strict=True)
    readiness = preflight_cpt_generation(root, bundle_id, tokenizer=tokenizer)
    artifact_root = root / "artifacts" / "synthetic" / bundle_id
    plan = json.loads((artifact_root / "cpt_generation_plan.json").read_text(encoding="utf-8"))
    profile_path = root / plan["profile_path"]
    profile = ProviderProfile.model_validate(yaml.safe_load(profile_path.read_text(encoding="utf-8")))
    if tokenizer is None:
        from ..data.tokenization import HuggingFaceTokenizer
        model = yaml.safe_load((root / "configs" / "model" / "qwen3-0.6b-base.yaml").read_text(encoding="utf-8"))["model"]
        tokenizer = HuggingFaceTokenizer(model["repo"], model.get("revision"), local_files_only=True)
    benchmark_path = root / "data" / "examples" / "policy" / "benchmark.jsonl"
    index = benchmark_index or benchmark_index_from_jsonl([benchmark_path])
    if provider is None:
        if profile.adapter == "openai_compatible":
            load_local_provider_credential(profile.api_key_env or "", root / ".env.local")
        provider = FakeProvider() if profile.adapter == "fake" else OpenAICompatibleProvider()
    parent_refs: list[dict[str, str]] = []
    if plan["mode"] == "targeted":
        seeds, parent_ref = _targeted_seed_rows(root, plan["targeted_plan_id"], plan["targeted_plan_version"])
        parent_refs = [parent_ref]
    else:
        seeds = []
    guarded_provider = RequestGuard(artifact_root, execution_id, provider)
    result = generate_cpt_candidates(
        provider=guarded_provider, profile=profile, domain=plan["domain"], topic=plan["topic"],
        target_token_budget=plan["target_token_budget"], request_budget=plan["request_budget"],
        candidate_budget=plan["candidate_budget"], max_output_tokens=plan["max_output_tokens"],
        timeout_seconds=plan.get("timeout_seconds", 120), tokenizer_id=tokenizer.name, count_tokens=tokenizer.count_text,
        benchmark_index=index, benchmark_index_id=readiness["benchmark_index_id"],
        benchmark_index_sha256=readiness["benchmark_index_sha256"], training_seeds=seeds,
        progress_callback=progress_callback, stop_requested=stop_requested,
    )
    result.update({"execution_id": execution_id, "bundle_id": bundle_id,
                   "timeout_seconds": plan.get("timeout_seconds", 120),
                   "mode": plan["mode"], "targeted_plan_id": plan["targeted_plan_id"],
                   "targeted_plan_version": plan["targeted_plan_version"],
                   "parent_dataset_refs": parent_refs,
                   "contamination_scope": readiness["contamination_scope"],
                   "limitations": readiness["limitations"]})
    output = artifact_root / "executions" / execution_id / "candidates.jsonl"
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                        for row in result["candidates"]), encoding="utf-8")
    result["candidate_file"] = str(output.relative_to(root))
    result["candidate_sha256"] = sha256_file(output)
    result["registered"] = False
    return result


def preview_cpt_generation(workspace: str | Path, result: dict[str, Any], *, limit: int = 3) -> dict[str, Any]:
    if not 1 <= limit <= 5:
        raise ValueError("preview limit must be between 1 and 5")
    root = Path(workspace).resolve(strict=True)
    rel = Path(result["candidate_file"])
    path = (root / rel).resolve(strict=True)
    if not path.is_relative_to(root) or sha256_file(path) != result.get("candidate_sha256"):
        raise ValueError("CPT 候选文件路径或摘要校验失败")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"execution_id": result["execution_id"], "registered": False,
            "candidate_count": result["candidate_count"], "candidate_sha256": result["candidate_sha256"],
            "factuality": "not_verified", "examples": rows[:limit]}


def register_cpt_generation(workspace: str | Path, result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") not in {"completed", "partial", "cancelled"} or result.get("candidate_count", 0) < 1:
        raise ValueError("仅可登记含有效候选的已结束生成任务")
    preview_cpt_generation(workspace, result, limit=1)
    root = Path(workspace).resolve(strict=True)
    provider_meta = result["candidates"][0]["metadata"]
    source = DataSource(source_id=f"source-cpt-generated-{result['execution_id']}",
                        kind="model_generated", name="CPT model-generated candidates",
                        revision=provider_meta.get("provider_revision"),
                        revision_status="known" if provider_meta.get("provider_revision") else "unknown",
                        license_status="not_provided", declared_by="provider")
    path = root / result["candidate_file"]
    metadata = {
        "provider_profile_id": provider_meta["provider_profile_id"],
        "provider_adapter": provider_meta.get("provider_adapter"),
        "provider_model": provider_meta.get("provider_model"),
        "provider_revision": provider_meta.get("provider_revision"),
        "provider_revision_status": provider_meta.get("provider_revision_status", "unavailable"),
        "provider_identity_status": provider_meta.get(
            "provider_identity_status", "profile_declared_not_independently_verified"),
        "registered_model_association_status": "not_proven",
        "registered_model_id": None,
        "registered_model_sha256": None,
        "generation_protocol": "cpt-domain-text-v1", "verification_scope": "benchmark_example_fixture_only",
        "benchmark_index_id": result["benchmark_index_id"],
        "benchmark_index_sha256": result["benchmark_index_sha256"], "tokenizer_id": result["tokenizer_id"],
        "factuality": "not_verified",
        "generation_execution_id": result["execution_id"],
        "generation_candidate_sha256": result["candidate_sha256"],
        "generation_request_count": result["request_count"],
    }
    if result.get("targeted_plan_id"):
        metadata.update({"targeted_plan_id": result["targeted_plan_id"],
                         "targeted_plan_version": result["targeted_plan_version"],
                         "parent_dataset_ids": [ref["dataset_id"] for ref in result.get("parent_dataset_refs", [])],
                         "parent_dataset_refs": result.get("parent_dataset_refs", [])})
    registration = register_jsonl(BytesIO(path.read_bytes()), workspace=root,
                                  output_root=root / "data" / "processed" / "generated",
                                  stage="cpt", origin=source, registration_metadata=metadata)
    return {"dataset_id": registration.dataset_id, "record_count": registration.record_count,
            "canonical_sha256": registration.canonical_sha256,
            "processing_report_id": (load_manifest(registration.path / "manifests" / "cpt-model-generated-v1.yaml")
                                      .processing_report.report_id),
            "factuality": "not_verified", "registered": True}


class CPTGenerationJobs:
    """Persisted single-worker supervisor; refresh reads state and never replays Provider calls."""

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve(strict=True)
        self.root = self.workspace / "runs" / "cpt-generation"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._active_id: str | None = None
        self._stops: dict[str, threading.Event] = {}
        self._recover()

    def _dir(self, execution_id: str) -> Path:
        if not re.fullmatch(r"cptgenrun-[a-f0-9]{20}", execution_id):
            raise ValueError("execution_id 无效")
        path = self.root / execution_id
        if path.is_symlink() or not path.resolve(strict=False).is_relative_to(self.root):
            raise ValueError("execution 路径不可用")
        return path

    def _read(self, execution_id: str) -> dict[str, Any] | None:
        path = self._dir(execution_id) / "state.json"
        if path.is_symlink() or not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if value.get("execution_id") == execution_id else None
        except (OSError, TypeError, UnicodeError, json.JSONDecodeError):
            return None

    def _write(self, state: dict[str, Any]) -> None:
        path = self._dir(state["execution_id"]) / "state.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)

    def _recover(self) -> None:
        for path in self.root.glob("cptgenrun-*/state.json"):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if state.get("status") in {"queued", "running"}:
                state.update(status="interrupted", phase="interrupted",
                             detail="服务重启时任务仍在执行；未自动重放 Provider 请求")
                self._write(state)

    def start(self, bundle_id: str, *, confirm_start: bool) -> dict[str, Any]:
        if confirm_start is not True:
            raise ValueError("启动前必须明确确认 confirm_start=true")
        readiness = preflight_cpt_generation(self.workspace, bundle_id)
        if readiness["status"] != "preflight_ready":
            raise ValueError("CPT 生成预检未通过")
        ensure_no_uncertain_requests(self.workspace / "artifacts" / "synthetic" / bundle_id)
        with self._lock:
            if self._active_id:
                previous = self._read(self._active_id)
                if previous and previous.get("status") not in {"completed", "partial", "cancelled", "failed", "interrupted"}:
                    raise RuntimeError("已有 CPT 生成任务运行中")
            execution_id = "cptgenrun-" + __import__("uuid").uuid4().hex[:20]
            self._dir(execution_id).mkdir(parents=True, exist_ok=False)
            state = {"execution_id": execution_id, "bundle_id": bundle_id, "stage": "cpt",
                     "mode": readiness["mode"], "status": "queued", "phase": "queued",
                     "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                     "updated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                     "progress": {"attempted_requests": 0, "accepted_candidates": 0, "usable_tokens": 0},
                     "result": None, "detail": None}
            self._write(state)
            stop = threading.Event()
            self._stops[execution_id] = stop
            self._active_id = execution_id
            threading.Thread(target=self._run, args=(state, stop), daemon=True,
                             name=f"onegpu-{execution_id}").start()
            return state

    def _run(self, state: dict[str, Any], stop: threading.Event) -> None:
        execution_id = state["execution_id"]
        state.update(status="running", phase="generation")
        self._write(state)

        def progress(item: dict[str, int]) -> None:
            state["progress"] = item
            state["updated_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
            self._write(state)

        try:
            result = execute_cpt_generation(self.workspace, state["bundle_id"], execution_id=execution_id,
                                            progress_callback=progress, stop_requested=stop.is_set)
            private = self._dir(execution_id) / "result.json"
            private.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
            safe_result = {key: value for key, value in result.items() if key != "candidates"}
            state.update(status=result["status"], phase="review", result=safe_result)
        except Exception as exc:  # provider exceptions may include secrets: only persist sanitized fields
            state.update(status="failed", phase="failed", detail=f"生成失败（{type(exc).__name__}）；请检查本地日志")
            if isinstance(exc, ProviderError):
                state["provider_error"] = {
                    "category": exc.category,
                    "status_code": exc.status_code,
                    "provider_request_id": exc.provider_request_id,
                    "timeout_seconds": exc.timeout_seconds,
                }
                if exc.category == "timeout":
                    timeout = f"{exc.timeout_seconds:g}" if exc.timeout_seconds is not None else "配置时限"
                    state["detail"] = (f"Provider 请求超过 {timeout} 秒未返回；本次结果未知，可能已到达服务端。"
                                        "请核查 Provider 调用记录；系统未自动重试。")
                elif exc.status_code is not None:
                    state["detail"] = f"Provider 返回 HTTP {exc.status_code}（{exc.category}）；请检查凭据、权限、额度或服务状态。"
                else:
                    state["detail"] = f"Provider 请求失败（{exc.category}）；请查看请求账本中的安全诊断信息。"
        finally:
            self._write(state)
            with self._lock:
                self._stops.pop(execution_id, None)
                if self._active_id == execution_id:
                    self._active_id = None

    def get(self, execution_id: str) -> dict[str, Any] | None:
        return self._read(execution_id)

    def request_review(self, bundle_id: str, *, attempt_id: str | None = None) -> list[dict[str, Any]]:
        if not _ID.fullmatch(bundle_id):
            raise ValueError("CPT generation bundle ID 无效")
        bundle_dir = self.workspace / "artifacts" / "synthetic" / bundle_id
        if bundle_dir.is_symlink() or not (bundle_dir / "cpt_generation_plan.json").is_file():
            raise ValueError("CPT generation bundle 不存在")
        if attempt_id is not None:
            if self._active_id:
                state = self._read(self._active_id)
                if state and state.get("status") in {"queued", "running"}:
                    raise RuntimeError("任务运行中，暂不能处理未知请求")
            acknowledge_uncertain_attempt(bundle_dir, attempt_id)
        return unresolved_attempts(bundle_dir)

    def cancel(self, execution_id: str) -> dict[str, Any]:
        state = self._read(execution_id)
        stop = self._stops.get(execution_id)
        if state is None:
            raise KeyError(execution_id)
        if stop is None or state.get("status") not in {"queued", "running"}:
            raise RuntimeError("任务当前不可取消")
        stop.set()
        return state | {"cancel_requested": True}

    def preview(self, execution_id: str, *, limit: int = 3) -> dict[str, Any]:
        state = self._read(execution_id)
        if state is None:
            raise KeyError(execution_id)
        result_path = self._dir(execution_id) / "result.json"
        if not result_path.is_file() or result_path.is_symlink():
            raise ValueError("生成任务尚无可预览结果")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        return preview_cpt_generation(self.workspace, result, limit=limit)

    def register(self, execution_id: str) -> dict[str, Any]:
        state = self._read(execution_id)
        if state is None:
            raise KeyError(execution_id)
        if state.get("status") not in {"completed", "partial", "cancelled"}:
            raise RuntimeError("任务尚未结束或没有可登记候选")
        if state.get("registration"):
            return state
        result_path = self._dir(execution_id) / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        registration = register_cpt_generation(self.workspace, result)
        state["registration"] = registration
        state["registered_dataset_id"] = registration["dataset_id"]
        state["updated_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        self._write(state)
        return state

    def shutdown(self) -> None:
        with self._lock:
            for stop in self._stops.values():
                stop.set()
