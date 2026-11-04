"""Optional loopback-only API and same-origin production UI."""

from __future__ import annotations

import hashlib
from io import BytesIO
import ipaddress
import json
import os
import re
import secrets
import tempfile
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import yaml
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ..data.capabilities import source_stage_capabilities
from ..data.dataset_import import register_jsonl
from ..data.dataset_mixture import MixtureInput, build_mixture, preflight_mixture
from ..data.grpo_seed import (
    GRPOSeedRequest,
    build_grpo_seed_dataset,
    list_grpo_seed_candidates,
    preflight_grpo_seed,
)
from ..data.importer import MAX_IMPORT_BYTES
from ..data.source_formats import SFT_SOURCE_FORMATS, MAX_PREVIEW_RECORDS, convert_sft_jsonl
from ..data.objective_evaluation_import import (
    OBJECTIVE_EVALUATION_FORMAT, convert_objective_evaluation_jsonl,
)
from ..data.policy import DataPolicy, parse_policy, policy_templates
from ..data.policy_build import InputUnavailable, execute_policy, load_source
from ..data.policy_experiment import Budget, DataEffectExperiment, ExperimentArm
from ..data.registry import DataSource
from ..synthetic.request_guard import UncertainProviderRequest
from .catalog import ArtifactCatalog, ArtifactError
from .cpt_generation_workflow import (
    CPTGenerationJobs,
    preflight_cpt_generation,
    prepare_cpt_generation,
)
from .cpt_stage_adapter import (
    CPTPreflightRequest,
    CPTPreflightResponse,
    CPTStageAdapter,
)
from .dataset_preview import PreviewUnavailable, dataset_status, preview_dataset
from .evaluation_matrix import matrix_detail, matrix_index
from .grpo_generation_jobs import GRPOGenerationJobs
from .launcher import DATASET_ID, SFTLauncher
from .model_playground_catalog import ModelPlaygroundCatalog
from .model_display_names import DisplayNameStoreError, ModelDisplayNameStore
from .model_playground_jobs import ModelPlaygroundJobs
from .preference_generation_jobs import PreferenceGenerationJobs
from .preflight_confirmations import StagePreflightConfirmations
from .run_workspace import RunWorkspace
from .sft_generation_jobs import SFTGenerationJobs
from .sft_generation_setup import (
    discover_sft_generation_setup,
    prepare_ordinary_sft_generation,
    prepare_prompt_only_sft_generation,
)
from .stage_profiles import StageProfileCatalog, stage_profile_catalog
from .stage_training import RegisteredStageTraining, RegisteredStageTrainingRequest
from .targeted_capabilities import stage_capabilities
from .training_preflight import (
    StagePreflightRequest,
    preflight_stage,
    stage_workbench_options,
)
from ..assistant.api import AssistantAskRequest, AssistantProfileDraftRequest, AssistantProfileRequest, ContextSummaryRequest, MemoryDraftRequest
from ..assistant.context import rebuild_context_summary
from ..assistant.contracts import ContextSummary, WorkspaceMemory
from ..assistant.evaluation import PlatformFixtureHarness, evaluate_harness, load_mvp_cases
from ..assistant.evidence_reader import WorkspaceEvidenceReader
from ..assistant.agentscope_runtime import AgentScopeRuntime
from ..assistant.langfuse_observer import LangfuseObserver
from ..assistant.model_adapter import BailianAssistantModelAdapter
from ..assistant.profiles import AssistantProfileCatalog, draft_profile_from_description
from ..assistant.service import WorkspaceAssistantService
from ..assistant.store import WorkspaceAssistantStore
from ..weight_space.workbench import WeightSpaceWorkbench


class WeightSpaceStartRequest(BaseModel):
    confirmation_token: str


class WeightSpaceCandidateRequest(BaseModel):
    candidate_id: str


class WeightSpaceTransitionRequest(BaseModel):
    candidate_id: str
    action: str


def _aggregate_choice_report_metrics(rows: list[dict]) -> dict[str, dict]:
    """Aggregate effective judge outcomes over the full registered question set."""
    metric_groups: dict[str, list[dict]] = {
        "overall": rows,
        "single_choice": [row for row in rows if row.get("question_type") == "single_choice"],
        "multiple_choice": [row for row in rows if row.get("question_type") == "multiple_choice"],
        "true_false": [row for row in rows if row.get("question_type") == "true_false"],
    }
    metrics: dict[str, dict] = {}
    for group, group_rows in metric_groups.items():
        counts = {"correct": 0, "incorrect": 0, "undetermined": 0,
                  "provider_errors": 0, "not_evaluated": 0}
        for row in group_rows:
            outcome = row.get("effective_judge_outcome")
            if outcome in {"correct", "incorrect", "undetermined", "provider_error"}:
                counts["provider_errors" if outcome == "provider_error" else outcome] += 1
            else:
                counts["not_evaluated"] += 1
        denominator = len(group_rows)
        decided = counts["correct"] + counts["incorrect"]
        metrics[group] = {
            **counts,
            "denominator": denominator,
            "decided_denominator": decided,
            "accuracy": counts["correct"] / decided if decided else None,
            "decision_coverage": decided / denominator if denominator else None,
            "status": "measured" if decided else "not_measured",
        }
    return metrics


def create_app(
    workspace: Path,
    static_dir: Path | None = None,
    *,
    enable_policy_editor: bool = False,
    enable_experiment_launcher: bool = False,
    enable_synthetic_generation: bool = False,
    allow_remote: bool = False,
) -> FastAPI:
    if (enable_experiment_launcher or enable_synthetic_generation) and allow_remote:
        raise ValueError("本地训练与模型生成操作仅允许 loopback 本机访问")
    workspace = workspace.resolve()
    catalog = ArtifactCatalog(workspace)
    run_workspace = RunWorkspace(workspace)
    model_playground_catalog = ModelPlaygroundCatalog(workspace)
    model_display_names = ModelDisplayNameStore(workspace)
    assistant_store = WorkspaceAssistantStore(workspace)
    assistant_profiles = AssistantProfileCatalog(assistant_store)
    assistant_observer = LangfuseObserver(workspace)
    assistant_model = BailianAssistantModelAdapter(workspace)
    assistant_runtime_name = os.environ.get("ONEGPU_ASSISTANT_RUNTIME", "agentscope").strip().lower()
    if assistant_runtime_name not in {"legacy", "agentscope"}:
        raise ValueError("ONEGPU_ASSISTANT_RUNTIME 只支持 agentscope 或 legacy")
    assistant_service = WorkspaceAssistantService(
        store=assistant_store,
        evidence_reader=WorkspaceEvidenceReader(workspace, catalog=catalog),
        profiles=assistant_profiles,
        model=assistant_model,
        observer=assistant_observer,
        agent_runtime=AgentScopeRuntime(workspace, assistant_model) if assistant_runtime_name == "agentscope" else None,
    )
    choice_retry_lock = threading.RLock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        launcher = app.state.experiment_launcher
        if launcher is not None:
            launcher.shutdown()
        generation_jobs = app.state.sft_generation_jobs
        if generation_jobs is not None:
            generation_jobs.shutdown()
        cpt_generation_jobs = app.state.cpt_generation_jobs
        if cpt_generation_jobs is not None:
            cpt_generation_jobs.shutdown()
        preference_generation_jobs = app.state.preference_generation_jobs
        if preference_generation_jobs is not None:
            preference_generation_jobs.shutdown()
        grpo_generation_jobs = app.state.grpo_generation_jobs
        if grpo_generation_jobs is not None:
            grpo_generation_jobs.shutdown()
        model_playground_jobs = app.state.model_playground_jobs
        if model_playground_jobs is not None:
            model_playground_jobs.shutdown()

    app = FastAPI(title="OneGPU PostTrain · Data", docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=lifespan)
    app.state.catalog = catalog
    app.state.assistant_service = assistant_service
    app.state.assistant_observer = assistant_observer
    app.state.policy_editor_enabled = enable_policy_editor
    app.state.policy_editor_token = secrets.token_urlsafe(32) if enable_policy_editor else None
    app.state.policy_preview_lock = threading.Lock()
    app.state.experiment_launcher_enabled = enable_experiment_launcher
    app.state.experiment_launcher = SFTLauncher(workspace) if enable_experiment_launcher else None
    app.state.experiment_launcher_token = secrets.token_urlsafe(32) if enable_experiment_launcher else None
    app.state.synthetic_generation_enabled = enable_synthetic_generation
    app.state.sft_generation_jobs = SFTGenerationJobs(workspace) if enable_synthetic_generation else None
    app.state.cpt_generation_jobs = CPTGenerationJobs(workspace) if enable_synthetic_generation else None
    app.state.preference_generation_jobs = PreferenceGenerationJobs(workspace) if enable_synthetic_generation else None
    app.state.grpo_generation_jobs = GRPOGenerationJobs(workspace) if enable_synthetic_generation else None
    app.state.model_playground_jobs = (ModelPlaygroundJobs(workspace, model_playground_catalog)
                                       if enable_experiment_launcher and not allow_remote else None)
    app.state.synthetic_generation_token = secrets.token_urlsafe(32) if enable_synthetic_generation else None
    app.state.weight_space_workbench = WeightSpaceWorkbench(workspace) if enable_experiment_launcher and not allow_remote else None
    app.state.stage_preflight_confirmations = StagePreflightConfirmations()
    app.state.stage_training = (RegisteredStageTraining(
        workspace, app.state.experiment_launcher, app.state.stage_preflight_confirmations,
    ) if enable_experiment_launcher else None)
    allowed_hosts = ["*"] if allow_remote else ["localhost", "127.0.0.1", "[::1]", "testserver"]
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    @app.middleware("http")
    async def read_only_headers(request, call_next):
        if request.url.path.startswith(("/api/assistant/", "/api/launcher/", "/api/synthetic/sft/", "/api/synthetic/cpt/", "/api/synthetic/preference/", "/api/synthetic/grpo/", "/api/training/", "/api/data/mixtures/", "/api/model-playground/", "/api/weight-space/")) and request.method == "POST":
            try:
                limit = (16 * 1024 if request.url.path.startswith("/api/assistant/") else
                         72 * 1024 if request.url.path.startswith("/api/model-playground/") else
                         96 * 1024 if request.url.path == "/api/synthetic/preference/prepare" else
                         40 * 1024 if request.url.path == "/api/synthetic/grpo/prepare" else 4096)
                if int(request.headers.get("content-length", "0")) > limit:
                    return JSONResponse({"detail": "本地生成请求体超过该操作的大小上限"}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "无效 Content-Length"}, status_code=400)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    @app.get("/api/health")
    def health():
        return {"status": "ok", "read_only": not (enable_policy_editor or enable_experiment_launcher or enable_synthetic_generation), "page": "data", "api_version": "1"}

    def local_weight_space(request: Request) -> WeightSpaceWorkbench:
        service = app.state.weight_space_workbench
        if service is None:
            raise HTTPException(403, "权重空间工作台仅在本机 Launcher 模式可用")
        client_host = request.client.host if request.client else ""
        try:
            client_ok = ipaddress.ip_address(client_host).is_loopback
        except ValueError:
            client_ok = client_host in {"testclient", "localhost"}
        try:
            host_ok = ipaddress.ip_address(request.url.hostname or "").is_loopback
        except ValueError:
            host_ok = (request.url.hostname or "").lower() in {"localhost", "testserver"}
        if not client_ok or not host_ok:
            raise HTTPException(403, "权重空间工作台仅接受 loopback 本机请求")
        return service

    @app.get("/api/weight-space/status")
    def weight_space_status(request: Request):
        return local_weight_space(request).snapshot()

    @app.post("/api/weight-space/iteration/preflight")
    def weight_space_preflight(request: Request):
        try:
            return local_weight_space(request).preflight()
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)[:240]) from exc

    @app.post("/api/weight-space/iteration/start", status_code=202)
    def weight_space_start(request: Request, body: WeightSpaceStartRequest):
        try:
            return local_weight_space(request).start_iteration(body.confirmation_token)
        except ValueError as exc:
            raise HTTPException(409, str(exc)[:240]) from exc

    @app.post("/api/weight-space/candidates/final", status_code=202)
    def weight_space_final(request: Request, body: WeightSpaceCandidateRequest):
        try:
            return local_weight_space(request).start_final(body.candidate_id)
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)[:240]) from exc

    @app.post("/api/weight-space/candidates/transition")
    def weight_space_transition(request: Request, body: WeightSpaceTransitionRequest):
        try:
            return local_weight_space(request).transition(body.candidate_id, body.action)
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)[:240]) from exc

    def require_local_assistant_request(request: Request) -> None:
        host = request.client.host if request.client else ""
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = host in {"testclient", "localhost"}
        if not local:
            raise HTTPException(403, "工作区助手仅接受本机请求")
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "工作区助手仅接受同源请求")

    @app.get("/api/assistant/status")
    def assistant_status(request: Request, session_id: str | None = None):
        require_local_assistant_request(request)
        session = assistant_store.get_session(session_id) if session_id else None
        return {
            "provider": "阿里云百炼",
            "runtime": assistant_runtime_name,
            "model_profile_id": "dashscope-qwen3.6-plus-local",
            "provider_configured": assistant_service.model.is_configured(),
            "data_egress_consent": bool(session and session.data_egress_consent_at),
            "data_egress_consent_at": session.data_egress_consent_at.isoformat() if session and session.data_egress_consent_at else None,
            "read_only": True,
            "observability": assistant_observer.status(),
        }

    @app.get("/api/assistant/observability")
    def assistant_observability_status(request: Request):
        require_local_assistant_request(request)
        return assistant_observer.status()

    @app.post("/api/assistant/observability/check")
    def assistant_observability_check(request: Request):
        require_local_assistant_request(request)
        return assistant_observer.check_connection()

    @app.get("/api/assistant/profiles")
    def assistant_list_profiles(request: Request):
        require_local_assistant_request(request)
        return [item.model_dump(mode="json") for item in assistant_profiles.list_profiles()]

    @app.post("/api/assistant/profiles/draft")
    def assistant_profile_draft(body: AssistantProfileDraftRequest, request: Request):
        require_local_assistant_request(request)
        try:
            return {"draft": draft_profile_from_description(body.description), "saved": False,
                    "capabilities": "仅允许现有只读工作区证据"}
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/assistant/profiles")
    def assistant_create_profile(body: AssistantProfileRequest, request: Request):
        require_local_assistant_request(request)
        try:
            return assistant_profiles.create(body.name, body.role_instruction).model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.put("/api/assistant/profiles/{profile_id}")
    def assistant_update_profile(profile_id: str, body: AssistantProfileRequest, request: Request):
        require_local_assistant_request(request)
        try:
            return assistant_profiles.update(
                profile_id, name=body.name, role_instruction=body.role_instruction,
            ).model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(404, "助手配置不存在") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.delete("/api/assistant/profiles/{profile_id}")
    def assistant_delete_profile(profile_id: str, request: Request):
        require_local_assistant_request(request)
        try:
            if not assistant_profiles.delete(profile_id):
                raise HTTPException(404, "助手配置不存在")
            return {"deleted": True}
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/assistant/memories")
    def assistant_list_memories(request: Request):
        require_local_assistant_request(request)
        return [item.model_dump(mode="json") for item in assistant_store.list_memories()]

    @app.post("/api/assistant/memories/draft")
    def assistant_memory_draft(body: MemoryDraftRequest, request: Request):
        require_local_assistant_request(request)
        try:
            draft = WorkspaceMemory(
                memory_id=f"memory-{secrets.token_hex(10)}", content=body.content,
                status="active", created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        except ValidationError as exc:
            raise HTTPException(422, "记忆草稿无效") from exc
        return {"draft": draft.model_dump(mode="json"), "saved": False, "confirmation_required": True}

    @app.post("/api/assistant/memories")
    def assistant_save_memory(body: MemoryDraftRequest, request: Request):
        require_local_assistant_request(request)
        try:
            memory = WorkspaceMemory(
                memory_id=f"memory-{secrets.token_hex(10)}", content=body.content,
                status="active", created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            return assistant_store.save_memory(memory).model_dump(mode="json")
        except ValidationError as exc:
            raise HTTPException(422, "记忆内容无效") from exc

    @app.put("/api/assistant/memories/{memory_id}")
    def assistant_update_memory(memory_id: str, body: MemoryDraftRequest, request: Request):
        require_local_assistant_request(request)
        current = next((item for item in assistant_store.list_memories() if item.memory_id == memory_id), None)
        if current is None:
            raise HTTPException(404, "工作区记忆不存在")
        try:
            updated = current.model_copy(update={"content": body.content, "updated_at": datetime.now(timezone.utc)})
            updated = WorkspaceMemory.model_validate(updated.model_dump())
            return assistant_store.save_memory(updated).model_dump(mode="json")
        except ValidationError as exc:
            raise HTTPException(422, "记忆内容无效") from exc

    @app.patch("/api/assistant/memories/{memory_id}/status")
    def assistant_set_memory_status(memory_id: str, enabled: bool, request: Request):
        require_local_assistant_request(request)
        current = next((item for item in assistant_store.list_memories() if item.memory_id == memory_id), None)
        if current is None:
            raise HTTPException(404, "工作区记忆不存在")
        updated = current.model_copy(update={
            "status": "active" if enabled else "disabled",
            "updated_at": datetime.now(timezone.utc),
        })
        return assistant_store.save_memory(updated).model_dump(mode="json")

    @app.delete("/api/assistant/memories/{memory_id}")
    def assistant_delete_memory(memory_id: str, request: Request):
        require_local_assistant_request(request)
        if not assistant_store.delete_memory(memory_id):
            raise HTTPException(404, "工作区记忆不存在")
        return {"deleted": True}

    @app.get("/api/assistant/sessions")
    def assistant_list_sessions(request: Request, limit: int = 50):
        require_local_assistant_request(request)
        return [{
            "session_id": item.session_id, "title": item.title,
            "assistant_profile_id": item.assistant_profile_id,
            "updated_at": item.updated_at.isoformat(),
            "data_egress_consent_at": item.data_egress_consent_at.isoformat() if item.data_egress_consent_at else None,
        } for item in assistant_store.list_sessions(limit)]

    @app.get("/api/assistant/tasks")
    def assistant_list_tasks(request: Request, session_id: str | None = None, limit: int = 50):
        require_local_assistant_request(request)
        return [item.model_dump(mode="json") for item in assistant_store.list_tasks(
            session_id=session_id, limit=limit,
        )]

    @app.get("/api/assistant/evaluations")
    def assistant_list_evaluations(request: Request, limit: int = 20):
        require_local_assistant_request(request)
        return assistant_store.list_evaluation_reports(limit)

    @app.post("/api/assistant/evaluations/reference")
    def assistant_run_reference_evaluation(request: Request):
        require_local_assistant_request(request)
        report = evaluate_harness(
            PlatformFixtureHarness.harness_id,
            PlatformFixtureHarness(),
            load_mvp_cases(),
            configuration_id=PlatformFixtureHarness.configuration_id,
        )
        assistant_store.save_evaluation_report(report)
        return report.model_dump(mode="json")

    @app.get("/api/assistant/sessions/{session_id}")
    def assistant_get_session(session_id: str, request: Request):
        require_local_assistant_request(request)
        session = assistant_store.get_session(session_id)
        if session is None:
            raise HTTPException(404, "助手会话不存在")
        return session.model_dump(mode="json")

    @app.get("/api/assistant/sessions/{session_id}/summary")
    def assistant_get_context_summary(session_id: str, request: Request):
        require_local_assistant_request(request)
        if assistant_store.get_session(session_id) is None:
            raise HTTPException(404, "助手会话不存在")
        summary = assistant_store.get_summary(session_id)
        return summary.model_dump(mode="json") if summary else None

    @app.post("/api/assistant/sessions/{session_id}/summary/rebuild")
    def assistant_rebuild_context_summary(session_id: str, request: Request):
        require_local_assistant_request(request)
        session = assistant_store.get_session(session_id)
        if session is None:
            raise HTTPException(404, "助手会话不存在")
        summary = rebuild_context_summary(session, assistant_store.get_summary(session_id))
        if summary is None:
            raise HTTPException(409, "对话尚未达到上下文压缩条件")
        return assistant_store.save_summary(summary).model_dump(mode="json")

    @app.put("/api/assistant/sessions/{session_id}/summary")
    def assistant_edit_context_summary(session_id: str, body: ContextSummaryRequest, request: Request):
        require_local_assistant_request(request)
        current = assistant_store.get_summary(session_id)
        if current is None:
            raise HTTPException(404, "上下文摘要不存在")
        try:
            updated = current.model_copy(update={"summary": body.summary, "updated_at": datetime.now(timezone.utc)})
            updated = ContextSummary.model_validate(updated.model_dump())
            return assistant_store.save_summary(updated).model_dump(mode="json")
        except ValidationError as exc:
            raise HTTPException(422, "摘要内容无效") from exc

    @app.post("/api/assistant/sessions/{session_id}/consent")
    def assistant_confirm_consent(session_id: str, request: Request):
        require_local_assistant_request(request)
        try:
            session = assistant_service.confirm_data_egress(session_id)
        except KeyError as exc:
            raise HTTPException(404, "助手会话不存在") from exc
        return {
            "session_id": session.session_id,
            "data_egress_consent": True,
            "consent_at": session.data_egress_consent_at.isoformat(),
            "consent_version": session.data_egress_consent_version,
        }

    @app.post("/api/assistant/ask")
    def assistant_ask(body: AssistantAskRequest, request: Request):
        require_local_assistant_request(request)
        try:
            result = assistant_service.ask(
                body.question, session_id=body.session_id,
                assistant_profile_id=body.assistant_profile_id,
            )
        except KeyError as exc:
            raise HTTPException(404, "助手会话不存在") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {
            "task": result.task.model_dump(mode="json"),
            "answer": result.answer.model_dump(mode="json") if result.answer else None,
            "evidence": [item.model_dump(mode="json") for item in result.evidence],
            "comparisons": [item.model_dump(mode="json") for item in result.comparisons],
            "consent_required": result.consent_required,
        }

    @app.get("/api/runs")
    def runs_index(stage: str | None = None, status: str | None = None, q: str | None = None,
                   limit: int = 100, offset: int = 0):
        if stage is not None and len(stage) > 40 or status is not None and len(status) > 40 or q is not None and len(q) > 120:
            raise HTTPException(422, "Runs 筛选条件超长")
        try:
            return run_workspace.list_runs(stage=stage, status=status, q=q, limit=limit, offset=offset)
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, "Runs 分页参数无效") from exc

    @app.get("/api/model-playground/models")
    def model_playground_models(request: Request):
        # Unlike general read-only workspace metadata, model selection is always a
        # loopback-only boundary, even when the data UI is configured for remote use.
        client_host = request.client.host if request.client else ""
        try:
            loopback_client = ipaddress.ip_address(client_host).is_loopback
        except ValueError:
            loopback_client = client_host in {"testclient", "localhost"}
        try:
            loopback_host = ipaddress.ip_address(request.url.hostname or "").is_loopback
        except ValueError:
            loopback_host = (request.url.hostname or "").lower() in {"localhost", "testserver"}
        if not loopback_client or not loopback_host:
            raise HTTPException(403, "模型体验仅允许 loopback 本机访问")
        snapshot = model_playground_catalog.snapshot()
        names = model_display_names.get_all(model.get("model_id") for model in snapshot["models"])
        allowed = ("model_id", "kind", "stage", "digest", "source_run_id", "produced_by_run_id",
                   "training_completed_at", "display_name", "available", "unavailable_reason")
        return {"status": snapshot["status"], "detail": snapshot["detail"],
                "models": [{key: (names.get(model.get("model_id")) if key == "display_name" else model.get(key))
                            for key in allowed} for model in snapshot["models"]]}

    @app.put("/api/model-playground/models/{model_id}/display-name")
    def model_display_name_set(request: Request, model_id: str, payload: dict,
                               x_local_launcher_token: str | None = Header(default=None)):
        require_launcher(request, x_local_launcher_token)
        if not re.fullmatch(r"model-[a-zA-Z0-9][a-zA-Z0-9._-]{0,159}", model_id):
            raise HTTPException(422, "model_id 无效")
        if set(payload) != {"display_name"}:
            raise HTTPException(422, "请求只接受 display_name")
        snapshot = model_playground_catalog.snapshot()
        if snapshot["status"] == "loading":
            raise HTTPException(409, "本地模型目录仍在核验；请稍后重试")
        if snapshot["status"] != "ready":
            raise HTTPException(503, snapshot.get("detail") or "本地模型目录核验失败")
        registered = {model.get("model_id") for model in snapshot["models"]}
        try:
            display_name = model_display_names.set(model_id, payload["display_name"], registered_model_ids=registered)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except DisplayNameStoreError as exc:
            raise HTTPException(500, str(exc)) from exc
        return {"model_id": model_id, "display_name": display_name}

    @app.delete("/api/model-playground/models/{model_id}/display-name", status_code=204)
    def model_display_name_clear(request: Request, model_id: str,
                                 x_local_launcher_token: str | None = Header(default=None)):
        require_launcher(request, x_local_launcher_token)
        if not re.fullmatch(r"model-[a-zA-Z0-9][a-zA-Z0-9._-]{0,159}", model_id):
            raise HTTPException(422, "model_id 无效")
        snapshot = model_playground_catalog.snapshot()
        if snapshot["status"] == "loading":
            raise HTTPException(409, "本地模型目录仍在核验；请稍后重试")
        if snapshot["status"] != "ready":
            raise HTTPException(503, snapshot.get("detail") or "本地模型目录核验失败")
        registered = {model.get("model_id") for model in snapshot["models"]}
        try:
            model_display_names.clear(model_id, registered_model_ids=registered)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except DisplayNameStoreError as exc:
            raise HTTPException(500, str(exc)) from exc
        return Response(status_code=204)

    @app.get("/api/runs/{run_id}")
    def runs_detail(run_id: str):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", run_id):
            raise HTTPException(404, "Unknown run")
        detail = run_workspace.run_detail(run_id)
        if detail is None:
            raise HTTPException(404, "Unknown run")
        detail["produced_artifacts"] = attach_workspace_model_display_names(
            detail.get("produced_artifacts"), id_field="id")
        detail["handoff_models"] = attach_workspace_model_display_names(
            detail.get("handoff_models"), id_field="model_id")
        return detail

    @app.get("/api/runs/{run_id}/artifacts/{artifact_name}")
    def runs_artifact(run_id: str, artifact_name: str):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", run_id):
            raise HTTPException(404, "Unknown run")
        artifact = run_workspace.read_artifact(run_id, artifact_name)
        if artifact is None:
            raise HTTPException(404, "Artifact 不可用或不在允许列表")
        content, media_type = artifact
        return Response(content=content, media_type=media_type, headers={"Content-Disposition": "inline"})

    @app.get("/api/launcher/capabilities")
    def launcher_capabilities():
        launcher = app.state.experiment_launcher
        if launcher is None:
            return {"enabled": False, "read_only": True, "reason": "请使用 --enable-experiment-launcher 显式启用"}
        return launcher.capabilities() | {"token": app.state.experiment_launcher_token,
                                          "read_only": False, "local_only": True}

    def require_launcher(request: Request, token: str | None) -> SFTLauncher:
        launcher = app.state.experiment_launcher
        if launcher is None:
            raise HTTPException(403, "本地训练启动未启用；请使用 --enable-experiment-launcher 启动")
        if token != app.state.experiment_launcher_token:
            raise HTTPException(403, "本地 Launcher 令牌无效")
        origin = request.headers.get("origin")
        if origin:
            expected = f"{request.url.scheme}://{request.url.netloc}"
            if origin != expected:
                raise HTTPException(403, "只接受同源 Launcher 请求")
        client_host = request.client.host if request.client else ""
        try:
            loopback_client = ipaddress.ip_address(client_host).is_loopback
        except ValueError:
            loopback_client = client_host in {"testclient", "localhost"}
        try:
            loopback_host = ipaddress.ip_address(request.url.hostname or "").is_loopback
        except ValueError:
            loopback_host = (request.url.hostname or "").lower() in {"localhost", "testserver"}
        if not loopback_client or not loopback_host:
            raise HTTPException(403, "Launcher 仅接受 loopback 本机请求")
        return launcher

    def require_model_playground(request: Request, token: str | None) -> ModelPlaygroundJobs:
        require_launcher(request, token)
        jobs = app.state.model_playground_jobs
        if jobs is None or allow_remote:
            raise HTTPException(403, "模型体验仅在启用 Launcher 的本机 loopback 服务中可用")
        return jobs

    def attach_workspace_model_display_names(rows: object, *, id_field: str) -> list[dict]:
        if not isinstance(rows, list):
            return []
        snapshot = model_playground_catalog.snapshot()
        if snapshot["status"] != "ready":
            names: dict[str, str] = {}
        else:
            registered_ids = {model.get("model_id") for model in snapshot["models"]}
            names = model_display_names.get_all(registered_ids)
        enriched: list[dict] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            model_id = item.get(id_field)
            item["display_name"] = names.get(model_id) if isinstance(model_id, str) else None
            enriched.append(item)
        return enriched

    @app.post("/api/model-playground/sessions", status_code=202)
    def model_playground_session_start(request: Request, payload: dict,
                                       x_local_launcher_token: str | None = Header(default=None)):
        jobs = require_model_playground(request, x_local_launcher_token)
        if set(payload) != {"model_id"} or not isinstance(payload.get("model_id"), str):
            raise HTTPException(422, "会话启动只接受已验证 model_id")
        try:
            return jobs.start_session(payload["model_id"])
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/model-playground/sessions/{session_id}")
    def model_playground_session_status(request: Request, session_id: str,
                                       x_local_launcher_token: str | None = Header(default=None)):
        jobs = require_model_playground(request, x_local_launcher_token)
        if not re.fullmatch(r"[a-f0-9]{32}", session_id):
            raise HTTPException(404, "Unknown model session")
        result = jobs.session_status(session_id)
        if result is None:
            raise HTTPException(404, "Unknown model session")
        return result

    @app.post("/api/model-playground/sessions/{session_id}/messages", status_code=202)
    def model_playground_message(request: Request, session_id: str, payload: dict,
                                 x_local_launcher_token: str | None = Header(default=None)):
        jobs = require_model_playground(request, x_local_launcher_token)
        if (not re.fullmatch(r"[a-f0-9]{32}", session_id)
                or set(payload) - {"message", "parameters"}
                or not isinstance(payload.get("message"), str)
                or ("parameters" in payload and not isinstance(payload["parameters"], dict))):
            raise HTTPException(422, "消息请求只接受 message 和可选生成 parameters")
        try:
            return jobs.submit_message(session_id, payload["message"], payload.get("parameters"))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/model-playground/comparisons", status_code=202)
    def model_playground_compare(request: Request, payload: dict,
                                 x_local_launcher_token: str | None = Header(default=None)):
        jobs = require_model_playground(request, x_local_launcher_token)
        if (set(payload) - {"model_ids", "message", "parameters"}
                or not isinstance(payload.get("model_ids"), list)
                or not isinstance(payload.get("message"), str)
                or ("parameters" in payload and not isinstance(payload["parameters"], dict))):
            raise HTTPException(422, "对照请求只接受 model_ids、message 和可选生成 parameters")
        try:
            return jobs.submit_comparison(payload["model_ids"], payload["message"], payload.get("parameters"))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/model-playground/jobs/{job_id}")
    def model_playground_job_status(request: Request, job_id: str,
                                    x_local_launcher_token: str | None = Header(default=None)):
        jobs = require_model_playground(request, x_local_launcher_token)
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise HTTPException(404, "Unknown model job")
        result = jobs.job_status(job_id)
        if result is None:
            raise HTTPException(404, "Unknown model job")
        return result

    @app.post("/api/model-playground/jobs/{job_id}/cancel")
    def model_playground_job_cancel(request: Request, job_id: str,
                                   x_local_launcher_token: str | None = Header(default=None)):
        jobs = require_model_playground(request, x_local_launcher_token)
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise HTTPException(404, "Unknown model job")
        result = jobs.cancel(job_id)
        if result is None:
            raise HTTPException(404, "Unknown model job")
        return result

    @app.delete("/api/model-playground/sessions/{session_id}")
    def model_playground_session_close(request: Request, session_id: str,
                                       x_local_launcher_token: str | None = Header(default=None)):
        jobs = require_model_playground(request, x_local_launcher_token)
        if not re.fullmatch(r"[a-f0-9]{32}", session_id):
            raise HTTPException(404, "Unknown model session")
        if not jobs.close_session(session_id):
            raise HTTPException(404, "Unknown model session")
        return {"session_id": session_id, "status": "closed"}

    @app.post("/api/launcher/preflight")
    def launcher_preflight(request: Request, payload: dict, x_local_launcher_token: str | None = Header(default=None)):
        launcher = require_launcher(request, x_local_launcher_token)
        if (set(payload) - {"dataset_id", "parameter_overrides", "memory_mode_id", "validation", "checkpoint_policy"}
                or ("dataset_id" in payload and not isinstance(payload["dataset_id"], str))
                  or ("parameter_overrides" in payload and not isinstance(payload["parameter_overrides"], dict))
                  or ("memory_mode_id" in payload and not isinstance(payload["memory_mode_id"], str))):
            raise HTTPException(422, "预检只接受已登记 SFT dataset_id 和 profile 白名单参数，不接受 recipe、路径、命令或环境")
        try:
            if "validation" in payload:
                return launcher.preflight(payload.get("dataset_id", DATASET_ID), payload.get("parameter_overrides", {}),
                                          payload.get("memory_mode_id"), payload.get("validation"),
                                          payload.get("checkpoint_policy", "final"))
            if "checkpoint_policy" in payload:
                return launcher.preflight(payload.get("dataset_id", DATASET_ID), payload.get("parameter_overrides", {}),
                                          payload.get("memory_mode_id"), checkpoint_policy=payload["checkpoint_policy"])
            return launcher.preflight(payload.get("dataset_id", DATASET_ID), payload.get("parameter_overrides", {}),
                                      payload.get("memory_mode_id"))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/launcher/start")
    def launcher_start(request: Request, payload: dict, x_local_launcher_token: str | None = Header(default=None)):
        launcher = require_launcher(request, x_local_launcher_token)
        if (set(payload) - {"confirmation_token", "acknowledged_dataset_id"}
                or "confirmation_token" not in payload
                or not isinstance(payload.get("confirmation_token"), str)
                or ("acknowledged_dataset_id" in payload and not isinstance(payload["acknowledged_dataset_id"], str))):
            raise HTTPException(422, "启动只接受当前预检令牌及可选的已确认 dataset_id")
        try:
            return launcher.start(payload["confirmation_token"], payload.get("acknowledged_dataset_id"))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/launcher/evaluation/preflight")
    def launcher_evaluation_preflight(request: Request, payload: dict, x_local_launcher_token: str | None = Header(default=None)):
        launcher = require_launcher(request, x_local_launcher_token)
        if set(payload) != {"training_run_id"} or not isinstance(payload.get("training_run_id"), str):
            raise HTTPException(422, "评测预检只接受 training_run_id")
        return launcher.evaluation_preflight(payload["training_run_id"])

    @app.post("/api/launcher/evaluation/start")
    def launcher_evaluation_start(request: Request, payload: dict, x_local_launcher_token: str | None = Header(default=None)):
        launcher = require_launcher(request, x_local_launcher_token)
        if set(payload) != {"confirmation_token"} or not isinstance(payload.get("confirmation_token"), str):
            raise HTTPException(422, "评测启动只接受当前预检确认令牌")
        try:
            return launcher.start_evaluation(payload["confirmation_token"])
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/training/evaluation/preflight")
    def stage_evaluation_preflight(request: Request, payload: dict, x_local_launcher_token: str | None = Header(default=None)):
        launcher = require_launcher(request, x_local_launcher_token)
        allowed = {"training_run_id", "model_id", "profile_id", "dataset_id", "scoring_strategy"}
        if set(payload) - allowed or "profile_id" not in payload or ("training_run_id" in payload) == ("model_id" in payload):
            raise HTTPException(422, "阶段评测预检必须提供 training_run_id 或 model_id 之一，以及 profile_id 和可选 dataset_id/scoring_strategy")
        if any(not isinstance(value, str) for value in payload.values()):
            raise HTTPException(422, "阶段评测身份字段必须是字符串")
        if "training_run_id" in payload and not re.fullmatch(r"run-[a-z0-9][a-z0-9-]{0,119}", payload["training_run_id"]):
            raise HTTPException(422, "training_run_id 格式无效")
        if "model_id" in payload and not re.fullmatch(r"model-[a-zA-Z0-9][a-zA-Z0-9._-]{0,159}", payload["model_id"]):
            raise HTTPException(422, "model_id 格式无效")
        if not payload["profile_id"] or len(payload["profile_id"]) > 100:
            raise HTTPException(422, "profile_id 格式无效")
        try:
            return launcher.stage_evaluation_preflight(payload)
        except ValidationError as exc:
            raise HTTPException(422, "阶段评测身份格式无效") from exc

    @app.post("/api/training/evaluation/start")
    def stage_evaluation_start(request: Request, payload: dict, x_local_launcher_token: str | None = Header(default=None)):
        launcher = require_launcher(request, x_local_launcher_token)
        if set(payload) != {"confirmation_token"} or not isinstance(payload.get("confirmation_token"), str):
            raise HTTPException(422, "阶段评测启动只接受当前预检确认令牌")
        try:
            return launcher.start_stage_evaluation(payload["confirmation_token"])
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/launcher/run")
    def launcher_current_run():
        launcher = app.state.experiment_launcher
        if launcher is None:
            return {"enabled": False, "run": None}
        return {"enabled": True, "run": launcher.get_run()}

    @app.get("/api/launcher/runs/{run_id}")
    def launcher_run(run_id: str):
        launcher = app.state.experiment_launcher
        if launcher is None:
            raise HTTPException(404, "Launcher 未启用")
        if not re.fullmatch(r"run-[a-z0-9][a-z0-9-]{0,119}", run_id):
            raise HTTPException(404, "Unknown run")
        value = launcher.get_run(run_id)
        if value is None:
            raise HTTPException(404, "Unknown run")
        result = dict(value)
        result["model_artifacts"] = attach_workspace_model_display_names(
            value.get("model_artifacts"), id_field="model_id")
        return result

    @app.get("/api/launcher/runs/{run_id}/lineage")
    def launcher_lineage(run_id: str):
        launcher = app.state.experiment_launcher
        if launcher is None:
            raise HTTPException(404, "Launcher 未启用")
        state = launcher.get_run(run_id)
        if state is None:
            raise HTTPException(404, "Unknown run")
        path = workspace / str(state.get("output", "")) / "lineage.json"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 128 * 1024 or not path.resolve().is_relative_to(workspace):
            raise HTTPException(404, "Lineage 尚不可用")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            raise HTTPException(422, "Lineage 无法读取") from exc
        if not isinstance(value, dict) or value.get("run_id") != run_id:
            raise HTTPException(422, "Lineage 身份不匹配")
        return value

    @app.get("/api/launcher/evaluations/{run_id}/lineage")
    def launcher_evaluation_lineage(run_id: str):
        launcher = app.state.experiment_launcher
        if launcher is None or not re.fullmatch(r"run-[a-z0-9][a-z0-9-]{0,119}", run_id):
            raise HTTPException(404, "Unknown evaluation run")
        state = launcher.get_run(run_id)
        if not state or state.get("run_kind") != "evaluation":
            raise HTTPException(404, "Unknown evaluation run")
        path = workspace / str(state.get("output", "")) / "evaluation_lineage.json"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 128 * 1024 or not path.resolve().is_relative_to(workspace):
            raise HTTPException(404, "Evaluation lineage 尚不可用")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            raise HTTPException(422, "Evaluation lineage 无法读取") from exc
        if not isinstance(value, dict) or value.get("produced_by_run_id") != run_id:
            raise HTTPException(422, "Evaluation lineage 身份不匹配")
        return value

    @app.get("/api/launcher/evaluations/{run_id}/result")
    def launcher_evaluation_result(run_id: str):
        launcher = app.state.experiment_launcher
        if launcher is None or not re.fullmatch(r"run-[a-z0-9][a-z0-9-]{0,119}", run_id):
            raise HTTPException(404, "Unknown evaluation run")
        state = launcher.get_run(run_id)
        if not state or state.get("run_kind") != "evaluation":
            raise HTTPException(404, "Unknown evaluation run")
        path = workspace / str(state.get("output", "")) / "eval.json"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024 or not path.resolve().is_relative_to(workspace):
            raise HTTPException(404, "Evaluation result 尚不可用")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            raise HTTPException(422, "Evaluation result 无法读取") from exc
        return value

    @app.get("/api/launcher/evaluations/{run_id}/choice-report")
    def launcher_choice_evaluation_report(run_id: str):
        """Join objective-choice predictions or judge outcomes by verified record ID."""
        from ..data.io import read_jsonl
        from ..data.manifest import load_manifest, resolve_canonical_path, validate_manifest_evidence
        from ..lineage import sha256_file
        from .training_preflight import _find_manifest, _has_symlink_between

        launcher = app.state.experiment_launcher
        if launcher is None or not re.fullmatch(r"run-[a-z0-9][a-z0-9-]{0,119}", run_id):
            raise HTTPException(404, "Unknown evaluation run")
        state = launcher.get_run(run_id)
        if not state or state.get("run_kind") != "evaluation":
            raise HTTPException(404, "Unknown evaluation run")
        output = workspace / str(state.get("output", ""))
        if output.is_symlink() or not output.is_dir() or not output.resolve().is_relative_to(workspace):
            raise HTTPException(404, "Evaluation output 尚不可用")
        result_path = output / "eval.json"
        lineage_path = output / "evaluation_lineage.json"
        predictions_path = output / "choice_accuracy" / "choice_predictions.jsonl"
        for path, limit in ((result_path, 2 * 1024 * 1024), (lineage_path, 128 * 1024),
                            (predictions_path, 8 * 1024 * 1024)):
            if (path.is_symlink() or not path.is_file() or path.stat().st_size > limit
                    or not path.resolve().is_relative_to(output)):
                raise HTTPException(404, "逐题预测或评测证据尚不可用")
        try:
            evaluation = json.loads(result_path.read_text(encoding="utf-8"))
            lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
            if not isinstance(evaluation, dict) or not isinstance(lineage, dict):
                raise ValueError("评测结果或 lineage 格式无效")
            protocol = evaluation.get("protocol_version")
            if (protocol not in {"choice-accuracy-v1", "choice-llm-judge-v1"}
                    or lineage.get("protocol_version") != protocol
                    or evaluation.get("run_id") != run_id or lineage.get("produced_by_run_id") != run_id):
                raise ValueError("评测协议或 Run 身份不匹配")
            dataset_id = evaluation.get("dataset_id")
            if (not isinstance(dataset_id, str) or lineage.get("dataset_ids") != [dataset_id]
                    or evaluation.get("model_id") != lineage.get("model_id")):
                raise ValueError("评测模型或数据集身份不匹配")
            manifest_path = _find_manifest(workspace, dataset_id)
            manifest = load_manifest(manifest_path)
            metadata = manifest.metadata if isinstance(manifest.metadata, dict) else {}
            role = metadata.get("dataset_role") or metadata.get("role")
            if manifest.dataset_id != dataset_id or manifest.stage != "eval" or role != "evaluation_only":
                raise ValueError("登记数据集不是本次 objective evaluation-only suite")
            evidence = validate_manifest_evidence(manifest_path)
            if evidence:
                raise ValueError("评测数据登记证据无效：" + "; ".join(evidence[:3]))
            canonical_path = resolve_canonical_path(manifest_path, manifest)
            canonical_hash = manifest.hashes.get("canonical_records_sha256")
            if not canonical_hash or sha256_file(canonical_path) != canonical_hash:
                raise ValueError("评测集 canonical 摘要与登记不匹配")
            artifacts = lineage.get("dataset_artifacts", [])
            matching = [item for item in artifacts if isinstance(item, dict) and item.get("id") == dataset_id]
            digest = matching[0].get("digest") if len(matching) == 1 else None
            if not isinstance(digest, dict) or digest.get("value") != canonical_hash:
                raise ValueError("评测 lineage 未绑定当前评测集摘要")
            suite = evaluation.get("evaluation_suite") or {}
            is_judge = protocol == "choice-llm-judge-v1"
            choice_result = evaluation.get("choice_llm_judge" if is_judge else "choice_accuracy") or {}
            if not isinstance(suite, dict) or not isinstance(choice_result, dict):
                raise ValueError("评测 suite 或逐题结果格式无效")
            if (suite.get("dataset_id") != dataset_id or suite.get("dataset_sha256") != canonical_hash
                    or choice_result.get("predictions_path") != predictions_path.name):
                raise ValueError("评测结果未绑定当前评测集摘要")

            records = list(read_jsonl(canonical_path))
            if len(records) != manifest.sample_count or len(records) > 5000:
                raise ValueError("评测集数量与登记不符或超过逐题报告上限")
            canonical_by_id: dict[str, dict] = {}
            for record in records:
                record_id = record.get("id")
                if not isinstance(record_id, str) or not record_id or record_id in canonical_by_id:
                    raise ValueError("canonical 评测集题目 ID 缺失或重复")
                canonical_by_id[record_id] = record
            predictions: dict[str, dict] = {}
            with predictions_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if len(predictions) >= 5000:
                        raise ValueError("逐题预测超过报告上限")
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ValueError("逐题预测记录格式无效")
                    record_id = item.get("id")
                    if not isinstance(record_id, str) or record_id not in canonical_by_id or record_id in predictions:
                        raise ValueError("预测题目 ID 缺失、重复或不属于当前评测集")
                    canonical = canonical_by_id[record_id]
                    if item.get("gold") != canonical.get("answer") or item.get("question_type") != canonical.get("question_type"):
                        raise ValueError("预测题型或标准答案与登记评测集不匹配")
                    if is_judge:
                        outcome = item.get("judge_outcome")
                        if outcome not in {"correct", "incorrect", "undetermined", "provider_error"}:
                            raise ValueError("LLM 逐题评判状态无效")
                        expected_correct = True if outcome == "correct" else False if outcome == "incorrect" else None
                        if item.get("correct") is not expected_correct:
                            raise ValueError("LLM 逐题判定字段与 judge outcome 不一致")
                        if item.get("prediction") is not None or not isinstance(item.get("raw_output"), str):
                            raise ValueError("LLM 逐题模型回答字段无效")
                        rationale = item.get("judge_rationale")
                        if rationale is not None and (not isinstance(rationale, str) or len(rationale) > 1000):
                            raise ValueError("LLM 评判理由无效")
                        if outcome in {"correct", "incorrect", "undetermined"} and not rationale:
                            raise ValueError("LLM 已判定记录缺少评判理由")
                        judge_error = item.get("judge_error")
                        if judge_error is not None and (not isinstance(judge_error, str) or len(judge_error) > 80):
                            raise ValueError("LLM provider error category 无效")
                        request_id = item.get("provider_request_id")
                        if request_id is not None and (not isinstance(request_id, str) or len(request_id) > 200):
                            raise ValueError("LLM provider request ID 无效")
                        request_uncertain = item.get("judge_request_uncertain", False)
                        if not isinstance(request_uncertain, bool):
                            raise ValueError("LLM provider 请求确定性字段无效")
                    else:
                        if (not isinstance(item.get("parse_success"), bool)
                                or item.get("prediction") is not None and not isinstance(item.get("prediction"), str)
                                or not isinstance(item.get("correct"), bool)):
                            raise ValueError("逐题预测判定字段无效")
                        if item["correct"] != (item["prediction"] == item["gold"]):
                            raise ValueError("逐题判定与预测/标准答案不一致")
                    predictions[record_id] = item
            rows = []
            retry_path = workspace / ".onegpu-posttrain" / "launcher" / "choice_judge_retries" / f"{run_id}.jsonl"
            retries_by_id: dict[str, list[dict]] = {}
            if retry_path.exists() or retry_path.is_symlink():
                if (retry_path.is_symlink() or not retry_path.is_file() or retry_path.stat().st_size > 8 * 1024 * 1024
                        or not retry_path.resolve().is_relative_to(workspace)):
                    raise ValueError("逐题重试记录路径无效或超过大小上限")
                with choice_retry_lock, retry_path.open("r", encoding="utf-8") as retry_stream:
                    for line in retry_stream:
                        attempt = json.loads(line)
                        rationale = attempt.get("judge_rationale") if isinstance(attempt, dict) else None
                        error_category = attempt.get("judge_error") if isinstance(attempt, dict) else None
                        request_id = attempt.get("provider_request_id") if isinstance(attempt, dict) else None
                        if (not isinstance(attempt, dict) or attempt.get("source_run_id") != run_id
                                or attempt.get("question_id") not in canonical_by_id
                                or attempt.get("judge_outcome") not in {"correct", "incorrect", "undetermined", "provider_error"}
                                or isinstance(attempt.get("attempt_number"), bool)
                                or not isinstance(attempt.get("attempt_number"), int)
                                or not isinstance(attempt.get("judge_request_uncertain", False), bool)
                                or rationale is not None and (not isinstance(rationale, str) or len(rationale) > 1000)
                                or error_category is not None and (not isinstance(error_category, str)
                                    or len(error_category) > 80 or not re.fullmatch(r"[a-z0-9_]+", error_category))
                                or request_id is not None and (not isinstance(request_id, str) or len(request_id) > 200
                                    or not re.fullmatch(r"[A-Za-z0-9._:-]+", request_id))):
                            raise ValueError("逐题重试记录格式无效")
                        retries_by_id.setdefault(attempt["question_id"], []).append({
                            "attempt_number": attempt["attempt_number"],
                            "judge_outcome": attempt["judge_outcome"],
                            "judge_rationale": attempt.get("judge_rationale"),
                            "judge_error": attempt.get("judge_error"),
                            "judge_request_uncertain": attempt.get("judge_request_uncertain", False),
                            "provider_request_id": attempt.get("provider_request_id"),
                            "created_at": attempt.get("created_at"),
                        })
                        if sum(len(items) for items in retries_by_id.values()) > 1000:
                            raise ValueError("逐题重试记录超过上限")
            for record_id, record in canonical_by_id.items():
                prediction = predictions.get(record_id)
                retry_attempts = retries_by_id.get(record_id, [])
                effective_outcome = prediction.get("judge_outcome") if prediction and is_judge else None
                if is_judge:
                    for attempt in retry_attempts:
                        if attempt["judge_outcome"] in {"correct", "incorrect", "undetermined"}:
                            effective_outcome = attempt["judge_outcome"]
                rows.append({
                    "id": record_id,
                    "question_type": record.get("question_type"),
                    "question": record.get("prompt"),
                    "options": record.get("options", []),
                    "gold": record.get("answer"),
                    "prediction": prediction.get("prediction") if prediction else None,
                    "raw_output": prediction.get("raw_output") if prediction else None,
                    "parse_success": prediction.get("parse_success") if prediction and not is_judge else None,
                    "correct": prediction.get("correct") if prediction else None,
                    "judge_outcome": prediction.get("judge_outcome") if prediction and is_judge else None,
                    "judge_rationale": prediction.get("judge_rationale") if prediction and is_judge else None,
                    "judge_error": prediction.get("judge_error") if prediction and is_judge else None,
                    "judge_request_uncertain": prediction.get("judge_request_uncertain", False) if prediction and is_judge else False,
                    "provider_request_id": prediction.get("provider_request_id") if prediction and is_judge else None,
                    "effective_judge_outcome": effective_outcome if is_judge else None,
                    "retry_attempts": retry_attempts,
                    "status": ("not_evaluated" if prediction is None else
                               prediction.get("judge_outcome") if is_judge else
                               "parse_failed" if not prediction.get("parse_success") else
                               "correct" if prediction.get("correct") else "incorrect"),
                })
            original_metrics = choice_result.get("metrics", {})
            current_metrics = _aggregate_choice_report_metrics(rows) if is_judge else None
            return {"run_id": run_id, "dataset_id": dataset_id, "dataset_sha256": canonical_hash,
                    "protocol_version": protocol, "status": "complete" if evaluation.get("status") == "completed" and len(predictions) == len(records) else "partial",
                    "provider_model": choice_result.get("provider_model") if is_judge else None,
                    "provider_profile_id": choice_result.get("provider_profile_id") if is_judge else None,
                    "request_count": choice_result.get("request_count") if is_judge else None,
                    "prompt_version": choice_result.get("prompt_version") if is_judge else None,
                    "metrics": original_metrics,
                    "original_metrics": original_metrics if is_judge else None,
                    "current_metrics": current_metrics,
                    "sample_count": len(records), "evaluated_count": len(predictions), "rows": rows}
        except HTTPException:
            raise
        except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
            raise HTTPException(409, f"逐题报告不可用：{str(exc)[:300]}") from exc

    @app.post("/api/launcher/evaluations/{run_id}/choice-retry")
    def launcher_choice_judgment_retry(
        run_id: str, request: Request, payload: dict,
        x_local_launcher_token: str | None = Header(default=None),
    ):
        launcher = require_launcher(request, x_local_launcher_token)
        if (set(payload) != {"question_id", "confirm_external_submission"}
                or not isinstance(payload.get("question_id"), str)
                or not isinstance(payload.get("confirm_external_submission"), bool)
                or payload["confirm_external_submission"] is not True):
            raise HTTPException(422, "重试必须提供 question_id 并明确确认向百炼重新发送评测内容")
        if not re.fullmatch(r"run-[a-z0-9][a-z0-9-]{0,119}", run_id):
            raise HTTPException(404, "Unknown evaluation run")
        state = launcher.get_run(run_id)
        if (not state or state.get("run_kind") != "evaluation"
                or state.get("status") not in {"completed", "partial", "failed", "interrupted"}):
            raise HTTPException(409, "仅可重试已结束的评测 Run")

        report = launcher_choice_evaluation_report(run_id)
        if report.get("protocol_version") != "choice-llm-judge-v1":
            raise HTTPException(409, "仅支持重试百炼语义评判的 API 错误")
        row = next((item for item in report["rows"] if item["id"] == payload["question_id"]), None)
        if row is None:
            raise HTTPException(404, "评测题目不存在")
        if row.get("judge_outcome") != "provider_error":
            raise HTTPException(409, "只有原始判定为 API 错误的题目可以重试")
        if not isinstance(row.get("raw_output"), str):
            raise HTTPException(409, "缺少已保存的模型原始输出，无法重试")

        try:
            from ..evaluation.choice_judge import BailianChoiceJudge, load_choice_judge_profile
            from ..schemas import EvalRecord
            from ..synthetic.providers import MissingCredentialError, ProviderError, require_provider_credentials
            from ..synthetic.secrets import load_local_provider_credential

            profile = load_choice_judge_profile(workspace)
            if (report.get("provider_profile_id") != profile.profile_id
                    or report.get("provider_model") != profile.model):
                raise HTTPException(409, "当前百炼 Profile/模型与原评测不一致，不能重试以免改变评判协议")
            load_local_provider_credential(profile.api_key_env or "", workspace / ".env.local")
            require_provider_credentials(profile)
            record = EvalRecord.model_validate({
                "id": row["id"], "problem_id": row["id"], "raw_problem": row["question"],
                "prompt": row["question"], "answer": row["gold"],
                "eval_prompt_version": "manual-retry-v1", "metadata": {},
                "question_type": row["question_type"], "options": row["options"],
            })
        except HTTPException:
            raise
        except MissingCredentialError as exc:
            raise HTTPException(409, "百炼凭据不可用；请配置本地 API Key 后再重试") from exc
        except (ValueError, OSError, ValidationError) as exc:
            raise HTTPException(409, f"百炼重试预检失败：{str(exc)[:240]}") from exc

        attempts_path = workspace / ".onegpu-posttrain" / "launcher" / "choice_judge_retries" / f"{run_id}.jsonl"
        try:
            with choice_retry_lock:
                retry_root = workspace / ".onegpu-posttrain"
                launcher_root = retry_root / "launcher"
                for directory in (retry_root, launcher_root, attempts_path.parent):
                    if directory.is_symlink() or directory.exists() and not directory.resolve().is_relative_to(workspace):
                        raise ValueError("重试记录目录越界")
                attempts_path.parent.mkdir(parents=True, exist_ok=True)
                if attempts_path.parent.is_symlink() or not attempts_path.parent.resolve().is_relative_to(workspace):
                    raise ValueError("重试记录目录越界")
                if attempts_path.exists() or attempts_path.is_symlink():
                    if (attempts_path.is_symlink() or not attempts_path.is_file()
                            or not attempts_path.resolve().is_relative_to(workspace)
                            or attempts_path.stat().st_size > 8 * 1024 * 1024):
                        raise ValueError("重试记录路径无效或超过大小上限")
                    previous = [json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines() if line]
                else:
                    previous = []
                if len(previous) >= 1000:
                    raise ValueError("该评测 Run 的重试次数已达到上限")
                attempt_number = 1 + sum(1 for item in previous if item.get("question_id") == row["id"])
                attempt: dict[str, object] = {
                    "source_run_id": run_id, "question_id": row["id"], "attempt_number": attempt_number,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "judge_outcome": "provider_error", "judge_rationale": None,
                    "judge_error": None, "judge_request_uncertain": False, "provider_request_id": None,
                }
                try:
                    decision, evidence = BailianChoiceJudge(profile).judge(record, row["raw_output"])
                    attempt.update({
                        "judge_outcome": decision.outcome, "judge_rationale": decision.rationale,
                        "provider_request_id": evidence.get("provider_request_id"),
                    })
                except ProviderError as exc:
                    attempt.update({"judge_error": exc.category, "judge_request_uncertain": exc.uncertain,
                    "provider_request_id": (exc.provider_request_id if isinstance(exc.provider_request_id, str)
                        and len(exc.provider_request_id) <= 200
                        and re.fullmatch(r"[A-Za-z0-9._:-]+", exc.provider_request_id) else None)})
                except Exception:
                    attempt.update({"judge_error": "judge_runtime_error"})
                with attempts_path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(attempt, ensure_ascii=False) + "\n")
                    stream.flush()
                return attempt
        except HTTPException:
            raise
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(409, f"百炼重试未完成：{str(exc)[:240]}") from exc

    @app.get("/api/launcher/runs/{run_id}/models/{model_id}")
    def launcher_model_artifact(run_id: str, model_id: str):
        launcher = app.state.experiment_launcher
        if launcher is None:
            raise HTTPException(404, "Launcher 未启用")
        if not re.fullmatch(r"run-[a-z0-9][a-z0-9-]{0,119}", run_id) or not re.fullmatch(r"model-[a-f0-9]{20}", model_id):
            raise HTTPException(404, "Unknown model artifact")
        state = launcher.get_run(run_id)
        if state is None:
            raise HTTPException(404, "Unknown run")
        path = workspace / str(state.get("output", "")) / "model_artifacts" / f"{model_id}.json"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 or not path.resolve().is_relative_to(workspace):
            raise HTTPException(404, "Model artifact 尚不可用")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            raise HTTPException(422, "Model artifact 无法读取") from exc
        if not isinstance(value, dict) or value.get("model_id") != model_id:
            raise HTTPException(422, "Model artifact 身份不匹配")
        return value

    def source_configs() -> dict[str, Path]:
        directory = workspace / "configs" / "data" / "policy-sources"
        if directory.is_symlink() or not directory.is_dir() or not directory.resolve().is_relative_to(workspace):
            return {}
        found = {}
        for path in directory.glob("*.yaml"):
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024 or not path.resolve().is_relative_to(directory.resolve()):
                continue
            identity = hashlib.sha256(path.name.encode()).hexdigest()[:16]
            found[identity] = path
        return found

    @app.get("/api/policies/capabilities")
    def policy_capabilities():
        sources = []
        discovered = source_configs()
        for identity, path in discovered.items():
            try:
                source = load_source(path)
                available = bool(source.benchmark_jsonl and Path(source.benchmark_jsonl).is_file())
                if source.input_jsonl:
                    available = available and Path(source.input_jsonl).is_file()
                if source.source_cache:
                    cache = Path(source.source_cache)
                    available = available and (cache.is_file() or (cache.is_dir() and (cache / "manifest.json").is_file()))
                sources.append({"id": identity, "title": source.title, "stage": source.existing_stage,
                                "input_scope": source.input_scope, "available": available,
                                "config_ref": path.relative_to(workspace).as_posix(),
                                "tokenizer_repo": source.tokenizer.repo, "tokenizer_revision": source.tokenizer.revision,
                                "test_only": source.tokenizer.test_only})
            except (OSError, ValueError, TypeError):
                continue
        return {"enabled": enable_policy_editor, "token": app.state.policy_editor_token,
                "templates": policy_templates(), "sources": sources}

    @app.get("/api/targeted/capabilities")
    def targeted_capabilities():
        """Return version-controlled examples, never inspect process secrets."""
        examples = workspace / "configs" / "synthetic" / "examples"
        if examples.is_symlink() or not examples.is_dir() or not examples.resolve().is_relative_to(workspace):
            return {"available": False, "profiles": [], "prompt": None, "protocol": None}

        def read_example(name: str) -> dict | None:
            path = examples / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
                return None
            try:
                value = yaml.safe_load(path.read_text(encoding="utf-8"))
                return value if isinstance(value, dict) else None
            except (OSError, UnicodeError, yaml.YAMLError):
                return None

        profiles = []
        for filename in ("provider.fake.yaml", "provider.openai-compatible.example.yaml"):
            value = read_example(filename)
            if not value:
                continue
            # Explicit allowlist: don't echo arbitrary keys from a local config.
            profiles.append({key: value.get(key) for key in (
                "profile_id", "adapter", "endpoint", "model", "api_key_env",
                "supported_parameters", "supports_json_object", "revision",
                "input_cost_per_million_tokens", "output_cost_per_million_tokens",
            ) if key in value})
        prompt = read_example("prompt.sft-math.yaml")
        protocol = read_example("output.sft-candidate.yaml")
        return {
            "available": bool(profiles and prompt and protocol),
            "profiles": profiles,
            "prompt": ({key: prompt.get(key) for key in ("template_id", "version", "template", "allowed_variables")}
                       if prompt else None),
            "protocol": ({key: protocol.get(key) for key in ("protocol_id", "version", "response_format", "required_fields")}
                         if protocol else None),
        }

    @app.get("/api/targeted/stages")
    def targeted_stage_capabilities():
        """Report per-stage prerequisites without issuing provider or training calls."""
        try:
            return stage_capabilities(
                workspace,
                synthetic_generation_enabled=enable_synthetic_generation,
                experiment_launcher_enabled=enable_experiment_launcher,
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(500, "阶段能力预检读取失败") from exc

    @app.get("/api/training/stage-profiles", response_model=StageProfileCatalog)
    def training_stage_profiles():
        """Return the closed, typed registry of approved stage training/evaluation profiles."""
        try:
            return stage_profile_catalog(workspace)
        except (OSError, ValueError) as exc:
            raise HTTPException(500, "阶段配置目录读取失败") from exc

    @app.get("/api/training/workbench-options")
    def training_workbench_options():
        """Small local selector catalog; every selected identity is rechecked by preflight."""
        try:
            return stage_workbench_options(workspace)
        except (OSError, ValueError) as exc:
            raise HTTPException(500, "阶段工作台选项读取失败") from exc

    @app.post("/api/training/preflight")
    def training_preflight(payload: StagePreflightRequest):
        """Read-only compatibility/evidence check; it never creates a run or token."""
        try:
            return preflight_stage(
                workspace, payload, launcher_enabled=enable_experiment_launcher,
            )
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(422, "阶段预检无法读取指定登记证据") from exc

    @app.post("/api/training/preflight/confirm")
    def confirm_training_preflight(payload: StagePreflightRequest):
        """Issue a short-lived capability bound to current evidence; this does not start training."""
        if allow_remote:
            raise HTTPException(403, "阶段训练确认仅允许本机 loopback 服务")
        try:
            summary = preflight_stage(
                workspace, payload, launcher_enabled=enable_experiment_launcher,
            )
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(422, "阶段预检无法读取指定登记证据") from exc
        if summary["status"] != "ready_for_confirmation":
            raise HTTPException(409, detail=summary)
        return app.state.stage_preflight_confirmations.issue(payload, summary)

    @app.post("/api/training/cpt/preflight", response_model=CPTPreflightResponse)
    def cpt_training_preflight(payload: CPTPreflightRequest):
        """CPT uses its fixed Base model root and registered token corpus; never downloads weights."""
        try:
            return CPTStageAdapter(workspace).preflight(payload.dataset_id)
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(422, "CPT 阶段预检无法读取登记数据或固定配置") from exc

    @app.post("/api/training/stage/preflight")
    def registered_stage_preflight(request: Request, payload: RegisteredStageTrainingRequest,
                                   x_local_launcher_token: str | None = Header(default=None)):
        require_launcher(request, x_local_launcher_token)
        service = app.state.stage_training
        if service is None:
            raise HTTPException(403, "本地阶段训练启动未启用")
        try:
            return service.preflight(payload)
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)[:500]) from exc

    @app.post("/api/training/stage/confirm")
    def registered_stage_confirm(request: Request, payload: RegisteredStageTrainingRequest,
                                 x_local_launcher_token: str | None = Header(default=None)):
        require_launcher(request, x_local_launcher_token)
        service = app.state.stage_training
        if service is None or allow_remote:
            raise HTTPException(403, "阶段训练确认仅允许启用 Launcher 的本机 loopback 服务")
        try:
            result = service.confirm(payload)
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)[:500]) from exc
        if result.get("status") != "ready_for_confirmation":
            raise HTTPException(409, detail=result)
        return result

    @app.post("/api/training/stage/start")
    def registered_stage_start(request: Request, payload: dict,
                                x_local_launcher_token: str | None = Header(default=None)):
        require_launcher(request, x_local_launcher_token)
        service = app.state.stage_training
        if service is None or allow_remote:
            raise HTTPException(403, "阶段训练启动仅允许启用 Launcher 的本机 loopback 服务")
        allowed = {"stage", "profile_id", "dataset_ids", "parent_model_id", "parameter_overrides",
                   "memory_mode_id", "validation", "checkpoint_policy", "confirmation_token", "acknowledged_dataset_id"}
        if set(payload) - allowed:
            raise HTTPException(422, "阶段启动请求包含不支持字段；不接受 recipe、路径或任意命令")
        try:
            training_request = RegisteredStageTrainingRequest.model_validate({
                key: payload[key] for key in ("stage", "profile_id", "dataset_ids", "parent_model_id", "parameter_overrides", "memory_mode_id", "validation", "checkpoint_policy")
                if key in payload
            })
            return service.start(training_request, payload.get("confirmation_token", ""),
                                 payload.get("acknowledged_dataset_id"))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)[:500]) from exc
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(409, str(exc)[:500]) from exc

    @app.get("/api/targeted/overview")
    def targeted_overview():
        """Read bounded, allowlisted summaries; private candidate/request data is never opened."""
        root = workspace / "artifacts" / "synthetic"
        if root.is_symlink() or not root.is_dir() or not root.resolve().is_relative_to(workspace):
            return {"available": False, "scenarios": [], "plans": [], "generations": [], "effects": []}
        scenarios, plans, generations, effects = [], [], [], []
        visited = 0
        total_bytes = 0
        allowed_files = {"scenario_report.json", "plan.json", "registered_plan.json",
                         "generation_validation_cost.json", "remediation_report.json"}
        for current, dirs, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            try:
                if not current_path.resolve().is_relative_to(root.resolve()):
                    dirs[:] = []
                    continue
                rel_depth = len(current_path.relative_to(root).parts)
            except (OSError, ValueError):
                dirs[:] = []
                continue
            dirs[:] = [name for name in dirs if name not in {"private", "responses", "canonical"}
                       and not (current_path / name).is_symlink()
                       and (current_path / name).resolve().is_relative_to(root.resolve())]
            if rel_depth > 5:
                dirs[:] = []
                continue
            for name in files:
                if name not in allowed_files:
                    continue
                visited += 1
                if visited > 500:
                    break
                path = current_path / name
                try:
                    size = path.stat().st_size
                    if path.is_symlink() or size > 512 * 1024 or total_bytes + size > 5 * 1024 * 1024:
                        continue
                    total_bytes += size
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(value, dict):
                        continue
                    if name == "scenario_report.json":
                        for item in value.get("scenarios", [])[:100]:
                            if isinstance(item, dict):
                                scenarios.append({key: item.get(key) for key in (
                                    "scenario_id", "sample_count", "denominator", "correct", "incorrect",
                                    "parse_failed", "unsupported", "missing_prediction", "accuracy", "status",
                                )} | {key: value.get(key) for key in (
                                    "report_id", "eval_id", "model_id", "dataset_id", "protocol_version",
                                )})
                    elif name == "plan.json":
                        plan_value = value.get("plan") if isinstance(value.get("plan"), dict) else value
                        scenarios_plan = {key: value.get(key, plan_value.get(key)) for key in (
                            "plan_id", "version", "target_scenario_id", "stage", "comparison_design",
                            "provider_profile_id", "training_budget", "seed",
                        )}
                        target = plan_value.get("target")
                        budget = plan_value.get("budget")
                        scenarios_plan["target"] = ({key: target.get(key) for key in (
                            "name", "direction", "threshold", "minimum_samples",
                        )} if isinstance(target, dict) else None)
                        scenarios_plan["budget"] = ({key: budget.get(key) for key in (
                            "max_requests", "max_output_tokens", "max_retries", "timeout_seconds", "target_candidates",
                        )} if isinstance(budget, dict) else None)
                        for ref_name in ("diagnostic", "training_seed", "starting_model", "final_evaluation"):
                            ref = plan_value.get(ref_name)
                            if isinstance(ref, dict):
                                scenarios_plan[ref_name] = {key: ref.get(key) for key in (
                                    "eval_id", "model_id", "dataset_id", "content_sha256", "protocol_version", "role",
                                )}
                        plans.append(scenarios_plan)
                    elif name == "registered_plan.json":
                        plan_value = value.get("plan") if isinstance(value.get("plan"), dict) else {}
                        scenarios_plan = {key: value.get(key, plan_value.get(key)) for key in (
                            "plan_id", "version", "target_scenario_id", "stage", "comparison_design",
                            "provider_profile_id", "training_budget", "seed",
                        )}
                        target = plan_value.get("target")
                        budget = plan_value.get("budget")
                        scenarios_plan["target"] = ({key: target.get(key) for key in (
                            "name", "direction", "threshold", "minimum_samples",
                        )} if isinstance(target, dict) else None)
                        scenarios_plan["budget"] = ({key: budget.get(key) for key in (
                            "max_requests", "max_output_tokens", "max_retries", "timeout_seconds", "target_candidates",
                        )} if isinstance(budget, dict) else None)
                        for ref_name in ("diagnostic", "training_seed", "starting_model", "final_evaluation"):
                            ref = plan_value.get(ref_name)
                            if isinstance(ref, dict):
                                scenarios_plan[ref_name] = {key: ref.get(key) for key in (
                                    "eval_id", "model_id", "dataset_id", "content_sha256", "protocol_version", "role",
                                )}
                        plans.append(scenarios_plan)
                    elif name == "generation_validation_cost.json":
                        gen = value.get("generation") if isinstance(value.get("generation"), dict) else {}
                        ledger = gen.get("ledger") if isinstance(gen.get("ledger"), dict) else {}
                        generations.append({key: gen.get(key) for key in (
                            "execution_status", "request_count", "successful_candidates", "target_candidate_count",
                            "shortfall", "stop_reason", "cache_reused_count", "network_used",
                        )} | {"cost_status": ledger.get("cost_status", "unknown"),
                             "request_attempts": ledger.get("request_attempts"),
                             "estimated_cost_usd": ledger.get("estimated_cost_usd"),
                             "provider_profile_id": value.get("provider_profile_id"),
                             "provider_adapter": value.get("provider_adapter"),
                             "model": value.get("model")})
                    elif name == "remediation_report.json":
                        effects.append({key: value.get(key) for key in (
                            "status", "before_eval_id", "after_eval_id", "before_model_id", "after_model_id",
                            "scenario_id", "target", "guardrails", "actual_added_records", "planned_training_budget",
                            "actual_training_budget", "budget_status", "cost_status", "estimated_cost_usd",
                            "comparison_limitations",
                        )})
                except (OSError, ValueError, UnicodeError, TypeError):
                    continue
        # GRPO stage-effect reports are stored with their Evaluation Runs, not
        # under synthetic artifacts. Read only this small, allowlisted summary.
        evaluations_root = workspace / "runs" / "evaluations"
        if (not evaluations_root.is_symlink() and evaluations_root.is_dir()
                and evaluations_root.resolve().is_relative_to(workspace)):
            for path in list(evaluations_root.glob("**/effect_report.json"))[:200]:
                try:
                    if (path.is_symlink() or not path.is_file()
                            or not path.resolve().is_relative_to(evaluations_root.resolve())):
                        continue
                    size = path.stat().st_size
                    if size > 256 * 1024 or total_bytes + size > 5 * 1024 * 1024:
                        continue
                    total_bytes += size
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(value, dict) or value.get("schema_version") != "1.0":
                        continue
                    target = value.get("target_metric") if isinstance(value.get("target_metric"), dict) else None
                    effects.append({key: value.get(key) for key in (
                        "status", "target_state", "targeted_plan_id", "targeted_plan_version",
                        "training_run_id", "model_id", "before_eval_id", "after_eval_id",
                        "comparison_limitations",
                    )} | {"stage": "grpo", "target_metric": (None if target is None else {
                        key: target.get(key) for key in ("name", "direction", "threshold", "threshold_source",
                                                       "before", "after", "delta", "threshold_status")}),
                        "guardrail_metrics": value.get("guardrail_metrics", {})})
                except (OSError, ValueError, UnicodeError, TypeError):
                    continue
        return {"available": True, "scenarios": scenarios[:200], "plans": plans[:100],
                "generations": generations[:100], "effects": effects[:100], "truncated": visited > 500}

    def require_synthetic_generation(request: Request, token: str | None) -> SFTGenerationJobs:
        jobs = app.state.sft_generation_jobs
        if jobs is None:
            raise HTTPException(403, "模型生成未启用；请使用 --enable-synthetic-generation 启动本地 UI")
        if token != app.state.synthetic_generation_token:
            raise HTTPException(403, "本地模型生成令牌无效")
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.url.netloc}":
            raise HTTPException(403, "只接受同源模型生成请求")
        client_host = request.client.host if request.client else ""
        try:
            client_ok = ipaddress.ip_address(client_host).is_loopback
        except ValueError:
            client_ok = client_host in {"testclient", "localhost"}
        try:
            host_ok = ipaddress.ip_address(request.url.hostname or "").is_loopback
        except ValueError:
            host_ok = (request.url.hostname or "").lower() in {"localhost", "testserver"}
        if not client_ok or not host_ok:
            raise HTTPException(403, "模型生成仅接受 loopback 本机请求")
        return jobs

    def require_preference_generation(request: Request, token: str | None) -> PreferenceGenerationJobs:
        require_synthetic_generation(request, token)
        jobs = app.state.preference_generation_jobs
        if jobs is None:
            raise HTTPException(403, "Preference 生成未启用")
        return jobs

    def require_grpo_generation(request: Request, token: str | None) -> GRPOGenerationJobs:
        require_synthetic_generation(request, token)
        jobs = app.state.grpo_generation_jobs
        if jobs is None:
            raise HTTPException(403, "GRPO 候选生成未启用")
        return jobs

    @app.get("/api/synthetic/grpo/capabilities")
    def grpo_generation_capabilities():
        jobs = app.state.grpo_generation_jobs
        if jobs is None:
            return {"enabled": False, "read_only": True,
                    "reason": "请使用 --enable-synthetic-generation 显式启用"}
        from .sft_generation_setup import _profiles
        profiles = [item for item in _profiles(Path(workspace))
                    if item.get("adapter") == "openai_compatible" and item.get("credential_env_configured")
                    and not item.get("is_example")]
        return {"enabled": True, "local_only": True, "token": app.state.synthetic_generation_token,
                "profiles": profiles, "targeted_plans": jobs.targeted_plans(),
                "warning": "模型只生成 GRPO prompt 候选；没有可信 reference 的候选不可登记或训练。"}

    @app.post("/api/synthetic/grpo/prepare")
    def grpo_generation_prepare(request: Request, payload: dict,
            x_local_synthetic_token: str | None = Header(default=None)):
        jobs = require_grpo_generation(request, x_local_synthetic_token)
        try:
            return jobs.prepare(payload)
        except (OSError, TypeError, ValueError, FileExistsError) as exc:
            raise HTTPException(422, f"GRPO 候选方案准备失败：{str(exc)[:300]}") from exc

    @app.post("/api/synthetic/grpo/preflight")
    def grpo_generation_preflight(request: Request, payload: dict,
            x_local_synthetic_token: str | None = Header(default=None)):
        jobs = require_grpo_generation(request, x_local_synthetic_token)
        if set(payload) != {"bundle_id"} or not isinstance(payload.get("bundle_id"), str):
            raise HTTPException(422, "预检只接受已准备的 bundle_id")
        try:
            return jobs.preflight(payload["bundle_id"])
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(422, f"GRPO 候选预检未通过：{str(exc)[:300]}") from exc

    @app.post("/api/synthetic/grpo/start", status_code=202)
    def grpo_generation_start(request: Request, payload: dict,
            x_local_synthetic_token: str | None = Header(default=None)):
        jobs = require_grpo_generation(request, x_local_synthetic_token)
        if set(payload) != {"bundle_id", "confirm_start"} or not isinstance(payload.get("bundle_id"), str):
            raise HTTPException(422, "启动只接受 bundle_id 和显式确认")
        try:
            return jobs.start(payload["bundle_id"], confirm_start=payload.get("confirm_start"))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(422, f"GRPO 候选未启动：{str(exc)[:300]}") from exc

    @app.get("/api/synthetic/grpo/runs/{execution_id}")
    def grpo_generation_status(request: Request, execution_id: str,
            x_local_synthetic_token: str | None = Header(default=None)):
        jobs = require_grpo_generation(request, x_local_synthetic_token)
        state = jobs.get(execution_id)
        if state is None:
            raise HTTPException(404, "Unknown GRPO generation execution")
        return state

    @app.get("/api/synthetic/grpo/runs/{execution_id}/preview")
    def grpo_generation_preview(request: Request, execution_id: str,
            x_local_synthetic_token: str | None = Header(default=None)):
        jobs = require_grpo_generation(request, x_local_synthetic_token)
        try:
            return jobs.preview(execution_id)
        except KeyError as exc:
            raise HTTPException(404, "Unknown GRPO generation execution") from exc
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(409, f"GRPO 候选预览不可用：{str(exc)[:300]}") from exc

    @app.post("/api/synthetic/grpo/runs/{execution_id}/cancel")
    def grpo_generation_cancel(request: Request, execution_id: str, payload: dict,
            x_local_synthetic_token: str | None = Header(default=None)):
        jobs = require_grpo_generation(request, x_local_synthetic_token)
        if payload:
            raise HTTPException(422, "取消请求不接受额外字段")
        try:
            return jobs.cancel(execution_id)
        except KeyError as exc:
            raise HTTPException(404, "Unknown GRPO generation execution") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/synthetic/preference/capabilities")
    def preference_generation_capabilities():
        jobs = app.state.preference_generation_jobs
        if jobs is None:
            return {"enabled": False, "read_only": True,
                    "reason": "请使用 --enable-synthetic-generation 显式启用"}
        from .sft_generation_setup import _profiles
        profiles = [item for item in _profiles(Path(workspace))
                    if item.get("adapter") == "openai_compatible" and item.get("credential_env_configured")
                    and not item.get("is_example")]
        return {"enabled": True, "local_only": True, "read_only": False,
                "token": app.state.synthetic_generation_token, "profiles": profiles,
                "plans": jobs.plans(), "targeted_plans": jobs.targeted_plans()}

    @app.post("/api/synthetic/preference/prepare")
    def preference_generation_prepare(request: Request, payload: dict,
            x_local_synthetic_token: str | None = Header(default=None)):
        jobs = require_preference_generation(request, x_local_synthetic_token)
        try:
            return jobs.prepare(payload)
        except (OSError, TypeError, ValueError, FileExistsError) as exc:
            raise HTTPException(422, "Preference 方案准备失败；请核对模式、训练种子、计划身份、Provider 和预算") from exc

    @app.post("/api/synthetic/preference/preflight")
    def preference_generation_preflight(request: Request, payload: dict,
            x_local_synthetic_token: str | None = Header(default=None)):
        jobs = require_preference_generation(request, x_local_synthetic_token)
        if set(payload) != {"bundle_id"} or not isinstance(payload.get("bundle_id"), str):
            raise HTTPException(422, "Preference 预检只接受已登记 bundle_id")
        try:
            return jobs.preflight(payload["bundle_id"])
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(422, "Preference 预检未通过；Provider 不会被调用") from exc

    @app.post("/api/synthetic/preference/start", status_code=202)
    def preference_generation_start(request: Request, payload: dict,
            x_local_synthetic_token: str | None = Header(default=None)):
        jobs = require_preference_generation(request, x_local_synthetic_token)
        if set(payload) != {"bundle_id", "confirm_start"} or not isinstance(payload.get("bundle_id"), str):
            raise HTTPException(422, "启动只接受 bundle_id 与显式确认")
        try:
            return jobs.start(payload["bundle_id"], confirm_start=payload.get("confirm_start"))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(422, "Preference 生成未启动；请重新预检") from exc

    @app.get("/api/synthetic/preference/runs/{execution_id}")
    def preference_generation_status(request: Request, execution_id: str,
            x_local_synthetic_token: str | None = Header(default=None)):
        jobs = require_preference_generation(request, x_local_synthetic_token)
        state = jobs.get(execution_id)
        if state is None:
            raise HTTPException(404, "Unknown Preference generation execution")
        return state

    @app.get("/api/synthetic/preference/runs/{execution_id}/preview")
    def preference_generation_preview(request: Request, execution_id: str,
            x_local_synthetic_token: str | None = Header(default=None)):
        jobs = require_preference_generation(request, x_local_synthetic_token)
        try:
            return jobs.preview(execution_id)
        except KeyError as exc:
            raise HTTPException(404, "Unknown Preference generation execution") from exc
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(409, "候选预览暂不可用；请检查任务状态") from exc

    @app.post("/api/synthetic/preference/runs/{execution_id}/review")
    def preference_generation_review(request: Request, execution_id: str, payload: dict,
            x_local_synthetic_token: str | None = Header(default=None)):
        jobs = require_preference_generation(request, x_local_synthetic_token)
        try:
            return jobs.review(execution_id, payload)
        except KeyError as exc:
            raise HTTPException(404, "Unknown Preference generation execution") from exc
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(422, "人工排序未保存；请核对候选 ID、同题约束及审阅者标识") from exc

    @app.post("/api/synthetic/preference/runs/{execution_id}/register")
    def preference_generation_register(request: Request, execution_id: str, payload: dict,
            x_local_synthetic_token: str | None = Header(default=None)):
        jobs = require_preference_generation(request, x_local_synthetic_token)
        if payload != {"confirm_reviewed": True}:
            raise HTTPException(422, "登记前须确认已审核；定向偏好对也需检查证据与来源许可")
        try:
            return jobs.register(execution_id)
        except KeyError as exc:
            raise HTTPException(404, "Unknown Preference generation execution") from exc
        except (OSError, TypeError, ValueError, FileExistsError) as exc:
            raise HTTPException(409, "Preference 数据集未登记；零可用 pair 不会被登记") from exc

    @app.get("/api/synthetic/sft/capabilities")
    def sft_generation_capabilities():
        jobs = app.state.sft_generation_jobs
        if jobs is None:
            return {"enabled": False, "read_only": True,
                    "reason": "请使用 --enable-synthetic-generation 显式启用"}
        return {"enabled": True, "read_only": False, "local_only": True,
                "token": app.state.synthetic_generation_token, "bundles": jobs.bundles()}

    @app.get("/api/synthetic/sft/readiness")
    def sft_generation_readiness(request: Request,
                                 x_local_synthetic_token: str | None = Header(default=None)):
        require_synthetic_generation(request, x_local_synthetic_token)
        try:
            return discover_sft_generation_setup(workspace)
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(500, "本地 SFT 准备状态读取失败") from exc

    @app.get("/api/synthetic/cpt/capabilities")
    def cpt_generation_capabilities():
        jobs = app.state.cpt_generation_jobs
        if jobs is None:
            return {"enabled": False, "read_only": True,
                    "reason": "请使用 --enable-synthetic-generation 显式启用"}
        from ..synthetic.providers import ProviderProfile
        from .sft_generation_setup import _credential_available
        profiles = []
        for path in (Path(workspace) / "configs" / "synthetic").rglob("provider*.yaml"):
            if path.is_symlink():
                continue
            try:
                profile = ProviderProfile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
                if (profile.adapter == "openai_compatible" and _credential_available(Path(workspace), profile.api_key_env)
                        and "example" not in path.name):
                    profiles.append({"profile_id": profile.profile_id, "model": profile.model})
            except (OSError, TypeError, ValueError, yaml.YAMLError):
                continue
        return {"enabled": True, "read_only": False, "local_only": True,
                "token": app.state.synthetic_generation_token, "profiles": profiles}

    @app.post("/api/synthetic/cpt/prepare")
    def cpt_generation_prepare(request: Request, payload: dict,
                               x_local_synthetic_token: str | None = Header(default=None)):
        require_synthetic_generation(request, x_local_synthetic_token)
        allowed = {"mode", "domain", "topic", "target_token_budget", "request_budget", "candidate_budget",
                   "max_output_tokens", "timeout_seconds", "profile_id", "plan_id", "plan_version"}
        required = {"mode", "domain", "topic", "target_token_budget", "request_budget", "candidate_budget",
                    "max_output_tokens", "profile_id"}
        if not required.issubset(payload) or set(payload) - allowed:
            raise HTTPException(422, "CPT 方案仅接受模式、主题、有限预算、Provider profile 与可选定向计划身份")
        if (payload.get("mode") not in {"ordinary", "targeted"}
                or not isinstance(payload.get("domain"), str) or not isinstance(payload.get("topic"), str)
                or not isinstance(payload.get("profile_id"), str)
                or any(not isinstance(payload.get(key), int) or isinstance(payload.get(key), bool)
                       or payload[key] < 1 for key in ("target_token_budget", "request_budget", "candidate_budget",
                                                        "max_output_tokens"))
                or ("timeout_seconds" in payload and
                    (not isinstance(payload["timeout_seconds"], (int, float))
                     or isinstance(payload["timeout_seconds"], bool)
                     or not 1 <= payload["timeout_seconds"] <= 600))):
            raise HTTPException(422, "CPT 方案字段格式无效")
        try:
            return prepare_cpt_generation(workspace, **payload)
        except (OSError, TypeError, ValueError, FileExistsError) as exc:
            raise HTTPException(422, "CPT 方案准备失败；请核对主题、预算、Provider 与定向计划身份") from exc

    @app.post("/api/synthetic/cpt/preflight")
    def cpt_generation_preflight(request: Request, payload: dict,
                                 x_local_synthetic_token: str | None = Header(default=None)):
        require_synthetic_generation(request, x_local_synthetic_token)
        if set(payload) != {"bundle_id"} or not isinstance(payload.get("bundle_id"), str):
            raise HTTPException(422, "CPT 预检只接受已登记 bundle_id")
        try:
            return preflight_cpt_generation(workspace, payload["bundle_id"])
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(422, "CPT 预检未通过；请检查本地 tokenizer、污染索引和 Provider 凭据") from exc

    @app.get("/api/synthetic/cpt/request-review/{bundle_id}")
    def cpt_request_review(request: Request, bundle_id: str,
                           x_local_synthetic_token: str | None = Header(default=None)):
        require_synthetic_generation(request, x_local_synthetic_token)
        try:
            return {"bundle_id": bundle_id, "read_only": True, "provider_calls": 0,
                    "unresolved": app.state.cpt_generation_jobs.request_review(bundle_id)}
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(422, "无法读取该 CPT 方案的 Provider 请求状态") from exc

    @app.post("/api/synthetic/cpt/request-review/{bundle_id}")
    def cpt_request_review_confirm(request: Request, bundle_id: str, payload: dict,
                                   x_local_synthetic_token: str | None = Header(default=None)):
        require_synthetic_generation(request, x_local_synthetic_token)
        if (set(payload) != {"attempt_id", "confirm_billing_risk"}
                or not isinstance(payload.get("attempt_id"), str)
                or not re.fullmatch(r"[a-f0-9]{32}", payload["attempt_id"])
                or payload.get("confirm_billing_risk") is not True):
            raise HTTPException(422, "须指定未知请求并明确确认可能重复计费")
        try:
            return {"bundle_id": bundle_id,
                    "unresolved": app.state.cpt_generation_jobs.request_review(bundle_id,
                                                                               attempt_id=payload["attempt_id"]),
                    "provider_calls": 0, "note": "原请求结果仍未知；不会自动发起新请求"}
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(422, "无法确认该未知请求；请刷新状态") from exc

    @app.post("/api/synthetic/cpt/start", status_code=202)
    def cpt_generation_start(request: Request, payload: dict,
                             x_local_synthetic_token: str | None = Header(default=None)):
        require_synthetic_generation(request, x_local_synthetic_token)
        if set(payload) != {"bundle_id", "confirm_start"} or not isinstance(payload.get("bundle_id"), str):
            raise HTTPException(422, "启动仅接受 bundle_id 与显式确认")
        try:
            return app.state.cpt_generation_jobs.start(payload["bundle_id"],
                                                       confirm_start=payload.get("confirm_start"))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(422, "CPT 生成未启动；请重新预检并确认本地资源") from exc

    @app.get("/api/synthetic/cpt/runs/{execution_id}")
    def cpt_generation_status(request: Request, execution_id: str,
                              x_local_synthetic_token: str | None = Header(default=None)):
        require_synthetic_generation(request, x_local_synthetic_token)
        state = app.state.cpt_generation_jobs.get(execution_id)
        if state is None:
            raise HTTPException(404, "Unknown CPT generation execution")
        return state

    @app.get("/api/synthetic/cpt/runs/{execution_id}/preview")
    def cpt_generation_preview(request: Request, execution_id: str,
                               x_local_synthetic_token: str | None = Header(default=None)):
        require_synthetic_generation(request, x_local_synthetic_token)
        try:
            return app.state.cpt_generation_jobs.preview(execution_id)
        except KeyError as exc:
            raise HTTPException(404, "Unknown CPT generation execution") from exc
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(409, "CPT 候选暂不可预览；任务可能尚未结束") from exc

    @app.post("/api/synthetic/cpt/runs/{execution_id}/register")
    def cpt_generation_register(request: Request, execution_id: str, payload: dict,
                                x_local_synthetic_token: str | None = Header(default=None)):
        require_synthetic_generation(request, x_local_synthetic_token)
        if payload != {"confirm_reviewed": True}:
            raise HTTPException(422, "登记前需审核候选并确认 confirm_reviewed=true")
        try:
            return app.state.cpt_generation_jobs.register(execution_id)
        except KeyError as exc:
            raise HTTPException(404, "Unknown CPT generation execution") from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (OSError, TypeError, ValueError, FileExistsError) as exc:
            raise HTTPException(422, "CPT 候选登记失败；检查 canonical schema 和质量报告") from exc

    @app.post("/api/synthetic/cpt/runs/{execution_id}/cancel")
    def cpt_generation_cancel(request: Request, execution_id: str, payload: dict,
                              x_local_synthetic_token: str | None = Header(default=None)):
        require_synthetic_generation(request, x_local_synthetic_token)
        if payload:
            raise HTTPException(422, "取消请求不接受参数")
        try:
            return app.state.cpt_generation_jobs.cancel(execution_id)
        except KeyError as exc:
            raise HTTPException(404, "Unknown CPT generation execution") from exc
        except RuntimeError as exc:
            raise HTTPException(409, "CPT 任务当前不可取消") from exc

    @app.post("/api/synthetic/sft/prepare")
    def sft_generation_prepare(request: Request, payload: dict,
                               x_local_synthetic_token: str | None = Header(default=None)):
        require_synthetic_generation(request, x_local_synthetic_token)
        if not isinstance(payload, dict):
            raise HTTPException(422, "准备请求格式无效")
        mode = payload.get("mode", "seeded")
        if mode == "prompt_only":
            allowed = {"mode", "topic", "skill", "difficulty", "constraints", "target_count", "profile_id"}
            required_fields = {"mode", "topic", "skill", "difficulty", "target_count"}
            if not required_fields.issubset(payload) or set(payload) - allowed:
                raise HTTPException(422, "按主题生成只接受主题、能力点、难度、可选约束、目标数和 Provider profile")
            if (any(not isinstance(payload.get(key), str) for key in ("topic", "skill", "difficulty"))
                    or not isinstance(payload["target_count"], int) or isinstance(payload["target_count"], bool)
                    or not 1 <= payload["target_count"] <= 1000
                    or ("constraints" in payload and not isinstance(payload["constraints"], str))
                    or ("profile_id" in payload and not isinstance(payload["profile_id"], str))):
                raise HTTPException(422, "按主题生成字段格式无效")
            try:
                return prepare_prompt_only_sft_generation(
                    workspace, topic=payload["topic"], skill=payload["skill"],
                    difficulty=payload["difficulty"], constraints=payload.get("constraints", ""),
                    target_count=payload["target_count"], profile_id=payload.get("profile_id"),
                )
            except (OSError, TypeError, ValueError, FileExistsError) as exc:
                raise HTTPException(422, "按主题生成方案准备失败；请检查 Provider 配置及输入范围") from exc
        allowed = {"dataset_id", "target_count", "profile_id"}
        if mode != "seeded" or not {"dataset_id", "target_count"}.issubset(payload) or set(payload) - allowed:
            raise HTTPException(422, "准备请求只接受已有数据集模式的 dataset_id、target_count 和可选 profile_id")
        if (not isinstance(payload["dataset_id"], str) or not payload["dataset_id"]
                or not isinstance(payload["target_count"], int) or isinstance(payload["target_count"], bool)
                or not 1 <= payload["target_count"] <= 10000
                or ("profile_id" in payload and not isinstance(payload["profile_id"], str))):
            raise HTTPException(422, "dataset_id、target_count 或 profile_id 格式无效")
        try:
            return prepare_ordinary_sft_generation(
                workspace, dataset_id=payload["dataset_id"], target_count=payload["target_count"],
                profile_id=payload.get("profile_id"),
            )
        except (OSError, TypeError, ValueError, FileExistsError) as exc:
            raise HTTPException(422, "普通 SFT 方案准备失败；请检查已登记数据、默认配置及本地 Provider 凭据") from exc

    @app.post("/api/synthetic/sft/preflight")
    def sft_generation_preflight(request: Request, payload: dict,
                                 x_local_synthetic_token: str | None = Header(default=None)):
        jobs = require_synthetic_generation(request, x_local_synthetic_token)
        required = {"bundle_id", "usable_target", "max_requests", "target_candidates"}
        allowed = required | {"stage_plan_id", "plan_version"}
        if not required.issubset(payload) or set(payload) - allowed or not isinstance(payload.get("bundle_id"), str):
            raise HTTPException(422, "预检只接受 bundle_id、有限预算及可选的共享 stage_plan_id/version")
        if ("stage_plan_id" in payload) != ("plan_version" in payload):
            raise HTTPException(422, "共享 stage plan 必须同时提供 plan_id 与 version")
        if ("stage_plan_id" in payload and (not isinstance(payload["stage_plan_id"], str)
                or not isinstance(payload["plan_version"], int) or isinstance(payload["plan_version"], bool))):
            raise HTTPException(422, "共享 stage plan 身份无效")
        if any(not isinstance(payload[key], int) or isinstance(payload[key], bool) or payload[key] < 1
               for key in ("usable_target", "max_requests", "target_candidates")):
            raise HTTPException(422, "目标数与预算必须是正整数")
        try:
            return jobs.preflight(payload["bundle_id"], usable_target=payload["usable_target"],
                                  max_requests=payload["max_requests"],
                                  target_candidates=payload["target_candidates"],
                                  stage_plan_id=payload.get("stage_plan_id"), plan_version=payload.get("plan_version"))
        except UncertainProviderRequest as exc:
            raise HTTPException(409, str(exc)) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(422, "生成预检失败；请检查已登记计划与本地输入摘要") from exc

    @app.get("/api/synthetic/sft/request-review/{bundle_id}")
    def sft_request_review_status(request: Request, bundle_id: str,
                                  x_local_synthetic_token: str | None = Header(default=None)):
        jobs = require_synthetic_generation(request, x_local_synthetic_token)
        try:
            return {"bundle_id": bundle_id, "read_only": True, "provider_calls": 0,
                    "unresolved": jobs.request_review(bundle_id)}
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(422, "无法读取该方案的请求审查状态") from exc

    @app.post("/api/synthetic/sft/request-review/{bundle_id}")
    def sft_request_review_confirm(request: Request, bundle_id: str, payload: dict,
                                   x_local_synthetic_token: str | None = Header(default=None)):
        jobs = require_synthetic_generation(request, x_local_synthetic_token)
        if (set(payload) != {"attempt_id", "confirm_billing_risk"}
                or not isinstance(payload.get("attempt_id"), str)
                or not re.fullmatch(r"[a-f0-9]{32}", payload["attempt_id"])
                or payload.get("confirm_billing_risk") is not True):
            raise HTTPException(422, "须指定未知请求并明确确认可能重复计费")
        try:
            return {"bundle_id": bundle_id, "unresolved": jobs.request_review(bundle_id, attempt_id=payload["attempt_id"]),
                    "provider_calls": 0, "note": "原请求结果仍未知；不会自动发起新请求"}
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(422, "无法确认该请求；请刷新状态") from exc

    @app.post("/api/synthetic/stage/preflight")
    def synthetic_stage_preflight(request: Request, payload: dict,
                                  x_local_synthetic_token: str | None = Header(default=None)):
        """Read-only stage preflight; only the established SFT adapter accepts a bundle today."""
        require_synthetic_generation(request, x_local_synthetic_token)
        stage, mode = payload.get("stage"), payload.get("mode")
        if (not isinstance(stage, str) or stage not in {"cpt", "sft", "preference", "grpo"}
                or not isinstance(mode, str) or mode not in {"ordinary", "targeted"}):
            raise HTTPException(422, "不支持的阶段或生成模式")
        if stage != "sft":
            if set(payload) != {"stage", "mode"}:
                raise HTTPException(422, "未接通阶段只接受阶段与模式")
            catalog = stage_capabilities(workspace, synthetic_generation_enabled=True,
                                         experiment_launcher_enabled=enable_experiment_launcher)
            capability = next(item for item in catalog["stages"] if item["stage"] == stage)
            mode_capability = capability["generation_modes"][mode]
            result = {"stage": stage, "mode": mode, "status": "blocked", "read_only": True,
                      "provider_calls": 0, **{key: mode_capability[key] for key in (
                          "adapter_available", "domain_adapter", "candidate_protocol",
                          "quality_verifier", "provider_configured", "blockers", "pending_checks",
                      )}}
            if stage == "grpo":
                reward_available = bool((capability.get("reward_contract") or {}).get("available"))
                result["training_handoff"] = {
                    "status": "blocked",
                    "reference": {"status": "not_provided", "reason": "未选择可核验的可信 reference 证据"},
                    "reward": {"status": "present_unverified" if reward_available else "missing",
                               "reason": ("插件文件存在，但尚未核验版本、协议和实现摘要" if reward_available
                                          else "缺少本地 reward 插件；须提供并核验受控契约")},
                    "note": "训练资格与生成候选分开判断；模型自报答案不构成可信 reference。",
                }
            return result
        required = {"stage", "mode", "bundle_id", "usable_target", "max_requests", "target_candidates"}
        allowed = required | {"stage_plan_id", "plan_version"}
        if not required.issubset(payload) or set(payload) - allowed:
            raise HTTPException(422, "阶段预检只接受登记方案、有限预算和可选定向计划身份")
        if mode == "targeted" and not {"stage_plan_id", "plan_version"}.issubset(payload):
            raise HTTPException(422, "定向生成必须提供计划 ID 和版本")
        if mode == "ordinary" and ({"stage_plan_id", "plan_version"} & set(payload)):
            raise HTTPException(422, "普通生成不能附带定向计划")
        if mode == "ordinary" and isinstance(payload.get("bundle_id"), str) and payload["bundle_id"].startswith("plan-targeted-"):
            raise HTTPException(422, "定向生成方案不能作为普通生成使用")
        legacy_payload = {key: value for key, value in payload.items() if key not in {"stage", "mode"}}
        prepared = sft_generation_preflight(request, legacy_payload, x_local_synthetic_token)
        from ..synthetic.ordinary_bundle import load_ordinary_sft_bundle
        from ..synthetic.prompt_only import load_prompt_only_bundle
        from ..synthetic.providers import ProviderProfile
        from ..synthetic.workflow_bundle import load_sft_workflow_bundle
        from .sft_generation_setup import _credential_available

        bundle_id = payload["bundle_id"]
        loader = (load_prompt_only_bundle if bundle_id.startswith("plan-sft-prompt-") else
                  load_ordinary_sft_bundle if bundle_id.startswith("plan-sft-ordinary-") else
                  load_sft_workflow_bundle)
        try:
            _, files = loader(workspace, bundle_id)
            profile = ProviderProfile.model_validate(yaml.safe_load(files["profile"].read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
            raise HTTPException(422, "登记方案中的 Provider 配置不可用") from exc
        blockers = []
        if profile.adapter != "openai_compatible" or not _credential_available(Path(workspace).resolve(), profile.api_key_env):
            blockers.append("登记方案的 Provider 不是已配置凭据的真实 Provider")
        if prepared.get("eligible_source_count") == 0:
            blockers.append("所选数据集没有可用于生成的训练种子")
        bundle_checks = dict(prepared)
        bundle_checks["status"] = "passed"
        return {"stage": stage, "mode": mode, "status": "blocked" if blockers else "preflight_ready",
                "read_only": True, "provider_calls": 0, "blockers": blockers,
                "bundle_checks": bundle_checks}

    @app.post("/api/synthetic/sft/start", status_code=202)
    def sft_generation_start(request: Request, payload: dict,
                             x_local_synthetic_token: str | None = Header(default=None)):
        jobs = require_synthetic_generation(request, x_local_synthetic_token)
        required = {"bundle_id", "usable_target", "max_requests", "target_candidates", "confirm_start"}
        allowed = required | {"stage_plan_id", "plan_version"}
        if (not required.issubset(payload) or set(payload) - allowed or not isinstance(payload.get("bundle_id"), str)
                or payload.get("confirm_start") is not True):
            raise HTTPException(422, "启动需明确 confirm_start=true，并仅提交登记计划及有限预算字段")
        if ("stage_plan_id" in payload) != ("plan_version" in payload):
            raise HTTPException(422, "共享 stage plan 必须同时提供 plan_id 与 version")
        if ("stage_plan_id" in payload and (not isinstance(payload["stage_plan_id"], str)
                or not isinstance(payload["plan_version"], int) or isinstance(payload["plan_version"], bool))):
            raise HTTPException(422, "共享 stage plan 身份无效")
        if any(not isinstance(payload[key], int) or isinstance(payload[key], bool) or payload[key] < 1
               for key in ("usable_target", "max_requests", "target_candidates")):
            raise HTTPException(422, "目标数与预算必须是正整数")
        try:
            return jobs.start(payload["bundle_id"], usable_target=payload["usable_target"],
                              max_requests=payload["max_requests"],
                              target_candidates=payload["target_candidates"],
                              stage_plan_id=payload.get("stage_plan_id"), plan_version=payload.get("plan_version"))
        except (RuntimeError, UncertainProviderRequest) as exc:
            raise HTTPException(409, str(exc)) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(422, "生成任务未启动；请检查预检结果与已登记输入") from exc

    @app.post("/api/synthetic/stage/start", status_code=202)
    def synthetic_stage_start(request: Request, payload: dict,
                              x_local_synthetic_token: str | None = Header(default=None)):
        """Dispatch only explicitly confirmed, currently supported stage requests."""
        require_synthetic_generation(request, x_local_synthetic_token)
        if payload.get("confirm_start") is not True:
            raise HTTPException(422, "启动模型生成须明确确认")
        stage_payload = {key: value for key, value in payload.items() if key != "confirm_start"}
        result = synthetic_stage_preflight(request, stage_payload, x_local_synthetic_token)
        if result["status"] != "preflight_ready" or stage_payload["stage"] != "sft":
            raise HTTPException(409, "阶段生成尚不可启动，请检查预检阻断项")
        legacy_payload = {key: value for key, value in payload.items() if key not in {"stage", "mode"}}
        return sft_generation_start(request, legacy_payload, x_local_synthetic_token)

    @app.get("/api/synthetic/sft/runs/{execution_id}")
    def sft_generation_status(request: Request, execution_id: str,
                              x_local_synthetic_token: str | None = Header(default=None)):
        jobs = require_synthetic_generation(request, x_local_synthetic_token)
        try:
            state = jobs.get(execution_id)
        except ValueError as exc:
            raise HTTPException(404, "Unknown generation execution") from exc
        if state is None:
            raise HTTPException(404, "Unknown generation execution")
        return state

    @app.get("/api/synthetic/sft/runs/{execution_id}/preview")
    def sft_generation_preview(request: Request, execution_id: str, limit: int = Query(default=3, ge=1, le=5),
                               x_local_synthetic_token: str | None = Header(default=None)):
        jobs = require_synthetic_generation(request, x_local_synthetic_token)
        try:
            return jobs.preview(execution_id, limit=limit)
        except KeyError as exc:
            raise HTTPException(404, "Unknown generation execution") from exc
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(409, "生成输出暂不可预览；请检查任务状态与 schema 验证结果") from exc

    @app.post("/api/synthetic/sft/runs/{execution_id}/cancel")
    def sft_generation_cancel(request: Request, execution_id: str, payload: dict,
                              x_local_synthetic_token: str | None = Header(default=None)):
        jobs = require_synthetic_generation(request, x_local_synthetic_token)
        if payload:
            raise HTTPException(422, "取消请求不接受参数")
        try:
            return jobs.cancel(execution_id)
        except KeyError as exc:
            raise HTTPException(404, "Unknown generation execution") from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409, "任务当前不可取消") from exc

    @app.get("/api/synthetic/stage/runs/{execution_id}")
    def synthetic_stage_status(request: Request, execution_id: str,
                               x_local_synthetic_token: str | None = Header(default=None)):
        state = sft_generation_status(request, execution_id, x_local_synthetic_token)
        mode = state.get("source_mode")
        if mode is None:
            mode = {"prompt_only_sft": "ordinary", "ordinary_sft": "ordinary",
                    "targeted": "targeted"}.get(state.get("mode"))
        return {**state, "stage": state.get("stage", "sft"), "source_mode": mode}

    @app.get("/api/synthetic/status/{stage}/{execution_id}")
    def synthetic_generation_snapshot(request: Request, stage: str, execution_id: str,
                                      x_local_synthetic_token: str | None = Header(default=None)):
        """Stable read-only status view; it never starts or resumes Provider work."""
        require_synthetic_generation(request, x_local_synthetic_token)
        jobs_by_stage = {
            "sft": app.state.sft_generation_jobs,
            "cpt": app.state.cpt_generation_jobs,
            "preference": app.state.preference_generation_jobs,
            "grpo": app.state.grpo_generation_jobs,
        }
        if stage not in jobs_by_stage or jobs_by_stage[stage] is None:
            raise HTTPException(404, "Unknown generation stage")
        jobs = jobs_by_stage[stage]
        try:
            state = jobs.get(execution_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(404, "Unknown generation execution") from exc
        if not isinstance(state, dict):
            raise HTTPException(404, "Unknown generation execution")
        result = state.get("result") if isinstance(state.get("result"), dict) else {}
        registration = state.get("registration")
        if not isinstance(registration, dict):
            registration = {}
        source_mode = state.get("source_mode") or state.get("mode")
        progress = state.get("progress") if isinstance(state.get("progress"), dict) else {}
        error = state.get("error")
        errors = state.get("errors") if isinstance(state.get("errors"), dict) else {}
        dataset_id = registration.get("dataset_id") or result.get("training_dataset_id")
        content_sha256 = registration.get("canonical_sha256") or result.get("canonical_sha256")
        missing_evidence = []
        if not registration.get("dataset_id") and stage != "grpo":
            missing_evidence.append("registered_dataset")
        if registration.get("dataset_id") and not registration.get("canonical_sha256"):
            missing_evidence.append("registered_dataset_hash")
        if stage == "grpo" and not registration.get("dataset_id"):
            # Prompt candidates intentionally stop before dataset registration; a
            # separately verified reference/reward-evidence workflow is required.
            missing_evidence.extend(["trusted_reference", "registered_trainable_dataset"])
        return {
            "schema_version": "generation-status-v1", "read_only": True,
            "execution_id": execution_id, "bundle_id": state.get("bundle_id"),
            "stage": stage, "source_mode": source_mode,
            "status": state.get("status", "unknown"),
            "created_at": state.get("created_at"), "updated_at": state.get("updated_at"),
            "progress": progress,
            "provider_requests": {"completed": result.get("request_count", state.get("request_count")),
                                  "budget": result.get("request_budget", state.get("request_budget")),
                                  "errors": errors or result.get("request_errors", {})},
            "inputs": {key: state.get(key, result.get(key)) for key in
                       ("source_dataset_id", "source_dataset_sha256", "targeted_plan_id",
                        "targeted_plan_version", "provider_profile_id")
                       if state.get(key, result.get(key)) is not None},
            "outputs": {"dataset_id": dataset_id, "canonical_sha256": content_sha256,
                        "registration": registration or None,
                        "artifacts": result.get("generation_artifacts")},
            "error": str(error)[:500] if error else None,
            "missing_evidence": missing_evidence,
        }

    @app.post("/api/synthetic/stage/runs/{execution_id}/cancel")
    def synthetic_stage_cancel(request: Request, execution_id: str, payload: dict,
                               x_local_synthetic_token: str | None = Header(default=None)):
        return sft_generation_cancel(request, execution_id, payload, x_local_synthetic_token)

    @app.post("/api/synthetic/sft/runs/{execution_id}/register")
    def sft_generation_register(request: Request, execution_id: str, payload: dict,
                                x_local_synthetic_token: str | None = Header(default=None)):
        jobs = require_synthetic_generation(request, x_local_synthetic_token)
        if payload != {"confirm_reviewed": True}:
            raise HTTPException(422, "登记前需确认已审核生成数据：confirm_reviewed=true")
        try:
            return jobs.register(execution_id)
        except KeyError as exc:
            raise HTTPException(404, "Unknown generation execution") from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (OSError, TypeError, ValueError, FileExistsError) as exc:
            raise HTTPException(422, "生成数据未能登记；数据为空、格式无效或证据不完整") from exc

    def require_editor(request: Request, token: str | None) -> None:
        if not enable_policy_editor:
            raise HTTPException(403, "本地策略编辑未启用；请使用 --enable-policy-editor 启动")
        origin = request.headers.get("origin")
        if origin:
            expected = f"{request.url.scheme}://{request.url.netloc}"
            if origin != expected:
                raise HTTPException(403, "只接受同源编辑请求")
        if token != app.state.policy_editor_token:
            raise HTTPException(403, "本地编辑令牌无效")

    @app.post("/api/policies/preview")
    def preview_policy(request: Request, payload: dict, x_local_editor_token: str | None = Header(default=None)):
        require_editor(request, x_local_editor_token)
        source_id = payload.get("source_id")
        config_path = source_configs().get(source_id)
        if config_path is None:
            raise HTTPException(422, "请选择已发现的本地来源配置")
        if not app.state.policy_preview_lock.acquire(blocking=False):
            raise HTTPException(409, "已有预览正在执行；请稍后重试")
        try:
            raw_policy = payload.get("policy")
            if not isinstance(raw_policy, dict):
                raise HTTPException(422, "缺少策略配置")
            raw_policy = dict(raw_policy)
            raw_policy.pop("tokenizer", None)
            source = load_source(config_path)
            policy = DataPolicy.model_validate({**raw_policy, "tokenizer": source.tokenizer.model_dump(mode="json")})
            count = payload.get("rows", 200)
            if type(count) is not int or not 1 <= count <= 1000:
                raise HTTPException(422, "预览行数必须为 1..1000")
            _, report = execute_policy(policy, source, preview_rows=count, seconds=30)
            return {"policy_id": policy.policy_id, "source_id": source_id,
                    "source_pool_id": report["source_pool"]["pool_id"],
                    "split_snapshot_id": report["split_snapshot"]["snapshot_id"],
                    "report": report}
        except HTTPException:
            raise
        except InputUnavailable as exc:
            raise HTTPException(409, str(exc)) from exc
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            app.state.policy_preview_lock.release()

    @app.post("/api/policies/save")
    def save_policies(request: Request, payload: dict, x_local_editor_token: str | None = Header(default=None)):
        require_editor(request, x_local_editor_token)
        try:
            policies = payload.get("policies")
            if not isinstance(policies, list) or len(policies) != 2:
                raise ValueError("必须保存 A、B 两份策略")
            source_path = source_configs().get(payload.get("source_id"))
            if source_path is None:
                raise ValueError("必须使用已发现的操作员来源配置")
            source = load_source(source_path)
            models = [DataPolicy.model_validate({**item, "tokenizer": source.tokenizer.model_dump(mode="json")})
                      for item in policies]
            if models[0].stage != models[1].stage:
                raise ValueError("A/B 阶段必须一致")
            identity = "draft-" + secrets.token_hex(8)
            output = workspace / "artifacts" / "data-experiments" / identity
            parent = workspace
            for part in ("artifacts", "data-experiments"):
                parent = parent / part
                if parent.is_symlink():
                    raise ValueError("策略草稿目录不能经过符号链接")
                parent.mkdir(exist_ok=True)
                if not parent.resolve().is_relative_to(workspace):
                    raise ValueError("策略草稿目录越过工作区边界")
            output.mkdir(parents=True, exist_ok=False)
            for label, policy in zip(("a", "b"), models):
                (output / f"policy-{label}.yaml").write_text(
                    yaml.safe_dump(policy.artifact(), allow_unicode=True, sort_keys=False), encoding="utf-8")
            (output / "README.txt").write_text(
                "策略草稿。此文件夹不表示已完成数据构建、训练或评测。\n"
                "请用 preview-data-policy/build-policy-data 与 register-data-experiment CLI 继续。\n",
                encoding="utf-8")
            return {"draft_id": identity, "path": output.relative_to(workspace).as_posix(),
                    "policy_ids": [policy.policy_id for policy in models], "status": "saved_policy_draft"}
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/policies/reports")
    def policy_reports():
        root = workspace / "artifacts" / "data-experiments"
        if root.is_symlink() or not root.is_dir() or not root.resolve().is_relative_to(workspace):
            return {"reports": []}
        reports = []
        visited = 0
        total_bytes = 0
        for current, dirs, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            if not current_path.resolve().is_relative_to(root.resolve()):
                dirs[:] = []
                continue
            dirs[:] = [name for name in dirs if not (current_path / name).is_symlink()
                       and (current_path / name).resolve().is_relative_to(root.resolve())]
            if len(current_path.relative_to(root).parts) > 4:
                dirs[:] = []
                continue
            for name in files:
                if not name.endswith(".json"):
                    continue
                visited += 1
                if visited > 2000:
                    break
                path = current_path / name
                try:
                    size = path.stat().st_size
                    if path.is_symlink() or size > 2 * 1024 * 1024 or total_bytes + size > 10 * 1024 * 1024:
                        continue
                    total_bytes += size
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(value, dict) or "comparison_status" not in value or "model_metrics" not in value:
                        continue
                    reports.append({key: value.get(key) for key in (
                        "report_id", "created_at", "experiment_id", "experiment_type", "evidence_kind",
                        "comparison_status", "compatibility_issues", "policy_changes", "data",
                        "model_metrics", "limitations")})
                except (OSError, ValueError, UnicodeError):
                    continue
        return {"reports": sorted(reports, key=lambda item: item.get("created_at") or "", reverse=True)[:100]}

    @app.post("/api/policies/save-experiment")
    def save_experiment(request: Request, payload: dict, x_local_editor_token: str | None = Header(default=None)):
        require_editor(request, x_local_editor_token)
        try:
            draft_id = payload.get("draft_id")
            if not isinstance(draft_id, str) or not re.fullmatch(r"draft-[a-f0-9]{16}", draft_id):
                raise ValueError("策略草稿身份无效")
            source_id = payload.get("source_id")
            if source_id not in source_configs():
                raise ValueError("来源配置不在本地登记列表")
            output = workspace / "artifacts" / "data-experiments" / draft_id
            if output.is_symlink() or not output.resolve().is_relative_to(workspace) or not output.is_dir():
                raise ValueError("找不到安全的策略草稿目录")
            models = [parse_policy(yaml.safe_load((output / f"policy-{label}.yaml").read_text(encoding="utf-8")))
                      for label in ("a", "b")]
            source = load_source(source_configs()[source_id])
            if any((policy.tokenizer.repo, policy.tokenizer.revision, policy.tokenizer.test_only)
                   != (source.tokenizer.repo, source.tokenizer.revision, source.tokenizer.test_only)
                   for policy in models):
                raise ValueError("实验来源与已保存策略的 tokenizer 身份不一致")
            stage = models[0].stage
            base = Path("artifacts") / "data-experiments" / draft_id
            fields = payload.get("experiment")
            if not isinstance(fields, dict) or any(key in fields for key in ("a", "b", "root", "output", "manifest")):
                raise ValueError("只提交实验描述字段；数据路径由服务端按策略草稿生成")
            value = dict(fields)
            budget = value.pop("budget", None)
            if not isinstance(budget, dict):
                raise TypeError("需要填写共同训练预算单位与值")
            arm_manifest = f"{base.as_posix()}/data-{{label}}/manifests/{stage}-policy-v1.yaml"
            value["a"] = ExperimentArm(policy=models[0], manifest=arm_manifest.format(label="a"), budget=Budget.model_validate(budget)).model_dump(mode="json")
            value["b"] = ExperimentArm(policy=models[1], manifest=arm_manifest.format(label="b"), budget=Budget.model_validate(budget)).model_dump(mode="json")
            plan = DataEffectExperiment.model_validate(value)
            plan_path = output / "experiment.yaml"
            with plan_path.open("x", encoding="utf-8") as stream:
                yaml.safe_dump(plan.model_dump(mode="json"), stream, allow_unicode=True, sort_keys=False)
            return {"plan_path": plan_path.relative_to(workspace).as_posix(), "experiment_id": plan.experiment_id,
                    "config_sha256": plan.config_sha256, "status": "draft_not_registered",
                    "registration_missing": ["A/B 数据构建 manifests", "训练前登记", "训练与评测结果"]}
        except FileExistsError as exc:
            raise HTTPException(409, "该草稿已有实验计划；为保留历史，请新建草稿") from exc
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/data")
    def builds():
        return catalog.scan()

    @app.get("/api/data/capabilities")
    def data_capabilities():
        return {"capabilities": [item.model_dump(mode="json") for item in source_stage_capabilities()]}

    @app.post("/api/data/mixtures/preflight")
    def dataset_mixture_preflight(payload: MixtureInput):
        try:
            return preflight_mixture(workspace, payload)
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)[:500]) from exc

    @app.post("/api/data/mixtures/build", status_code=201)
    async def dataset_mixture_build(payload: MixtureInput):
        if allow_remote:
            raise HTTPException(403, "数据集组合构建仅允许本机 loopback 服务")
        try:
            result, summary = await run_in_threadpool(build_mixture, workspace, payload)
        except FileExistsError as exc:
            raise HTTPException(409, "组合 dataset ID 已存在但证据不一致；既有产物未修改") from exc
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)[:500]) from exc
        catalog.scan()
        return summary | {"dataset_id": result.dataset_id, "reused": result.reused,
                          "manifest_path": (result.path / "manifests" / f"{payload.stage}-dataset-mixture-v1.yaml").relative_to(workspace).as_posix()}

    @app.get("/api/data/grpo-seeds")
    def grpo_seed_candidates():
        try:
            return {"protocol": "grpo-openr1-seed-v1", "candidates": list_grpo_seed_candidates(workspace)}
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)[:500]) from exc

    @app.post("/api/data/grpo-seeds/preflight")
    def grpo_seed_preflight(payload: GRPOSeedRequest):
        try:
            return preflight_grpo_seed(workspace, payload)
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)[:500]) from exc

    @app.post("/api/data/grpo-seeds/build", status_code=201)
    async def grpo_seed_build(payload: GRPOSeedRequest):
        if allow_remote:
            raise HTTPException(403, "GRPO seed 派生仅允许本机 loopback 服务")
        try:
            result = await run_in_threadpool(build_grpo_seed_dataset, workspace, payload)
        except FileExistsError as exc:
            raise HTTPException(409, "GRPO dataset ID 已存在但证据不一致；既有产物未修改") from exc
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)[:500]) from exc
        catalog.scan()
        return result

    @app.get("/api/data/datasets/{dataset_id}/preview")
    def dataset_sample_preview(dataset_id: str, limit: int = Query(default=3, ge=1, le=10)):
        try:
            return preview_dataset(workspace, dataset_id, limit=limit)
        except KeyError as exc:
            raise HTTPException(404, "Unknown registered dataset ID") from exc
        except PreviewUnavailable as exc:
            raise HTTPException(409, detail={"code": exc.code, "message": str(exc)}) from exc
        except ValueError as exc:
            raise HTTPException(422, "预览参数无效") from exc

    @app.get("/api/data/datasets/{dataset_id}/status")
    def dataset_status_snapshot(dataset_id: str):
        """Stable evidence-only view keyed by registered dataset ID."""
        try:
            return dataset_status(workspace, dataset_id)
        except KeyError as exc:
            raise HTTPException(404, "Unknown registered dataset ID") from exc
        except PreviewUnavailable as exc:
            raise HTTPException(409, detail={"code": exc.code, "message": str(exc)}) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(422, "数据集证据状态不可读取") from exc

    @app.post("/api/data/import/preview")
    async def preview_dataset_import(
        request: Request,
        stage: str = Query(..., min_length=1, max_length=20),
        source_format: str = Query(default="canonical-v1", max_length=80),
    ):
        if stage == "eval" and source_format == OBJECTIVE_EVALUATION_FORMAT:
            pass
        elif stage != "sft":
            raise HTTPException(422, "客观题评测集请选择 eval 阶段及 objective-choice-jsonl-v1 格式")
        elif source_format not in SFT_SOURCE_FORMATS:
            raise HTTPException(422, "请选择受支持的 SFT 来源格式")
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type not in {"application/x-ndjson", "application/jsonl", "application/octet-stream"}:
            raise HTTPException(415, "请以 JSONL 原始字节提交文件")
        try:
            declared = int(request.headers.get("content-length", "0"))
        except ValueError as exc:
            raise HTTPException(400, "无效 Content-Length") from exc
        if declared > MAX_IMPORT_BYTES:
            raise HTTPException(413, f"上传超过 {MAX_IMPORT_BYTES} 字节上限")
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > MAX_IMPORT_BYTES:
                raise HTTPException(413, f"上传超过 {MAX_IMPORT_BYTES} 字节上限")
        if stage == "eval":
            converted = await run_in_threadpool(
                convert_objective_evaluation_jsonl, BytesIO(bytes(raw)), preview_limit=MAX_PREVIEW_RECORDS,
            )
        else:
            converted = await run_in_threadpool(
                convert_sft_jsonl, BytesIO(bytes(raw)), source_format,
                preview_limit=MAX_PREVIEW_RECORDS,
            )
        report = converted.report()
        report["samples"] = list(converted.records[:MAX_PREVIEW_RECORDS])
        if stage != "eval":
            report["mapping"] = {"human": "user", "gpt": "assistant"} if source_format == "sharegpt-conversations-v1" else {}
        report["disclaimer"] = ("客观题答案仅按结构和选项引用校验，不验证答案事实正确性、来源许可或训练/评测污染；坏行会被排除。预览不会登记数据集或创建 Run。"
                                if stage == "eval" else
                                "结构转换预览不验证答案正确性、质量或许可；预览不会登记数据集或创建 Run。")
        report["read_only"] = True
        report["provider_calls"] = 0
        return report

    @app.post("/api/data/import", status_code=201)
    async def import_dataset(
        request: Request,
        stage: str = Query(..., min_length=1, max_length=20),
        source_name: str = Query(default="本地上传", max_length=200),
        source_uri: str | None = Query(default=None, max_length=2048),
        revision: str | None = Query(default=None, max_length=300),
        revision_status: str = Query(default="not_provided", max_length=30),
        license_id: str | None = Query(default=None, max_length=300),
        license_status: str = Query(default="not_provided", max_length=40),
        attribution: str | None = Query(default=None, max_length=2000),
        source_format: str = Query(default="canonical-v1", max_length=80),
    ):
        if stage not in {"cpt", "sft", "preference", "grpo", "eval"}:
            raise HTTPException(422, "stage 必须是 cpt、sft、preference、grpo 或 eval")
        if license_status == "verified":
            raise HTTPException(422, "本地上传的许可证只能按用户声明或未知登记，不能自助标记为已独立核验")
        if stage == "eval" and source_format != OBJECTIVE_EVALUATION_FORMAT:
            raise HTTPException(422, "客观题评测集必须使用 objective-choice-jsonl-v1 格式")
        if stage != "eval" and source_format not in SFT_SOURCE_FORMATS:
            raise HTTPException(422, "请选择受支持的 SFT 来源格式")
        if stage not in {"sft", "eval"} and source_format != "canonical-v1":
            raise HTTPException(422, "来源格式适配器目前仅用于 SFT 导入")
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() not in {
            "application/x-ndjson", "application/jsonl", "application/octet-stream",
        }:
            raise HTTPException(415, "请以 JSONL 原始字节提交文件")
        try:
            content_length = int(request.headers.get("content-length", "0"))
        except ValueError as exc:
            raise HTTPException(400, "无效 Content-Length") from exc
        if content_length > MAX_IMPORT_BYTES:
            raise HTTPException(413, f"上传超过 {MAX_IMPORT_BYTES} 字节上限")
        try:
            source = DataSource(
                source_id="upload-pending", kind="local_upload", name=source_name, uri=source_uri,
                revision=revision, revision_status=revision_status,
                license_id=license_id, license_status=license_status, attribution=attribution,
                declared_by="user",
            )
        except ValidationError as exc:
            raise HTTPException(422, "来源或许可状态字段无效") from exc

        # Keep even the transient upload bytes inside the configured workspace;
        # do not use the client filename as a path or echo user records in errors.
        try:
            with tempfile.TemporaryFile(mode="w+b", dir=workspace) as upload:
                size = 0
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_IMPORT_BYTES:
                        raise HTTPException(413, f"上传超过 {MAX_IMPORT_BYTES} 字节上限")
                    upload.write(chunk)
                upload.seek(0)
                raw_source = upload.read() if source_format != "canonical-v1" else None
                if source_format != "canonical-v1":
                    converted = await run_in_threadpool(
                        convert_objective_evaluation_jsonl if stage == "eval" else convert_sft_jsonl,
                        BytesIO(raw_source), *(() if stage == "eval" else (source_format,)),
                    )
                    if not converted.valid:
                        raise HTTPException(422, detail={
                            "code": "source_format_conversion_failed", "source_format": source_format,
                            "input_record_count": converted.source_record_count,
                            "output_record_count": converted.output_record_count,
                            "rejected_record_count": converted.source_record_count - converted.output_record_count,
                            "errors_truncated": converted.errors_truncated,
                            "errors": [issue.safe_dict() for issue in converted.errors],
                        })
                    if len(raw_source) + len(converted.canonical_bytes) > 512 * 1024 * 1024:
                        raise HTTPException(413, "原始文件与转换结果总大小超过 512 MiB 上限")
                    upload = BytesIO(converted.canonical_bytes)
                    converter_config = ({"source_format": OBJECTIVE_EVALUATION_FORMAT,
                                         "converter_version": converted.report()["converter_version"],
                                         "mapping": converted.report()["mapping"],
                                         "source_record_count": converted.source_record_count,
                                         "output_record_count": converted.output_record_count,
                                         "rejected_record_count": converted.source_record_count - converted.output_record_count,
                                         "rejections": [issue.safe_dict() for issue in converted.errors],
                                         "errors_truncated": converted.errors_truncated}
                                        if stage == "eval" else
                                        {**converted.config, "source_record_count": converted.source_record_count,
                                         "output_record_count": converted.output_record_count})
                else:
                    converter_config = None
                result = await run_in_threadpool(
                    register_jsonl, upload, workspace=workspace,
                    output_root=workspace / "data" / "processed" / "user-upload",
                    stage=stage, origin=source, raw_source_bytes=raw_source,
                    source_format=source_format if source_format != "canonical-v1" else None,
                    converter_config=converter_config,
                    registration_metadata={"dataset_role": "evaluation_only",
                                           "evaluation_protocol": "choice-accuracy-v1"}
                    if stage == "eval" else None,
                )
        except HTTPException:
            raise
        except FileExistsError as exc:
            raise HTTPException(409, "dataset ID 已存在但内容证据不匹配；既有产物未修改") from exc
        except ValueError as exc:
            try:
                report = json.loads(str(exc))
            except (ValueError, TypeError):
                report = None
            if isinstance(report, dict) and "errors" in report:
                raise HTTPException(422, detail=report) from exc
            raise HTTPException(422, "数据集登记失败；请检查阶段 schema 和来源元数据") from exc
        except OSError as exc:
            raise HTTPException(507, "本地数据集无法写入工作区；没有登记半成品") from exc

        catalog.scan()
        manifest_path = result.path / "manifests" / f"{stage}-user-upload-v1.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        quality_status = manifest.get("metadata", {}).get("quality_status", "NOT_EVALUATED")
        return {
            "dataset_id": result.dataset_id, "stage": result.stage,
            "record_count": result.record_count, "original_sha256": result.original_sha256,
            "canonical_sha256": result.canonical_sha256,
            "manifest_path": manifest_path.relative_to(workspace).as_posix(),
            "quality_status": quality_status, "reused": result.reused,
            "validation": result.validation.report(),
        }

    @app.get("/api/evaluation-matrices")
    def evaluation_matrices():
        return matrix_index(workspace)

    @app.get("/api/evaluation-matrices/{matrix_id}")
    def evaluation_matrix(matrix_id: str):
        if not re.fullmatch(r"[a-f0-9]{20}", matrix_id):
            raise HTTPException(404, "Unknown evaluation matrix")
        result = matrix_detail(workspace, matrix_id)
        if result is None:
            raise HTTPException(404, "Evaluation matrix is unavailable or invalid")
        return result

    @app.get("/api/data/{build_id}")
    def build(build_id: str):
        try:
            return catalog.detail(build_id)
        except KeyError as exc:
            raise HTTPException(404, "Unknown build; refresh the artifact index") from exc

    @app.get("/api/artifacts/{artifact_id}")
    def artifact(artifact_id: str):
        try:
            return catalog.artifact(artifact_id)
        except KeyError as exc:
            raise HTTPException(404, "Only artifacts discovered in the report index are exposed") from exc
        except (OSError, ArtifactError, RecursionError) as exc:
            raise HTTPException(422, "Artifact is unavailable, unsafe or exceeds the read limit") from exc

    static_dir = static_dir or workspace / "web/dist"
    @app.get("/assets/{filename}")
    def asset(filename: str):
        root = static_dir / "assets"
        path = root / filename
        if not re.fullmatch(r"[A-Za-z0-9_-]+\.(js|css|svg|woff2)", filename) or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
            raise HTTPException(404, "Unknown UI asset")
        return FileResponse(path)

    @app.get("/")
    def index():
        path = static_dir / "index.html"
        if path.is_symlink() or not path.resolve().is_relative_to(static_dir.resolve()):
            raise HTTPException(404, "Invalid UI index")
        if not path.is_file():
            raise HTTPException(503, "Frontend has not been built. Run pnpm install && pnpm build in web/.")
        return FileResponse(path, media_type="text/html")

    return app


def serve(
    workspace: Path,
    port: int = 8765,
    *,
    host: str = "127.0.0.1",
    allow_remote: bool = False,
    enable_policy_editor: bool = False,
    enable_experiment_launcher: bool = False,
    enable_synthetic_generation: bool = False,
) -> None:
    try:
        is_loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = host.lower() == "localhost"
    if not is_loopback and not allow_remote:
        raise SystemExit(
            "Refusing non-loopback UI bind without --allow-remote. "
            "Remote clients may access local UI data; no firewall rules will be changed."
        )
    if (enable_experiment_launcher or enable_synthetic_generation) and (allow_remote or not is_loopback):
        raise SystemExit("Refusing local training/generation with remote access; bind to loopback without --allow-remote")
    import uvicorn

    if allow_remote:
        print(
            "WARNING: remote UI access is enabled. Reachable clients can access local UI data "
            "and any enabled policy-editing endpoints. No firewall rules are changed."
        )
    print(f"Serving OneGPU PostTrain UI on http://{host}:{port}")
    uvicorn.run(
        create_app(workspace.resolve(), enable_policy_editor=enable_policy_editor,
                   enable_experiment_launcher=enable_experiment_launcher,
                   enable_synthetic_generation=enable_synthetic_generation, allow_remote=allow_remote),
        host=host,
        port=port,
    )
