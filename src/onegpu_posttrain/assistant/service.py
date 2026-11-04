"""Framework-neutral orchestration for workspace evidence Q&A."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from uuid import uuid4

from .comparison import EvaluationComparison, compare_evaluations
from .context import rebuild_context_summary
from .budget import (
    ContextBudgetExceeded,
    MAX_EVIDENCE_ITEMS,
    MAX_RECENT_TURNS,
    enforce_request_budget,
)
from .contracts import (
    AgentTaskRecord,
    AssistantAnswer,
    EvidenceQuery,
    EvidenceRef,
    ModelRequest,
    ObservationEvent,
    SessionMessage,
)
from .evidence_reader import WorkspaceEvidenceReader
from .model_adapter import AssistantModelError
from .ports import ModelAdapter
from .profiles import AssistantProfileCatalog
from .ports import Observer
from .store import WorkspaceAssistantStore

_ENTITY_ID = re.compile(
    r"(?<![A-Za-z0-9])(?:run|eval|model|ds|dataset)-[A-Za-z0-9][A-Za-z0-9._-]{0,158}"
    r"[A-Za-z0-9](?![A-Za-z0-9._-])", re.IGNORECASE,
)
_INVALID_ANSWER = AssistantAnswer(
    conclusion="当前证据不足以安全回答。",
    limitations=["无法确认根因；未找到可用于支撑回答的工作区证据。"],
    clarification_options=["请提供运行 ID、模型 ID 或数据集 ID。"],
)


def classify_question(question: str) -> str:
    text = question.casefold()
    explicit_comparison = ("对比", "比较", "相比", "差异", "变化", "上一次", "最近一次", "提升", "下降", "哪个更好")
    if any(word in text for word in explicit_comparison) and any(
        word in text for word in ("评测", "准确率", "结果", "模型")
    ):
        return "compare_evaluations"
    # A failed evaluation is a diagnosis request unless the user explicitly
    # asks to compare it. Do not route “评测为什么失败” into comparison mode.
    if any(word in text for word in ("失败", "报错", "异常", "没跑完")):
        return "recent_failed_runs"
    if "评测" in text and any(word in text for word in ("分数", "完整", "结果", "状态", "通过")):
        return "compare_evaluations"
    if any(word in text for word in ("血缘", "来源", "来自", "训练数据", "用了哪些数据", "数据集")):
        return "trace_data_lineage"
    return "general"


_EXECUTION_REQUEST = re.compile(
    r"(?:帮我|请|直接|替我)\s*(?:启动|运行|执行|修改|编辑|生成|创建|删除|安装|下载|联网搜索|调用|训练|微调|评测)"
    r"|(?:运行|执行|调用|使用)\s*(?:shell|python|mcp|terminal|subprocess|命令|脚本|工具)"
    r"|(?:启动|开始)\s*(?:训练|微调|评测|数据生成)",
    re.IGNORECASE,
)


class AssistantTurnResult:
    def __init__(self, *, task: AgentTaskRecord, answer: AssistantAnswer | None,
                 evidence: list[EvidenceRef], comparisons: list[EvaluationComparison],
                 consent_required: bool = False):
        self.task = task
        self.answer = answer
        self.evidence = evidence
        self.comparisons = comparisons
        self.consent_required = consent_required


class WorkspaceAssistantService:
    def __init__(
        self,
        *,
        store: WorkspaceAssistantStore,
        evidence_reader: WorkspaceEvidenceReader,
        profiles: AssistantProfileCatalog,
        model: ModelAdapter,
        observer: Observer | None = None,
        agent_runtime=None,
    ):
        self.store = store
        self.evidence_reader = evidence_reader
        self.profiles = profiles
        self.model = model
        self.observer = observer
        self.agent_runtime = agent_runtime

    def _emit_task_trace(self, task: AgentTaskRecord) -> AgentTaskRecord:
        if self.observer is None:
            return task
        summary = self.store.get_summary(task.session_id)
        event = ObservationEvent(
            event_name="assistant_task_finished",
            task_id=task.task_id,
            occurred_at=datetime.now(timezone.utc),
            attributes={
                "session_id": task.session_id,
                "assistant_profile_id": task.assistant_profile_id,
                "model_profile_id": task.model_profile_id,
                "status": task.status,
                "duration_ms": task.duration_ms,
                "input_tokens": task.input_tokens,
                "output_tokens": task.output_tokens,
                "runtime": task.metadata.get("runtime"),
                "tool_calls": task.metadata.get("tool_calls"),
                "model_calls": task.metadata.get("model_calls"),
                "provider_called": task.metadata.get("provider_called"),
                "error_code": task.error_code,
                "evidence_count": len(task.evidence),
                "evidence_ids": ",".join(item.evidence_id for item in task.evidence)[:2000],
                "evidence_versions": ",".join(item.version or "unknown" for item in task.evidence)[:2000],
                "source_entity_ids": ",".join(item.entity_id for item in task.evidence)[:2000],
                "memory_ids": ",".join(task.memory_ids),
                "context_summary_id": task.context_summary_id,
                "summary_revision": summary.revision if summary else None,
            },
        )
        try:
            trace_id, trace_url = self.observer.emit(event)
        except Exception:
            return task
        if not trace_id and not trace_url:
            return task
        task = task.model_copy(update={
            "langfuse_trace_id": trace_id,
            "langfuse_trace_url": trace_url,
        })
        try:
            self.store.update(task)
        except Exception:
            return task.model_copy(update={"langfuse_trace_id": None, "langfuse_trace_url": None})
        return task

    def confirm_data_egress(self, session_id: str):
        """Persist the explicit user action required before any Bailian request."""
        return self.store.grant_data_egress_consent(session_id, version="bailian-minimal-evidence-v1")

    def ask(
        self,
        question: str,
        *,
        session_id: str | None = None,
        assistant_profile_id: str = "workspace-analyst",
    ) -> AssistantTurnResult:
        question = question.strip()
        if not question or len(question) > 2000:
            raise ValueError("问题不能为空且不能超过 2000 个字符")
        profile = self.profiles.get(assistant_profile_id)
        if profile is None:
            raise ValueError("助手配置不存在")
        session = self.store.get_session(session_id) if session_id else None
        if session_id and session is None:
            raise KeyError(session_id)
        if session is None:
            session = self.store.create_session(question[:80], assistant_profile_id)
        elif session.assistant_profile_id != assistant_profile_id:
            raise ValueError("不能在已有会话中切换助手配置")

        now = datetime.now(timezone.utc)
        pending_message = next((
            item for item in session.messages
            if item.message_id == session.pending_egress_question_message_id and item.role == "user"
        ), None) if session.pending_egress_question == question and session.data_egress_consent_at else None
        if pending_message is not None:
            question_message = pending_message
            session = self.store.save_session(session.model_copy(update={
                "pending_egress_question": None,
                "pending_egress_question_message_id": None,
            }))
        else:
            question_message = SessionMessage(
                message_id=f"message-{uuid4().hex[:20]}", role="user", content=question, created_at=now
            )
            session = self.store.append_message(session.session_id, question_message)
        previous_summary = self.store.get_summary(session.session_id)
        compression_error = False
        try:
            context_summary = rebuild_context_summary(session, previous_summary)
            if context_summary is not None:
                context_summary = self.store.save_summary(context_summary)
        except Exception:
            context_summary = previous_summary
            compression_error = len(session.messages) > MAX_RECENT_TURNS
        active_memories = self.store.list_memories(include_disabled=False)[:20]
        task = AgentTaskRecord(
            task_id=f"task-{uuid4().hex[:24]}", session_id=session.session_id,
            assistant_profile_id=profile.profile_id,
            model_profile_id="dashscope-qwen3.6-plus-local",
            status="queued", created_at=now, updated_at=now,
            question_message_id=question_message.message_id,
        )
        if compression_error:
            task = task.model_copy(update={
                "status": "needs_clarification", "updated_at": datetime.now(timezone.utc),
                "clarification_required": True, "error_code": "context_compression_failed",
                "error_message": "较早对话摘要未能保存；原始消息已保留，本轮没有调用模型。",
            })
            self.store.create(task)
            answer = AssistantAnswer(
                conclusion="较早对话暂时无法安全压缩，因此本轮没有发送给模型。",
                limitations=["原始消息仍保存在会话中，压缩失败没有删除或覆盖对话。"],
                clarification_options=["请缩小当前问题范围，或稍后重试摘要。"],
            )
            self._append_answer(session.session_id, answer)
            return AssistantTurnResult(task=task, answer=answer, evidence=[], comparisons=[])
        if session.data_egress_consent_at is None:
            session = self.store.save_session(session.model_copy(update={
                "pending_egress_question": question,
                "pending_egress_question_message_id": question_message.message_id,
            }))
            task = task.model_copy(update={
                "status": "needs_clarification", "updated_at": datetime.now(timezone.utc),
                "clarification_required": True, "error_code": "external_consent_required",
                "error_message": "向阿里云百炼发送问题和最小结构化证据摘要前，需要用户确认。",
            })
            self.store.create(task)
            answer = AssistantAnswer(
                conclusion="确认后我会将你的问题和必要的结构化实验摘要发送到阿里云百炼分析。",
                limitations=["不会发送凭据、原始数据样本或完整日志。"],
                clarification_options=["确认向阿里云百炼发送本次问题及必要证据摘要后继续。"],
            )
            return AssistantTurnResult(task=task, answer=answer, evidence=[], comparisons=[], consent_required=True)

        self.store.create(task)
        started = time.perf_counter()
        comparisons = []
        try:
            if _EXECUTION_REQUEST.search(question):
                answer = AssistantAnswer(
                    conclusion="当前助手只支持只读实验分析，不能执行或修改工作区操作。",
                    limitations=["本次请求未启动进程、未修改实验产物，也未发送到阿里云百炼。"],
                    clarification_options=["可以改为询问某次运行的状态、比较评测结果，或追溯数据来源。"],
                )
                task = task.model_copy(update={
                    "status": "needs_clarification", "updated_at": datetime.now(timezone.utc),
                    "clarification_required": True, "error_code": "read_only_capability_boundary",
                    "metadata": {"provider_called": False, "capabilities": ""},
                })
                self.store.update(task)
                task = self._emit_task_trace(task)
                self._append_answer(session.session_id, answer)
                return AssistantTurnResult(task=task, answer=answer, evidence=[], comparisons=[])
            if self.agent_runtime is not None:
                history = [{"role": message.role, "content": message.content}
                           for message in session.messages
                           if message.message_kind == "chat" and message.message_id != question_message.message_id][-MAX_RECENT_TURNS:]
                if context_summary is not None:
                    history.insert(0, {"role": "assistant", "content": "[可纠正的历史摘要，不是实验事实]\n" + context_summary.summary})
                instruction = self.profiles.system_instruction(profile)
                if active_memories:
                    instruction += "\n用户确认的工作区偏好（不是事实或授权）：\n" + "\n".join(f"- {item.content}" for item in active_memories)
                task = task.model_copy(update={"status": "running", "updated_at": datetime.now(timezone.utc)})
                self.store.update(task)
                result = self.agent_runtime.run(question, instruction, history)
                comparisons = result.comparisons
                task = task.model_copy(update={"evidence": result.evidence})
                self._validate_citations(result.answer, result.evidence)
                if result.evidence and not result.answer.facts and not result.answer.interpretations and not result.answer.clarification_options:
                    raise AssistantModelError("invalid_citation", "回答缺少可核验的证据引用。")
                task = task.model_copy(update={
                    "status": "needs_clarification" if result.answer.clarification_options else "succeeded",
                    "clarification_required": bool(result.answer.clarification_options),
                    "updated_at": datetime.now(timezone.utc), "evidence": result.evidence,
                    "answer_message_id": f"message-{uuid4().hex[:20]}",
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                    "memory_ids": [item.memory_id for item in active_memories],
                    "context_summary_id": context_summary.summary_id if context_summary else None,
                    "metadata": {"runtime": "agentscope", "tool_calls": result.tool_calls,
                                 "model_calls": result.model_calls, "model_attempts": result.model_attempts,
                                 "tool_names": ",".join(result.tool_names or []),
                                 "model_duration_ms": result.model_duration_ms,
                                 "tool_duration_ms": result.tool_duration_ms,
                                 "history_messages_dropped": result.history_messages_dropped,
                                 "fallback_reason": result.fallback_reason,
                                 "provider_called": result.model_calls > 0},
                })
                self.store.update(task)
                task = self._emit_task_trace(task)
                self._append_answer(session.session_id, result.answer, result.evidence, message_id=task.answer_message_id)
                return AssistantTurnResult(task=task, answer=result.answer, evidence=result.evidence, comparisons=comparisons)
            kind = classify_question(question)
            selected = sorted(set(_ENTITY_ID.findall(question)))[:20]
            if not selected:
                prior_citations = next((
                    item.evidence_ids for item in reversed(session.messages[:-1])
                    if item.role == "assistant" and item.evidence_ids
                ), [])
                selected = sorted({
                    identifier.split(":", 1)[1]
                    for identifier in prior_citations
                    if ":" in identifier
                    and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,159}", identifier.split(":", 1)[1])
                })[:20]
                prefixes = {identifier.split(":", 1)[0] for identifier in prior_citations}
                if kind == "general":
                    if prefixes & {"dataset", "lineage", "model", "quality"}:
                        kind = "trace_data_lineage"
                    elif "evaluation" in prefixes:
                        kind = "compare_evaluations"
                    elif "run" in prefixes:
                        kind = "recent_failed_runs"
            recent_turns = [
                {"role": item.role, "content": item.content}
                for item in [
                    message for message in session.messages
                    if message.message_kind == "chat" and message.message_id != question_message.message_id
                ][-MAX_RECENT_TURNS:]
            ]
            if context_summary is not None:
                recent_turns.insert(0, {
                    "role": "assistant",
                    "content": "[可纠正的历史上下文摘要，不是实验事实]\n" + context_summary.summary,
                })
            query = EvidenceQuery(
                kind=kind,
                query=question[:2000],
                selected_entity_ids=selected,
                limit=10,
            )
            evidence = self.evidence_reader.search(query)
            if not evidence:
                task = task.model_copy(update={
                    "status": "needs_clarification", "updated_at": datetime.now(timezone.utc),
                    "clarification_required": True, "error_code": "evidence_not_found",
                    "error_message": "工作区没有找到匹配的实验记录。",
                })
                self.store.update(task)
                task = self._emit_task_trace(task)
                answer = AssistantAnswer(
                    conclusion="我没有在当前工作区找到能回答这个问题的记录。",
                    limitations=["没有基于记录之外的信息推断实验状态。"],
                    clarification_options=["请提供运行 ID、模型 ID、数据集 ID，或补充你要查找的训练阶段。"],
                )
                self._append_answer(session.session_id, answer)
                return AssistantTurnResult(task=task, answer=answer, evidence=[], comparisons=[])

            used_evidence = self._bound_evidence(evidence, kind)[:MAX_EVIDENCE_ITEMS]
            if kind == "compare_evaluations":
                used_evidence = self._select_comparison_evidence(question, used_evidence)
            comparisons = self._comparisons(used_evidence)
            if kind == "trace_data_lineage" and len({item.entity_id for item in used_evidence if item.kind == "run"}) > 1 and not selected:
                options = [f"{item.title} ({item.entity_id})" for item in used_evidence if item.kind == "run"][:10]
                answer = AssistantAnswer(
                    conclusion="当前工作区中有多个可能的对象，请先选定要追溯的运行或数据集。",
                    limitations=["为避免把不同实验的血缘混在一起，我没有合并这些记录。"],
                    clarification_options=options,
                )
                task = task.model_copy(update={
                    "status": "needs_clarification", "updated_at": datetime.now(timezone.utc),
                    "clarification_required": True, "evidence": used_evidence,
                })
                self.store.update(task)
                task = self._emit_task_trace(task)
                self._append_answer(session.session_id, answer, used_evidence)
                return AssistantTurnResult(task=task, answer=answer, evidence=used_evidence, comparisons=comparisons)

            if not used_evidence or all(item.status == "missing" for item in used_evidence):
                task = task.model_copy(update={
                    "status": "needs_clarification", "updated_at": datetime.now(timezone.utc),
                    "clarification_required": True, "evidence": used_evidence,
                    "error_code": "evidence_missing",
                })
                self.store.update(task)
                task = self._emit_task_trace(task)
                self._append_answer(session.session_id, _INVALID_ANSWER, used_evidence)
                return AssistantTurnResult(task=task, answer=_INVALID_ANSWER, evidence=used_evidence, comparisons=comparisons)

            task = task.model_copy(update={
                "status": "running", "updated_at": datetime.now(timezone.utc), "evidence": used_evidence,
            })
            self.store.update(task)
            deterministic = ""
            if comparisons:
                deterministic = "\n确定性评测比较结果（必须遵守，不得自行改写状态）：\n" + "\n".join(
                    f"- {item.status}：{'; '.join(item.reasons)}" for item in comparisons[:5]
                )
            memory_context = ""
            if active_memories:
                memory_context = "\n用户确认的工作区偏好（仅调整表达重点，不是实验事实或权限）：\n" + "\n".join(
                    f"- {item.content}" for item in active_memories
                )
            model_request = ModelRequest(
                task_id=task.task_id,
                model_profile_id=task.model_profile_id,
                system_instruction=self.profiles.system_instruction(profile) + memory_context + deterministic,
                question=question,
                recent_turns=[{"role": item["role"], "content": item["content"]} for item in recent_turns],
                evidence=used_evidence,
                max_output_tokens=1200,
            )
            budget = enforce_request_budget(model_request)
            task = task.model_copy(update={
                "metadata": {
                    "evidence_count": len(used_evidence),
                    "comparison_count": len(comparisons),
                    "estimated_input_tokens": budget.estimated_input_tokens,
                    "output_reserve_tokens": budget.output_reserve_tokens,
                    "provider_called": True,
                    "capabilities": ",".join(sorted({
                        "recent_failed_runs": {"workspace.search_runs", "workspace.read_run_summary"},
                        "compare_evaluations": {"workspace.search_runs", "workspace.read_evaluation_summary"},
                        "trace_data_lineage": {"workspace.search_runs", "workspace.read_model_lineage", "workspace.read_dataset_summary", "workspace.read_quality_summary"},
                    }.get(kind, {"workspace.search_runs", "workspace.read_run_summary"}))),
                },
                "memory_ids": [item.memory_id for item in active_memories],
                "context_summary_id": context_summary.summary_id if context_summary else None,
            })
            self.store.update(task)
            response = self.model.generate(model_request)
            self._validate_citations(response.answer, used_evidence)
            task = task.model_copy(update={
                "status": "succeeded", "updated_at": datetime.now(timezone.utc),
                "answer_message_id": f"message-{uuid4().hex[:20]}",
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
                "metadata": {
                    **task.metadata,
                    "input_tokens_reported": response.input_tokens,
                    "output_tokens_reported": response.output_tokens,
                },
            })
            self.store.update(task)
            task = self._emit_task_trace(task)
            self._append_answer(
                session.session_id, response.answer, used_evidence,
                message_id=task.answer_message_id,
            )
            return AssistantTurnResult(task=task, answer=response.answer, evidence=used_evidence, comparisons=comparisons)
        except ContextBudgetExceeded as exc:
            task = task.model_copy(update={
                "status": "needs_clarification", "updated_at": datetime.now(timezone.utc),
                "clarification_required": True, "error_code": "context_budget_exceeded",
                "error_message": str(exc), "metadata": {
                    "estimated_input_tokens": exc.estimate, "input_limit_tokens": exc.limit,
                    "provider_called": False,
                },
            })
            self.store.update(task)
            task = self._emit_task_trace(task)
            answer = AssistantAnswer(
                conclusion=str(exc),
                limitations=["为避免丢弃关键事实，本轮没有静默截断证据或历史，也没有调用模型。"],
                clarification_options=["指定一个运行、评测或数据集 ID，或在新会话中提问。"],
            )
            self._append_answer(session.session_id, answer, task.evidence)
            return AssistantTurnResult(task=task, answer=answer, evidence=task.evidence, comparisons=comparisons)
        except AssistantModelError as exc:
            error_evidence = getattr(exc, "evidence", task.evidence)
            error_model_calls = getattr(exc, "model_calls", None)
            error_tool_calls = getattr(exc, "tool_calls", None)
            task = task.model_copy(update={
                "status": "failed", "updated_at": datetime.now(timezone.utc),
                "error_code": exc.code, "error_message": str(exc)[:1000],
                "evidence": error_evidence,
                "metadata": {"runtime": "agentscope" if self.agent_runtime else "legacy",
                             "model_calls": error_model_calls, "tool_calls": error_tool_calls,
                             "model_attempts": getattr(exc, "model_attempts", None),
                             "tool_names": ",".join(getattr(exc, "tool_names", []) or []),
                             "model_duration_ms": getattr(exc, "model_duration_ms", None),
                             "tool_duration_ms": getattr(exc, "tool_duration_ms", None),
                             "history_messages_dropped": getattr(exc, "history_messages_dropped", None),
                             "provider_called": error_model_calls > 0 if error_model_calls is not None else None},
                "duration_ms": int((time.perf_counter() - started) * 1000),
            })
            self.store.update(task)
            task = self._emit_task_trace(task)
            answer = AssistantAnswer(
                conclusion=str(exc),
                limitations=["本次分析未完成，已读取的证据不代表最终结论。"],
                clarification_options=["检查提示后缩小查询范围或重试。"],
            )
            self._append_answer(session.session_id, answer, task.evidence)
            return AssistantTurnResult(task=task, answer=answer, evidence=task.evidence, comparisons=[])

    @staticmethod
    def _bound_evidence(evidence: list[EvidenceRef], kind: str) -> list[EvidenceRef]:
        if kind == "recent_failed_runs":
            allowed = {"run", "evaluation", "configuration", "log", "lineage"}
        elif kind == "compare_evaluations":
            allowed = {"evaluation"}
        elif kind == "trace_data_lineage":
            allowed = {"run", "lineage", "model", "dataset", "quality_report"}
        else:
            allowed = {"run", "evaluation", "dataset", "quality_report", "lineage"}
        return [item for item in evidence if item.kind in allowed][:50]

    @staticmethod
    def _comparisons(evidence: list[EvidenceRef]) -> list[EvaluationComparison]:
        evaluations = [item for item in evidence if item.kind == "evaluation"]
        results = []
        for index, left in enumerate(evaluations):
            for right in evaluations[index + 1:]:
                results.append(compare_evaluations(left, right))
                if len(results) >= 5:
                    return results
        return results

    @staticmethod
    def _select_comparison_evidence(question: str, evidence: list[EvidenceRef]) -> list[EvidenceRef]:
        """Keep only the two newest evaluations in the requested stage, when specified."""
        evaluations = [item for item in evidence if item.kind == "evaluation"]
        text = question.casefold()
        stage_markers = {
            "sft": ("sft", "监督微调"),
            "cpt": ("cpt", "预训练"),
            "dpo": ("dpo", "偏好优化"),
            "grpo": ("grpo", "强化学习"),
        }
        requested = next((markers for markers in stage_markers.values()
                          if any(marker in text for marker in markers)), None)
        if requested:
            evaluations = [item for item in evaluations if any(
                marker in f"{item.title} {item.entity_id}".casefold() for marker in requested
            )]
        return evaluations[:2]

    @staticmethod
    def _validate_citations(answer: AssistantAnswer, evidence: list[EvidenceRef]) -> None:
        allowed = {item.evidence_id for item in evidence}
        claims = [*answer.facts, *answer.interpretations]
        if any(not claim.evidence_ids or not set(claim.evidence_ids) <= allowed for claim in claims):
            raise AssistantModelError("invalid_citation", "回答的证据引用未通过校验。")

    def _append_answer(self, session_id: str, answer: AssistantAnswer,
                       evidence: list[EvidenceRef] | None = None,
                       *, message_id: str | None = None,
                       message_kind: str = "chat") -> str:
        now = datetime.now(timezone.utc)
        evidence_ids = sorted({
            identifier for claim in (*answer.facts, *answer.interpretations)
            for identifier in claim.evidence_ids
        })
        message = SessionMessage(
            message_id=message_id or f"message-{uuid4().hex[:20]}", role="assistant",
            content=answer.model_dump_json(), evidence_ids=evidence_ids,
            message_kind=message_kind, created_at=now,
        )
        self.store.append_message(session_id, message)
        return message.message_id
