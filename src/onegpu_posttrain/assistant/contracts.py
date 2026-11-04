"""Framework-neutral contracts for the workspace experiment assistant.

These models intentionally describe workspace identities and bounded summaries,
not filesystem paths, raw dataset rows, or executable tool arguments.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AssistantContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, protected_namespaces=()
    )


EvidenceKind = Literal[
    "run", "configuration", "log", "lineage", "model", "dataset", "quality_report", "evaluation"
]
EvidenceStatus = Literal[
    "available", "partial", "failed", "legacy", "missing", "unknown", "incompatible"
]
TaskStatus = Literal["queued", "running", "needs_clarification", "succeeded", "failed"]
QuestionKind = Literal[
    "recent_failed_runs", "compare_evaluations", "trace_data_lineage", "general"
]

# This is the complete MVP capability surface. It deliberately contains no
# filesystem, shell, Python, network, or experiment-mutation operation.
MVP_READ_CAPABILITIES = frozenset({
    "workspace.search_runs",
    "workspace.read_run_summary",
    "workspace.read_model_lineage",
    "workspace.read_dataset_summary",
    "workspace.read_quality_summary",
    "workspace.read_evaluation_summary",
})


class EvidenceFact(AssistantContract):
    key: str = Field(min_length=1, max_length=120)
    label: str = Field(min_length=1, max_length=160)
    value: str | int | float | bool | None
    unit: str | None = Field(default=None, max_length=40)
    denominator: str | int | None = None

    @field_validator("denominator")
    @classmethod
    def bound_denominator(cls, value: str | int | None) -> str | int | None:
        if isinstance(value, str) and len(value) > 80:
            raise ValueError("denominator text exceeds 80 characters")
        if isinstance(value, int) and (isinstance(value, bool) or value < 0):
            raise ValueError("denominator count must be a non-negative integer")
        return value


class EvidenceRef(AssistantContract):
    evidence_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,199}$")
    kind: EvidenceKind
    entity_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=240)
    status: EvidenceStatus
    version: str | None = Field(default=None, max_length=200)
    content_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    summary: str = Field(default="", max_length=1600)
    facts: list[EvidenceFact] = Field(default_factory=list, max_length=80)
    limitations: list[str] = Field(default_factory=list, max_length=20)
    source_href: str = Field(pattern=r"^/[^\r\n]{0,500}$")

    @field_validator("source_href")
    @classmethod
    def require_local_href(cls, value: str) -> str:
        if value.startswith("//"):
            raise ValueError("source_href must be a local absolute path")
        return value


class ConversationTurn(AssistantContract):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class AssistantProfile(AssistantContract):
    profile_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
    name: str = Field(min_length=1, max_length=80)
    role_instruction: str = Field(default="", max_length=3000)
    built_in: bool = False
    created_at: datetime
    updated_at: datetime


class WorkspaceMemory(AssistantContract):
    memory_id: str = Field(pattern=r"^memory-[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
    content: str = Field(min_length=1, max_length=1000)
    scope: Literal["workspace"] = "workspace"
    status: Literal["active", "disabled"] = "active"
    created_at: datetime
    updated_at: datetime


class ContextSummary(AssistantContract):
    summary_id: str = Field(pattern=r"^summary-[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
    session_id: str
    revision: int = Field(ge=1)
    summary: str = Field(min_length=1, max_length=4000)
    source_message_ids: list[str] = Field(min_length=1, max_length=200)
    created_at: datetime
    updated_at: datetime


class SessionMessage(AssistantContract):
    message_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=50)
    message_kind: Literal["chat", "consent_notice"] = "chat"
    created_at: datetime


class ConversationSession(AssistantContract):
    schema_version: Literal["1"] = "1"
    session_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
    title: str = Field(min_length=1, max_length=160)
    assistant_profile_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
    model_profile_id: str = Field(default="dashscope-qwen3.6-plus-local", max_length=80)
    data_egress_consent_at: datetime | None = None
    data_egress_consent_version: str | None = Field(default=None, max_length=40)
    pending_egress_question: str | None = Field(default=None, max_length=2000)
    pending_egress_question_message_id: str | None = Field(default=None, max_length=80)
    created_at: datetime
    updated_at: datetime
    revision: int = Field(default=1, ge=1)
    messages: list[SessionMessage] = Field(default_factory=list, max_length=200)


class EvidenceQuery(AssistantContract):
    kind: QuestionKind
    query: str = Field(min_length=1, max_length=2000)
    selected_entity_ids: list[str] = Field(default_factory=list, max_length=20)
    limit: int = Field(default=10, ge=1, le=50)

    @field_validator("selected_entity_ids")
    @classmethod
    def accept_ids_only(cls, values: list[str]) -> list[str]:
        if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}", value) for value in values):
            raise ValueError("selected_entity_ids accepts workspace identities, not paths")
        return values


class ModelRequest(AssistantContract):
    task_id: str
    model_profile_id: str
    system_instruction: str = Field(min_length=1, max_length=12000)
    question: str = Field(min_length=1, max_length=12000)
    recent_turns: list[ConversationTurn] = Field(default_factory=list, max_length=20)
    evidence: list[EvidenceRef] = Field(default_factory=list, max_length=50)
    max_output_tokens: int = Field(default=1200, ge=64, le=8192)


class CitedClaim(AssistantContract):
    text: str = Field(min_length=1, max_length=3000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)


class AssistantAnswer(AssistantContract):
    conclusion: str = Field(min_length=1, max_length=5000)
    facts: list[CitedClaim] = Field(default_factory=list, max_length=40)
    interpretations: list[CitedClaim] = Field(default_factory=list, max_length=40)
    limitations: list[str] = Field(default_factory=list, max_length=30)
    clarification_options: list[str] = Field(default_factory=list, max_length=20)


class ModelResponse(AssistantContract):
    answer: AssistantAnswer
    model_profile_id: str
    provider_request_id: str | None = Field(default=None, max_length=200)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class AgentTaskRecord(AssistantContract):
    task_id: str
    session_id: str
    assistant_profile_id: str
    model_profile_id: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    question_message_id: str
    evidence: list[EvidenceRef] = Field(default_factory=list, max_length=100)
    answer_message_id: str | None = None
    error_code: str | None = Field(default=None, max_length=100)
    error_message: str | None = Field(default=None, max_length=1000)
    duration_ms: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    clarification_required: bool = False
    memory_ids: list[str] = Field(default_factory=list, max_length=50)
    context_summary_id: str | None = None
    langfuse_trace_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    langfuse_trace_url: str | None = Field(default=None, max_length=1000)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict, max_length=40)


class ObservationEvent(AssistantContract):
    event_name: str = Field(min_length=1, max_length=120)
    task_id: str
    occurred_at: datetime
    attributes: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict, max_length=40
    )
