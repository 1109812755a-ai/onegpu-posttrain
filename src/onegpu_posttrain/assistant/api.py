"""Small request contracts for the local workspace assistant HTTP API."""

from .contracts import AssistantContract


class AssistantAskRequest(AssistantContract):
    question: str
    session_id: str | None = None
    assistant_profile_id: str = "workspace-analyst"


class AssistantProfileRequest(AssistantContract):
    name: str
    role_instruction: str = ""


class AssistantProfileDraftRequest(AssistantContract):
    description: str


class MemoryDraftRequest(AssistantContract):
    content: str


class ContextSummaryRequest(AssistantContract):
    summary: str
