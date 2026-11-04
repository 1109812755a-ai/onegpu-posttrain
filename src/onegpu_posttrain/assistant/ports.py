"""Ports owned by OneGPU, so agent framework internals stay replaceable."""

from __future__ import annotations

from typing import Protocol

from .contracts import (
    AgentTaskRecord,
    EvidenceQuery,
    EvidenceRef,
    ModelRequest,
    ModelResponse,
    ObservationEvent,
)


class ModelAdapter(Protocol):
    """Generate an answer from bounded context using a configured provider."""

    def generate(self, request: ModelRequest) -> ModelResponse: ...


class ReadOnlyEvidenceReader(Protocol):
    """Resolve workspace identities and return bounded, structured evidence."""

    def search(self, request: EvidenceQuery) -> list[EvidenceRef]: ...


class TaskStore(Protocol):
    """Persist assistant task state separately from experiment artifacts."""

    def create(self, task: AgentTaskRecord) -> None: ...

    def update(self, task: AgentTaskRecord) -> None: ...

    def get(self, task_id: str) -> AgentTaskRecord | None: ...


class Observer(Protocol):
    """Accept sanitized events; implementations may be unavailable."""

    def emit(self, event: ObservationEvent) -> tuple[str | None, str | None]: ...
