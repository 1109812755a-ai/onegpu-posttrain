"""Deterministic, reversible summaries for long workspace conversations."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from uuid import uuid4

from .budget import MAX_RECENT_TURNS
from .contracts import ContextSummary, ConversationSession

_ENTITY = re.compile(
    r"(?<![A-Za-z0-9])(?:run|eval|model|ds|dataset)-[A-Za-z0-9][A-Za-z0-9._-]{0,158}"
    r"[A-Za-z0-9](?![A-Za-z0-9._-])", re.I,
)
SUMMARY_TRIGGER_MESSAGES = MAX_RECENT_TURNS + 1


def rebuild_context_summary(session: ConversationSession, previous: ContextSummary | None = None) -> ContextSummary | None:
    """Summarize older turns as user intent and stable IDs; never assert lab facts."""
    cutoff = max(0, len(session.messages) - MAX_RECENT_TURNS)
    old_messages = session.messages[:cutoff]
    if len(session.messages) < SUMMARY_TRIGGER_MESSAGES or not old_messages:
        return previous
    source = list(dict.fromkeys([
        *(previous.source_message_ids if previous else []),
        *(message.message_id for message in old_messages),
    ]))[-200:]
    user_goals = [message.content.strip() for message in old_messages if message.role == "user"][-12:]
    entity_ids = sorted({entity for message in old_messages for entity in _ENTITY.findall(message.content)})
    rows = ["较早对话摘要（仅保留用户目标和对象引用，不代表当前实验状态）："]
    if user_goals:
        rows.append("用户目标：" + "；".join(user_goals))
    if entity_ids:
        rows.append("提及的对象 ID：" + ", ".join(entity_ids))
    rows.append("涉及运行、配置、数据或评测事实时，必须重新读取当前工作区证据。")
    return ContextSummary(
        summary_id=f"summary-{uuid4().hex[:20]}",
        session_id=session.session_id,
        revision=(previous.revision + 1) if previous else 1,
        summary="\n".join(rows)[:4000],
        source_message_ids=source,
        created_at=previous.created_at if previous else datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
