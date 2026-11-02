"""Shared GRPO canonical prompt construction, independent of workflow mode."""

from __future__ import annotations

from typing import Any

from ..schemas import GRPORecord, Message


def build_grpo_prompt_record(
    *, record_id: str, prompt: str, answer: str, problem_id: str,
    parent_record_ids: list[str], contract: Any, reference_source: str,
    context: dict[str, Any] | None = None,
    prompt_template_id: str = "grpo-prompt-v1",
) -> dict[str, Any]:
    """Build the shared canonical row; only trusted adapters provide extra context."""
    if not isinstance(context, (dict, type(None))):
        raise TypeError("GRPO prompt context must be an object")
    metadata = {
        "problem_id": problem_id, "data_role": "train_seed",
        "parent_record_ids": list(parent_record_ids),
        "reward_contract_id": contract.contract_id,
        "reward_contract_version": contract.version,
        "reward_contract_protocol_sha256": contract.protocol_sha256,
        "reward_contract_implementation_sha256": contract.implementation_sha256,
        "reference_source": reference_source,
    }
    if context:
        metadata.update(context)
    return GRPORecord(
        id=record_id, raw_problem=prompt, prompt=[Message(role="user", content=prompt)],
        answer=answer, prompt_template_id=prompt_template_id, metadata=metadata,
    ).model_dump(mode="json")
