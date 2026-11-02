"""Deterministic, explicitly scoped SFT answer-noise transform (v1)."""

from __future__ import annotations

import copy
import hashlib
import re
from typing import Any

from ..evaluation.answers import extract_answer

_BOXED_INTEGER = re.compile(r"(\\boxed\s*\{\s*)([-+]?\d+)(\s*\})")
_FINAL_INTEGER = re.compile(r"((?:final\s+answer|answer)\s*[:=]\s*)([-+]?\d+)(\s*\.?)$", re.IGNORECASE)


def _replace_supported_integer(solution: str, replacement: str) -> tuple[str | None, str | None]:
    boxed = list(_BOXED_INTEGER.finditer(solution))
    if boxed:
        match = boxed[-1]
        return solution[:match.start(2)] + replacement + solution[match.end(2):], match.group(2)
    final = list(_FINAL_INTEGER.finditer(solution))
    if final:
        match = final[-1]
        return solution[:match.start(2)] + replacement + solution[match.end(2):], match.group(2)
    return None, None


def inject_sft_answer_noise(
    records: list[dict[str, Any]],
    *,
    seed: int,
    target_ratio: float,
) -> dict[str, Any]:
    """Change only a supported integer final answer in SFT assistant messages.

    A trusted reference is required and the changed answer must be independently
    non-equivalent under the project's answer verifier. Unsupported examples are
    skipped; the function never duplicates rows to meet the requested ratio.
    """
    if seed < 0 or not 0 <= target_ratio <= 1:
        raise ValueError("seed 必须非负，target_ratio 必须在 [0, 1]")
    output = copy.deepcopy(records)
    for record in output:
        if "chosen" in record or "rejected" in record:
            raise ValueError("最终答案噪声 v1 只接受 SFT 数据，不处理 Preference 排序")
        if "messages" not in record or not isinstance(record["messages"], list):
            raise ValueError("输入必须符合 SFT messages 结构")
    target_count = int(len(output) * target_ratio + 0.5)
    ordered = sorted(
        range(len(output)),
        key=lambda index: hashlib.sha256(
            f"{seed}:{output[index].get('id', index)}".encode()
        ).hexdigest(),
    )
    selected_indices = set(ordered[:target_count])
    skipped: dict[str, int] = {}
    applied = 0
    for index, record in enumerate(output):
        if index not in selected_indices:
            continue
        metadata = record.setdefault("metadata", {})
        reference = metadata.get("reference_answer")
        if reference is None or isinstance(reference, bool) or not isinstance(reference, (str, int, float)):
            reason = "missing_or_unsupported_reference"
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        if not re.fullmatch(r"[-+]?\d+", str(reference).strip()):
            skipped["unsupported_reference_format"] = skipped.get("unsupported_reference_format", 0) + 1
            continue
        assistant_messages = [message for message in record["messages"]
                              if isinstance(message, dict) and message.get("role") == "assistant"]
        if not assistant_messages or not isinstance(assistant_messages[-1].get("content"), str):
            skipped["missing_assistant_solution"] = skipped.get("missing_assistant_solution", 0) + 1
            continue
        solution = assistant_messages[-1]["content"]
        extracted = extract_answer(solution)
        if extracted is None or not re.fullmatch(r"[-+]?\d+", str(extracted).strip()):
            skipped["unsupported_final_answer_format"] = skipped.get("unsupported_final_answer_format", 0) + 1
            continue
        value = int(str(extracted).strip())
        sign = 1 if int(hashlib.sha256(f"{seed}:{record.get('id', index)}".encode()).hexdigest()[-1], 16) % 2 == 0 else -1
        changed_value = str(value + sign)
        changed_solution, old_value = _replace_supported_integer(solution, changed_value)
        if changed_solution is None or old_value is None:
            skipped["unsupported_final_answer_format"] = skipped.get("unsupported_final_answer_format", 0) + 1
            continue
        if int(changed_value) == int(str(reference).strip()):
            # Defensive check for custom answer markers and parser behavior.
            skipped["answer_change_not_proven_non_equivalent"] = skipped.get(
                "answer_change_not_proven_non_equivalent", 0
            ) + 1
            continue
        assistant_messages[-1]["content"] = changed_solution
        metadata["correctness_math_verify"] = False
        metadata["answer_noise"] = {
            "transform": "sft_final_integer_answer_noise_v1",
            "version": "1",
            "seed": seed,
            "target_ratio": target_ratio,
            "parent_record_ids": list(metadata.get("parent_record_ids", [])),
            "original_answer": old_value,
            "injected_answer": changed_value,
            "answer_equivalence_method": "integer_scalar_inequality:v1",
            "equivalent_to_reference": False,
            "status": "confirmed_non_equivalent",
        }
        applied += 1
    actual_ratio = applied / len(output) if output else 0.0
    for record in output:
        noise = record.get("metadata", {}).get("answer_noise")
        if isinstance(noise, dict):
            noise["actual_ratio"] = actual_ratio
            noise["target_count"] = target_count
            noise["actual_count"] = applied
    return {
        "records": output,
        "report": {
            "transform": "sft_final_integer_answer_noise_v1",
            "version": "1",
            "seed": seed,
            "target_ratio": target_ratio,
            "target_count": target_count,
            "actual_count": applied,
            "actual_ratio": actual_ratio,
            "skipped": skipped,
            "input_count": len(records),
            "output_count": len(output),
            "preference_affected": False,
        },
    }
