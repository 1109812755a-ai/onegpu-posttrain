from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..data.io import read_jsonl, write_jsonl
from ..schemas import CPTRecord, GRPORecord, PreferenceRecord, SFTRecord


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def adapt_cpt(row: dict[str, Any]) -> dict[str, Any]:
    rec = CPTRecord.model_validate(row)
    # ms-swift pre-training standard format is assistant-only messages.
    return {"messages": [{"role": "assistant", "content": rec.text}]}


def adapt_sft(row: dict[str, Any]) -> dict[str, Any]:
    rec = SFTRecord.model_validate(row)
    return {"messages": [m.model_dump() for m in rec.messages]}


def adapt_preference(row: dict[str, Any]) -> dict[str, Any]:
    rec = PreferenceRecord.model_validate(row)
    messages = [m.model_dump() for m in rec.prompt]
    messages.append({"role": "assistant", "content": rec.chosen})
    return {
        "messages": messages,
        "rejected_response": rec.rejected,
    }


def adapt_grpo(row: dict[str, Any]) -> dict[str, Any]:
    rec = GRPORecord.model_validate(row)
    problem_id = rec.metadata.get("problem_id", rec.id)
    return {
        "messages": [m.model_dump() for m in rec.prompt],
        # Keep reward/verifier columns outside model inputs; ms-swift GRPO passes
        # extra dataset columns through to the reward function.
        "solution": rec.answer,
        "record_id": rec.id,
        "problem_id": problem_id,
    }


ADAPTERS = {
    "cpt": adapt_cpt,
    "sft": adapt_sft,
    "preference": adapt_preference,
    "dpo": adapt_preference,
    "simpo": adapt_preference,
    "grpo": adapt_grpo,
}


def validate_native_record(stage: str, row: dict[str, Any]) -> None:
    if "messages" not in row or not isinstance(row["messages"], list) or not row["messages"]:
        raise ValueError(f"{stage}: ms-swift native record requires non-empty messages")
    if stage in {"preference", "dpo", "simpo"}:
        rejected = row.get("rejected_response")
        if not isinstance(rejected, str) or not rejected.strip():
            raise ValueError(f"{stage}: rejected_response must be non-empty")
        if row["messages"][-1].get("role") != "assistant":
            raise ValueError(f"{stage}: chosen messages must end with assistant")
    if stage == "grpo":
        if not str(row.get("solution", "")).strip():
            raise ValueError("grpo: solution must be retained for reward")
        if row["messages"][-1].get("role") == "assistant":
            raise ValueError("grpo: prompt must not end with assistant target")


def adapt_file(stage: str, input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    stage = stage.lower()
    if stage not in ADAPTERS:
        raise ValueError(f"Unsupported ms-swift adapter stage: {stage}")
    input_path = Path(input_path)
    output_path = Path(output_path)
    adapter = ADAPTERS[stage]
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(read_jsonl(input_path), 1):
        native = adapter(row)
        validate_native_record(stage, native)
        rows.append(native)
    write_jsonl(output_path, rows)
    report = {
        "adapter": "ms-swift",
        "stage": stage,
        "input": str(input_path),
        "output": str(output_path),
        "records": len(rows),
        "input_sha256": _sha256(input_path),
        "output_sha256": _sha256(output_path),
    }
    report_path = output_path.with_suffix(output_path.suffix + ".adapter.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
