"""File-based registration for immutable targeted-data plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .targeted import TargetedGenerationPlan


def load_targeted_plan(path: str | Path) -> TargetedGenerationPlan:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "plan" in payload:
        plan = TargetedGenerationPlan.model_validate(payload["plan"])
        if payload.get("plan_id") != plan.plan_id or payload.get("config_sha256") != plan.config_sha256:
            raise ValueError("generation plan 身份摘要不匹配")
        return plan
    if not isinstance(payload, dict):
        raise TypeError("generation plan 必须是 YAML/JSON 对象")
    return TargetedGenerationPlan.model_validate(payload)


def save_targeted_plan(plan: TargetedGenerationPlan, output_path: str | Path) -> dict[str, str]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "plan_id": plan.plan_id,
        "config_sha256": plan.config_sha256,
        "plan": plan.model_dump(mode="json"),
        "status": "registered_not_executed",
    }
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return {"plan_id": plan.plan_id, "config_sha256": plan.config_sha256,
            "status": "registered_not_executed", "path": str(output_path)}
