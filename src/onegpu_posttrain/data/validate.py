from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..schemas import STAGE_MODELS
from .io import read_jsonl


@dataclass
class ValidationResult:
    stage: str
    path: str
    records: int
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok}


def validate_jsonl(path: str | Path, stage: str) -> ValidationResult:
    if stage not in STAGE_MODELS:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {sorted(STAGE_MODELS)}")
    model = STAGE_MODELS[stage]
    errors: list[str] = []
    count = 0
    for count, row in enumerate(read_jsonl(path), 1):
        try:
            model.model_validate(row)
        except (ValidationError, TypeError, ValueError) as exc:
            errors.append(f"record {count}: {exc}")
    return ValidationResult(stage=stage, path=str(path), records=count, errors=errors)
