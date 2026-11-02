"""Strict, bounded validation for user-supplied canonical dataset JSONL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import BinaryIO, Callable, Any

from pydantic import ValidationError

from ..schemas import STAGE_MODELS

MAX_IMPORT_BYTES = 256 * 1024 * 1024
MAX_IMPORT_RECORDS = 250_000
MAX_IMPORT_ERRORS = 50
SUPPORTED_IMPORT_STAGES = ("cpt", "sft", "preference", "grpo", "eval")


@dataclass(frozen=True)
class ImportIssue:
    line: int
    code: str
    message: str
    field: str | None = None

    def safe_dict(self) -> dict[str, Any]:
        return {key: value for key, value in {
            "line": self.line, "code": self.code, "message": self.message, "field": self.field,
        }.items() if value is not None}


@dataclass(frozen=True)
class ImportValidation:
    stage: str
    bytes_read: int
    record_count: int
    errors: tuple[ImportIssue, ...]
    errors_truncated: bool = False

    @property
    def valid(self) -> bool:
        return self.record_count > 0 and not self.errors

    def report(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "valid": self.valid,
            "bytes_read": self.bytes_read,
            "record_count": self.record_count,
            "error_count_reported": len(self.errors),
            "errors_truncated": self.errors_truncated,
            "errors": [issue.safe_dict() for issue in self.errors],
        }


def _safe_validation_issue(line_number: int, error: ValidationError) -> ImportIssue:
    detail = error.errors(include_url=False, include_context=False, include_input=False)[0]
    location = detail.get("loc", ())
    field = ".".join(str(part) for part in location) or None
    return ImportIssue(line_number, "schema_validation", str(detail.get("msg", "字段校验失败"))[:240], field)


def validate_canonical_jsonl(
    stream: BinaryIO,
    stage: str,
    *,
    max_bytes: int = MAX_IMPORT_BYTES,
    max_records: int = MAX_IMPORT_RECORDS,
    max_errors: int = MAX_IMPORT_ERRORS,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> ImportValidation:
    """Validate records incrementally; callbacks receive only schema-valid records.

    No input text is included in reports. Callers must write callback output to a
    temporary artifact and discard it if the final report is invalid.
    """
    if stage not in SUPPORTED_IMPORT_STAGES:
        raise ValueError(f"unsupported import stage: {stage}")
    if min(max_bytes, max_records, max_errors) < 1:
        raise ValueError("import limits must be positive")

    model = STAGE_MODELS[stage]
    bytes_read = 0
    record_count = 0
    line_number = 0
    errors: list[ImportIssue] = []
    errors_truncated = False

    def add(issue: ImportIssue) -> None:
        nonlocal errors_truncated
        if len(errors) < max_errors:
            errors.append(issue)
        else:
            errors_truncated = True

    while True:
        remaining = max_bytes - bytes_read
        raw_line = stream.readline(remaining + 1)
        if not raw_line:
            break
        line_number += 1
        bytes_read += len(raw_line)
        if bytes_read > max_bytes:
            add(ImportIssue(line_number, "size_limit", f"文件超过 {max_bytes} 字节上限"))
            break
        if line_number > max_records:
            add(ImportIssue(line_number, "row_limit", f"记录数超过 {max_records} 行上限"))
            break

        try:
            text = raw_line.decode("utf-8-sig" if line_number == 1 else "utf-8")
        except UnicodeDecodeError:
            add(ImportIssue(line_number, "invalid_utf8", "该行不是有效 UTF-8"))
            continue

        text = text.rstrip("\r\n")
        if not text.strip():
            add(ImportIssue(line_number, "empty_line", "JSONL 不允许空行"))
            continue
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, RecursionError):
            add(ImportIssue(line_number, "invalid_json", "该行不是有效 JSON 对象"))
            continue
        if not isinstance(payload, dict):
            add(ImportIssue(line_number, "invalid_record", "每行必须是 JSON 对象"))
            continue
        try:
            record = model.model_validate(payload).model_dump(mode="json")
        except ValidationError as exc:
            add(_safe_validation_issue(line_number, exc))
            continue

        record_count += 1
        if on_record is not None:
            on_record(record)

    if line_number == 0:
        add(ImportIssue(1, "empty_file", "文件不能为空"))
    elif bytes_read <= max_bytes and record_count == 0 and not errors:
        add(ImportIssue(1, "empty_dataset", "文件中没有可导入记录"))

    return ImportValidation(stage, bytes_read, record_count, tuple(errors), errors_truncated)
