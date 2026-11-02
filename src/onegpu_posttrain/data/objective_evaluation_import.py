"""Bounded JSONL adapter for user-authored multiple-choice evaluation suites."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, BinaryIO

from .importer import MAX_IMPORT_BYTES, MAX_IMPORT_ERRORS, MAX_IMPORT_RECORDS, ImportIssue
from ..schemas import EvalRecord

OBJECTIVE_EVALUATION_FORMAT = "objective-choice-jsonl-v1"
OBJECTIVE_EVALUATION_CONVERTER = "objective-choice-converter-v1"
MAX_PREVIEW_RECORDS = 3


@dataclass(frozen=True)
class ObjectiveEvaluationConversion:
    canonical_bytes: bytes
    records: tuple[dict[str, Any], ...]
    source_record_count: int
    output_record_count: int
    errors: tuple[ImportIssue, ...]
    errors_truncated: bool
    source_bytes: int

    @property
    def valid(self) -> bool:
        # Invalid rows are explicitly excluded; a partial import is usable only
        # when at least one row survived and all file-level limits were respected.
        return self.source_record_count > 0 and self.output_record_count > 0

    def report(self) -> dict[str, Any]:
        return {
            "source_format": OBJECTIVE_EVALUATION_FORMAT,
            "converter_version": OBJECTIVE_EVALUATION_CONVERTER,
            "valid": self.valid,
            "source_record_count": self.source_record_count,
            "output_record_count": self.output_record_count,
            "rejected_record_count": self.source_record_count - self.output_record_count,
            "source_bytes": self.source_bytes,
            "errors_truncated": self.errors_truncated,
            "errors": [issue.safe_dict() for issue in self.errors],
            "mapping": {"questionType": "question_type", "question": "prompt",
                        "options": "options", "correctAnswer": "answer"},
        }


def _json_array(value: Any, field: str) -> list[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValueError(f"{field} must be a JSON array") from exc
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def _answer_label(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Z]", value.strip().upper()):
        raise ValueError("correctAnswer must identify an option label")
    return value.strip().upper()


def _normalize(payload: dict[str, Any], line: int, duplicate_index: int) -> dict[str, Any]:
    allowed = {"id", "questionType", "question", "options", "correctAnswer", "tags"}
    if set(payload) - allowed:
        raise ValueError("record contains fields outside the supported evaluation format")
    kind = payload.get("questionType")
    question = payload.get("question")
    if kind not in {"single_choice", "multiple_choice", "true_false"}:
        raise ValueError("questionType must be single_choice, multiple_choice, or true_false")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if "correctAnswer" not in payload or payload.get("correctAnswer") in (None, ""):
        raise ValueError("correctAnswer is required")

    if kind == "true_false":
        if payload.get("options") not in (None, "", []):
            raise ValueError("true_false records must not declare options")
        truth = payload.get("correctAnswer")
        mapping = {"true": "true", "false": "false", "✅": "true", "❌": "false",
                   "对": "true", "错": "false", "正确": "true", "错误": "false"}
        if not isinstance(truth, str) or truth.strip().lower() not in mapping:
            raise ValueError("true_false correctAnswer must be true/false, ✅/❌, or 对/错")
        answer = mapping[truth.strip().lower()]
        options: list[str] = []
    else:
        raw_options = _json_array(payload.get("options"), "options")
        if not 2 <= len(raw_options) <= 26:
            raise ValueError("choice questions must have 2 to 26 options")
        if any(not isinstance(item, str) or not item.strip() for item in raw_options):
            raise ValueError("each option must be a non-empty string")
        options = [item.strip() for item in raw_options]
        raw_answer = payload.get("correctAnswer")
        if kind == "single_choice":
            answer = _answer_label(raw_answer)
            answers = [answer]
        else:
            answers_raw = _json_array(raw_answer, "correctAnswer")
            answers = sorted({_answer_label(item) for item in answers_raw})
            if not answers or len(answers) != len(answers_raw):
                raise ValueError("multiple_choice correctAnswer must be a non-empty set of unique labels")
            answer = ",".join(answers)
        if any(ord(label) - ord("A") >= len(options) for label in answers):
            raise ValueError("correctAnswer references an option that does not exist")

    source_fingerprint = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                                   separators=(",", ":")).encode("utf-8")).hexdigest()
    supplied_id = payload.get("id")
    record_id = (supplied_id.strip() if isinstance(supplied_id, str) and
                 re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}", supplied_id.strip())
                 else f"objective-{hashlib.sha256(f'{source_fingerprint}:{duplicate_index}'.encode()).hexdigest()[:24]}")
    record = {
        "id": record_id,
        "problem_id": record_id,
        "raw_problem": question.strip(),
        "prompt": question.strip(),
        "answer": answer,
        "eval_prompt_version": OBJECTIVE_EVALUATION_CONVERTER,
        "question_type": kind,
        "options": options,
        "metadata": {"dataset_role": "evaluation_only",
                     "source_locator": {"kind": "jsonl_line", "line": line}},
    }
    return EvalRecord.model_validate(record).model_dump(mode="json")


def convert_objective_evaluation_jsonl(
    stream: BinaryIO, *, max_bytes: int = MAX_IMPORT_BYTES,
    max_records: int = MAX_IMPORT_RECORDS, max_errors: int = MAX_IMPORT_ERRORS,
    preview_limit: int | None = None,
) -> ObjectiveEvaluationConversion:
    if min(max_bytes, max_records, max_errors) < 1:
        raise ValueError("import limits must be positive")
    output = bytearray()
    records: list[dict[str, Any]] = []
    errors: list[ImportIssue] = []
    errors_truncated = False
    seen: dict[str, int] = {}
    seen_record_ids: set[str] = set()
    source_bytes = source_rows = output_rows = 0

    def issue(line: int, code: str, message: str, field: str | None = None) -> None:
        nonlocal errors_truncated
        if len(errors) < max_errors:
            errors.append(ImportIssue(line, code, message[:240], field))
        else:
            errors_truncated = True

    while True:
        raw = stream.readline(max_bytes - source_bytes + 1)
        if not raw:
            break
        source_rows += 1
        source_bytes += len(raw)
        if source_bytes > max_bytes:
            issue(source_rows, "size_limit", f"文件超过 {max_bytes} 字节上限")
            break
        if source_rows > max_records:
            issue(source_rows, "row_limit", f"记录数超过 {max_records} 行上限")
            break
        try:
            text = raw.decode("utf-8-sig" if source_rows == 1 else "utf-8").rstrip("\r\n")
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError("each JSONL line must be an object")
            fingerprint = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                                     separators=(",", ":")).encode("utf-8")).hexdigest()
            occurrence = seen.get(fingerprint, 0)
            seen[fingerprint] = occurrence + 1
            record = _normalize(payload, source_rows, occurrence)
            if record["id"] in seen_record_ids:
                issue(source_rows, "duplicate_record_id", "id 与前序有效记录重复", "id")
                continue
            encoded = json.dumps(record, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8") + b"\n"
            if len(output) + len(encoded) > max_bytes:
                issue(source_rows, "canonical_size_limit", f"转换结果超过 {max_bytes} 字节上限")
                break
            output.extend(encoded)
            output_rows += 1
            seen_record_ids.add(record["id"])
            if preview_limit is None or len(records) < preview_limit:
                records.append(record)
        except UnicodeDecodeError:
            issue(source_rows, "invalid_utf8", "该行不是有效 UTF-8")
        except (json.JSONDecodeError, RecursionError):
            issue(source_rows, "invalid_json", "该行不是有效 JSON 对象")
        except (ValueError, TypeError, KeyError) as exc:
            # Never echo user-provided field values or question text.
            message = str(exc)
            field = None
            if "correctAnswer" in message:
                field = "correctAnswer"
            elif "questionType" in message:
                field = "questionType"
            elif "options" in message:
                field = "options"
            elif "question" in message:
                field = "question"
            issue(source_rows, "invalid_objective_record", message, field)
    if source_rows == 0:
        issue(1, "empty_file", "文件不能为空")
    return ObjectiveEvaluationConversion(bytes(output), tuple(records), source_rows, output_rows,
                                         tuple(errors), errors_truncated, source_bytes)
