"""Explicit, deterministic adapters for user supplied SFT JSONL formats."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, BinaryIO

from .importer import MAX_IMPORT_BYTES, MAX_IMPORT_ERRORS, MAX_IMPORT_RECORDS, ImportIssue
from pydantic import ValidationError

CONVERTER_VERSION = "sft-source-format-v1"
SFT_SOURCE_FORMATS = ("canonical-v1", "alpaca-instruction-v1", "messages-role-content-v1", "sharegpt-conversations-v1")
MAX_CANONICAL_BYTES = MAX_IMPORT_BYTES
MAX_PREVIEW_RECORDS = 3


@dataclass(frozen=True)
class ConversionResult:
    source_format: str
    canonical_bytes: bytes
    records: tuple[dict[str, Any], ...]
    source_record_count: int
    output_record_count: int
    errors: tuple[ImportIssue, ...]
    errors_truncated: bool
    source_bytes: int
    config: dict[str, Any]

    @property
    def valid(self) -> bool:
        return self.source_record_count > 0 and not self.errors

    def report(self) -> dict[str, Any]:
        return {"source_format": self.source_format, "valid": self.valid,
                "source_record_count": self.source_record_count,
                "output_record_count": self.output_record_count,
                "source_bytes": self.source_bytes, "converter_version": CONVERTER_VERSION,
                "config": self.config, "errors_truncated": self.errors_truncated,
                "errors": [issue.safe_dict() for issue in self.errors]}


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if not value.strip():
        raise ValueError(f"{field} must not be empty")
    return value


def _message(role: Any, content: Any, field: str) -> dict[str, str]:
    if role not in {"system", "user", "assistant", "human", "gpt"}:
        # Do not echo unknown role values; inputs may contain private text.
        raise ValueError(f"{field} has unsupported role")
    mapped = {"human": "user", "gpt": "assistant"}.get(role, role)
    if not isinstance(content, str):
        raise ValueError(f"{field}.content must be a string; tool/function and multimodal payloads are unsupported")
    if not content.strip():
        raise ValueError(f"{field}.content must not be empty")
    return {"role": mapped, "content": content}


def _convert(payload: dict[str, Any], source_format: str, line: int, duplicate_index: int) -> dict[str, Any]:
    if source_format == "canonical-v1":
        return payload
    source_id = payload.get("id")
    if source_format == "alpaca-instruction-v1":
        if set(payload) - {"id", "instruction", "input", "output", "system"}:
            raise ValueError("record contains fields outside the selected Alpaca format")
        instruction = payload.get("instruction", "")
        input_text = payload.get("input", "")
        if not isinstance(instruction, str) or not isinstance(input_text, str):
            raise ValueError("instruction and input must be strings")
        if not instruction.strip() and not input_text.strip():
            raise ValueError("instruction or input must contain non-empty text")
        user_content = (f"{instruction}\n\n{input_text}" if instruction.strip() and input_text.strip()
                        else instruction if instruction.strip() else input_text)
        messages = []
        if payload.get("system") not in (None, ""):
            messages.append(_message("system", payload["system"], "system"))
        messages.extend((_message("user", user_content, "instruction/input"),
                         _message("assistant", payload.get("output"), "output")))
    elif source_format == "messages-role-content-v1":
        if set(payload) - {"id", "messages"}:
            raise ValueError("record contains fields outside the selected messages format")
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError("messages must be a non-empty array")
        if any(not isinstance(item, dict) or set(item) != {"role", "content"} for item in raw_messages):
            raise ValueError("each messages item must contain only role and content")
        messages = [_message(item.get("role") if isinstance(item, dict) else None,
                             item.get("content") if isinstance(item, dict) else None,
                             f"messages[{index}]") for index, item in enumerate(raw_messages)]
    elif source_format == "sharegpt-conversations-v1":
        if set(payload) - {"id", "conversations"}:
            raise ValueError("record contains fields outside the selected ShareGPT format")
        raw_messages = payload.get("conversations")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError("conversations must be a non-empty array")
        messages = []
        for index, item in enumerate(raw_messages):
            if not isinstance(item, dict):
                raise ValueError(f"conversations[{index}] must be an object")
            if set(item) != {"from", "value"}:
                raise ValueError(f"conversations[{index}] must contain only from and value")
            messages.append(_message(item.get("from"), item.get("value"), f"conversations[{index}]"))
    else:
        raise ValueError("unsupported source_format")
    if not any(message["role"] == "user" for message in messages):
        raise ValueError("at least one user message is required")
    if not any(message["role"] == "assistant" for message in messages):
        raise ValueError("at least one assistant answer is required")
    if messages[-1]["role"] != "assistant":
        raise ValueError("the final message must be an assistant answer")
    raw_problem = "\n\n".join(message["content"] for message in messages if message["role"] == "user")
    source_fingerprint = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                                   separators=(",", ":")).encode("utf-8")).hexdigest()
    record_id = f"sft-src-{hashlib.sha256(f'{source_fingerprint}:{duplicate_index}'.encode()).hexdigest()[:24]}"
    metadata: dict[str, Any] = {"source_format": source_format, "source_locator": {"kind": "jsonl_line", "line": line}}
    if isinstance(source_id, str) and source_id.strip():
        metadata["source_record_id"] = source_id
    return {"id": record_id, "raw_problem": raw_problem, "messages": messages,
            "prompt_template_id": "imported-sft-v1", "metadata": metadata}


def convert_sft_jsonl(stream: BinaryIO, source_format: str, *, max_bytes: int = MAX_IMPORT_BYTES,
                      max_records: int = MAX_IMPORT_RECORDS, max_errors: int = MAX_IMPORT_ERRORS,
                      preview_limit: int | None = None) -> ConversionResult:
    """Convert an explicitly selected format. Any invalid row invalidates the full file."""
    if source_format not in SFT_SOURCE_FORMATS:
        raise ValueError("unsupported source_format")
    if min(max_bytes, max_records, max_errors) < 1:
        raise ValueError("import limits must be positive")
    from ..schemas import SFTRecord

    errors: list[ImportIssue] = []
    truncated = False
    output = bytearray()
    samples: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    source_bytes = 0
    source_rows = 0
    valid_count = 0

    def issue(line: int, code: str, msg: str, field: str | None = None) -> None:
        nonlocal truncated
        if len(errors) < max_errors:
            errors.append(ImportIssue(line, code, msg[:240], field))
        else:
            truncated = True

    while True:
        line_raw = stream.readline(max_bytes - source_bytes + 1)
        if not line_raw:
            break
        source_rows += 1
        source_bytes += len(line_raw)
        if source_bytes > max_bytes:
            issue(source_rows, "size_limit", f"文件超过 {max_bytes} 字节上限")
            break
        if source_rows > max_records:
            issue(source_rows, "row_limit", f"记录数超过 {max_records} 行上限")
            break
        try:
            text = line_raw.decode("utf-8-sig" if source_rows == 1 else "utf-8").rstrip("\r\n")
        except UnicodeDecodeError:
            issue(source_rows, "invalid_utf8", "该行不是有效 UTF-8")
            continue
        if not text.strip():
            issue(source_rows, "empty_line", "JSONL 不允许空行")
            continue
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, RecursionError):
            issue(source_rows, "invalid_json", "该行不是有效 JSON 对象")
            continue
        if not isinstance(payload, dict):
            issue(source_rows, "invalid_record", "每行必须是 JSON 对象")
            continue
        try:
            if source_format != "canonical-v1":
                for unsupported in ("tool_calls", "function_call", "image", "audio", "video"):
                    if unsupported in payload:
                        raise ValueError(f"{unsupported} payload is unsupported")
            fingerprint = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                                     separators=(",", ":")).encode("utf-8")).hexdigest()
            duplicate = seen.get(fingerprint, 0)
            seen[fingerprint] = duplicate + 1
            converted = _convert(payload, source_format, source_rows, duplicate)
            canonical = SFTRecord.model_validate(converted).model_dump(mode="json")
        except ValidationError as exc:
            detail = exc.errors(include_url=False, include_context=False, include_input=False)[0]
            field = ".".join(str(part) for part in detail.get("loc", ())) or None
            issue(source_rows, "schema_validation", str(detail.get("msg", "字段校验失败")), field)
            continue
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            # Validation messages are safe static schema messages; never return payload values.
            message = str(exc) if isinstance(exc, ValueError) else "记录格式无效"
            field = None
            if ":" in message and message.split(":", 1)[0].replace("_", "").isalnum():
                field, message = message.split(":", 1)
            issue(source_rows, "conversion_error", message, field)
            continue
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
        if len(output) + len(encoded) > MAX_CANONICAL_BYTES:
            issue(source_rows, "canonical_size_limit", f"转换结果超过 {MAX_CANONICAL_BYTES} 字节上限")
            break
        output.extend(encoded)
        valid_count += 1
        if preview_limit is None or len(samples) < preview_limit:
            samples.append(canonical)
    if source_rows == 0:
        issue(1, "empty_file", "文件不能为空")
    elif valid_count == 0 and not errors:
        issue(1, "empty_dataset", "文件中没有可导入记录")
    config = {"source_format": source_format, "converter_version": CONVERTER_VERSION,
              "role_mapping": {"human": "user", "gpt": "assistant"} if source_format == "sharegpt-conversations-v1" else {},
              "record_id_protocol": "source-row-content-occurrence-v1", "prompt_template_id": "imported-sft-v1"}
    return ConversionResult(source_format, bytes(output), tuple(samples), source_rows, valid_count,
                           tuple(errors), truncated, source_bytes, config)
