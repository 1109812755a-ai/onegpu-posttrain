"""Canonical CPT corpus checks for targeted, role-isolated text adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Literal

from .importer import ImportIssue, validate_canonical_jsonl

CPTRole = Literal["train_seed", "diagnostic_dev", "final_holdout"]
_ROLES = {"train_seed", "diagnostic_dev", "final_holdout"}


@dataclass(frozen=True)
class CPTCorpusValidation:
    """Validation result; never includes record text or source secrets."""

    valid: bool
    record_count: int
    errors: tuple[ImportIssue, ...]

    def report(self) -> dict:
        return {
            "schema": "cpt-corpus-canonical-v1",
            "valid": self.valid,
            "record_count": self.record_count,
            "error_count": len(self.errors),
            "errors": [item.safe_dict() for item in self.errors],
            "semantic_factuality": "not_measured",
        }


def validate_cpt_corpus_jsonl(
    stream: BinaryIO,
    *,
    expected_role: CPTRole = "train_seed",
    max_bytes: int = 256 * 1024 * 1024,
    max_records: int = 250_000,
    max_errors: int = 50,
) -> CPTCorpusValidation:
    """Validate CPT JSONL (`id`, `text`, `source`, `metadata`) and role isolation.

    CPT is a corpus-text objective, not instruction tuning: prompt/answer fields
    are rejected by the canonical CPT schema. Targeted datasets must explicitly
    tag metadata.data_role; an absent label is never assumed to be train-safe.
    """
    if expected_role not in _ROLES:
        raise ValueError(f"unsupported CPT role: {expected_role}")
    raw_records: list[dict] = []
    base = validate_canonical_jsonl(
        stream, "cpt", max_bytes=max_bytes, max_records=max_records,
        max_errors=max_errors, on_record=raw_records.append,
    )
    errors = list(base.errors)
    if not errors:
        seen: set[str] = set()
        for line, row in enumerate(raw_records, 1):
            record_id = row.get("id")
            if record_id in seen:
                errors.append(ImportIssue(line, "duplicate_record_id", "CPT record id 重复", "id"))
                if len(errors) >= max_errors:
                    break
            seen.add(record_id)
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            role = metadata.get("data_role")
            if role not in _ROLES:
                errors.append(ImportIssue(line, "missing_or_invalid_role", "metadata.data_role 必须是 train_seed/diagnostic_dev/final_holdout", "metadata.data_role"))
            elif role != expected_role:
                errors.append(ImportIssue(line, "role_isolation_violation", f"记录角色 {role} 不允许进入 {expected_role} 语料", "metadata.data_role"))
            parents = metadata.get("parent_record_ids", [])
            if not isinstance(parents, list) or any(not isinstance(parent, str) or not parent for parent in parents):
                errors.append(ImportIssue(line, "invalid_parent_ids", "metadata.parent_record_ids 必须是字符串 ID 列表", "metadata.parent_record_ids"))
            if len(errors) >= max_errors:
                break
    return CPTCorpusValidation(
        valid=base.valid and not errors,
        record_count=base.record_count,
        errors=tuple(errors[:max_errors]),
    )
