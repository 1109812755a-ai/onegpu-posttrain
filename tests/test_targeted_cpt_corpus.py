from __future__ import annotations

import io
import json

import pytest

from onegpu_posttrain.data.cpt_corpus import validate_cpt_corpus_jsonl


def row(record_id="cpt-1", role="train_seed", **extra):
    metadata = {"data_role": role, "parent_record_ids": ["raw-1"]}
    metadata.update(extra.pop("metadata", {}))
    return {"id": record_id, "text": "A mathematical source text.", "source": "fixture",
            "metadata": metadata, **extra}


def jsonl(*rows):
    return io.BytesIO(b"".join((json.dumps(item, ensure_ascii=False) + "\n").encode() for item in rows))


def test_accepts_text_corpus_without_question_answer_schema():
    result = validate_cpt_corpus_jsonl(jsonl(row()))
    assert result.valid
    assert result.record_count == 1
    assert result.report()["semantic_factuality"] == "not_measured"


@pytest.mark.parametrize("record,code", [
    ({"id": "missing-text", "source": "fixture", "metadata": {"data_role": "train_seed"}}, "schema_validation"),
    ({**row(), "metadata": {"parent_record_ids": ["raw-1"]}}, "missing_or_invalid_role"),
    (row(metadata={"data_role": "mystery"}), "missing_or_invalid_role"),
    ({**row(), "question": "not CPT"}, "schema_validation"),
    (row("same"), "duplicate_record_id"),
])
def test_rejects_invalid_cpt_jsonl(record, code):
    rows = [record, row("same")] if code == "duplicate_record_id" else [record]
    result = validate_cpt_corpus_jsonl(jsonl(*rows))
    assert not result.valid
    assert code in {issue.code for issue in result.errors}


@pytest.mark.parametrize("role", ["diagnostic_dev", "final_holdout"])
def test_training_corpus_cannot_include_diagnostic_or_final_holdout(role):
    result = validate_cpt_corpus_jsonl(jsonl(row(role=role)), expected_role="train_seed")
    assert not result.valid
    assert result.errors[0].code == "role_isolation_violation"


def test_holdout_file_can_only_be_validated_as_holdout_role():
    assert validate_cpt_corpus_jsonl(jsonl(row(role="final_holdout")), expected_role="final_holdout").valid
    assert not validate_cpt_corpus_jsonl(jsonl(row(role="train_seed")), expected_role="final_holdout").valid
