import io
import json

import pytest

from onegpu_posttrain.data.importer import validate_canonical_jsonl


STAGE_RECORDS = {
    "cpt": {"id": "c1", "text": "proof text", "source": "fixture", "metadata": {}},
    "sft": {"id": "s1", "raw_problem": "1+1", "messages": [
        {"role": "user", "content": "1+1"}, {"role": "assistant", "content": "2"}],
        "prompt_template_id": "fixture-v1", "metadata": {}},
    "preference": {"id": "p1", "raw_problem": "1+1", "prompt": [
        {"role": "user", "content": "1+1"}], "chosen": "2", "rejected": "3",
        "prompt_template_id": "fixture-v1", "metadata": {}},
    "grpo": {"id": "g1", "raw_problem": "1+1", "prompt": [
        {"role": "user", "content": "1+1"}], "answer": "2",
        "prompt_template_id": "fixture-v1", "metadata": {}},
}


@pytest.mark.parametrize("stage", ["cpt", "sft", "preference", "grpo"])
def test_accepts_each_canonical_stage(stage):
    raw = json.dumps(STAGE_RECORDS[stage]).encode("utf-8") + b"\n"
    written = []
    result = validate_canonical_jsonl(io.BytesIO(raw), stage, on_record=written.append)
    assert result.valid
    assert result.record_count == 1
    assert written[0]["id"] == STAGE_RECORDS[stage]["id"]


@pytest.mark.parametrize("raw,code", [
    (b"", "empty_file"),
    (b"not-json\n", "invalid_json"),
    (b"[]\n", "invalid_record"),
    (b"\xff\n", "invalid_utf8"),
])
def test_rejects_empty_or_malformed_jsonl(raw, code):
    result = validate_canonical_jsonl(io.BytesIO(raw), "sft")
    assert not result.valid
    assert result.errors[0].code == code


def test_rejects_wrong_stage_and_forbidden_fields_without_echoing_record():
    bad = dict(STAGE_RECORDS["sft"], secret_text="do-not-echo")
    raw = json.dumps(bad).encode() + b"\n"
    result = validate_canonical_jsonl(io.BytesIO(raw), "sft")
    assert not result.valid
    rendered = json.dumps(result.report())
    assert "secret_text" in rendered
    assert "do-not-echo" not in rendered


def test_row_and_byte_limits_are_enforced_and_errors_are_bounded():
    valid_line = json.dumps(STAGE_RECORDS["cpt"]).encode()
    raw = b"\n".join([valid_line] * 3) + b"\n"
    result = validate_canonical_jsonl(io.BytesIO(raw), "cpt", max_records=2)
    assert result.record_count == 2
    assert result.errors[0].code == "row_limit"

    malformed = validate_canonical_jsonl(io.BytesIO(b"bad\n" * 5), "cpt", max_errors=1)
    assert malformed.errors[0].code == "invalid_json"
    assert malformed.errors_truncated

    oversized = validate_canonical_jsonl(io.BytesIO(b"{}\n"), "cpt", max_bytes=1)
    assert oversized.errors[0].code == "size_limit"


def test_rejects_unsupported_stage():
    with pytest.raises(ValueError, match="unsupported import stage"):
        validate_canonical_jsonl(io.BytesIO(b"{}\n"), "unknown")
