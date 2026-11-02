import hashlib
import io
import json

import pytest

from onegpu_posttrain.data.source_formats import convert_sft_jsonl
from onegpu_posttrain.data.dataset_import import register_jsonl
from onegpu_posttrain.data.registry import DataSource
from onegpu_posttrain.lineage import ArtifactRef, ModelLineage, RunRecord, sha256_directory
from onegpu_posttrain.ui.training_preflight import StagePreflightRequest, preflight_stage


def _jsonl(rows):
    return b"".join(json.dumps(row, ensure_ascii=False).encode() + b"\n" for row in rows)


@pytest.mark.parametrize(("fmt", "row", "expected"), [
    ("alpaca-instruction-v1", {"instruction": "Q", "input": "context", "output": "A", "system": "Be brief"},
     [("system", "Be brief"), ("user", "Q\n\ncontext"), ("assistant", "A")]),
    ("messages-role-content-v1", {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},
     [("user", "Q"), ("assistant", "A")]),
    ("sharegpt-conversations-v1", {"id": "upstream-9", "conversations": [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}]},
     [("user", "Q"), ("assistant", "A")]),
])
def test_explicit_adapters_emit_canonical_messages(fmt, row, expected):
    result = convert_sft_jsonl(io.BytesIO(_jsonl([row])), fmt)
    assert result.valid
    record = json.loads(result.canonical_bytes)
    assert [(item["role"], item["content"]) for item in record["messages"]] == expected
    assert record["metadata"]["source_locator"] == {"kind": "jsonl_line", "line": 1}
    if "id" in row:
        assert record["metadata"]["source_record_id"] == row["id"]


def test_converter_is_deterministic_and_duplicate_occurrences_get_distinct_ids():
    row = {"instruction": "Q", "output": "A"}
    raw = _jsonl([row, row])
    first = convert_sft_jsonl(io.BytesIO(raw), "alpaca-instruction-v1")
    second = convert_sft_jsonl(io.BytesIO(raw), "alpaca-instruction-v1")
    assert first.canonical_bytes == second.canonical_bytes
    ids = [json.loads(line)["id"] for line in first.canonical_bytes.splitlines()]
    assert ids[0] != ids[1]
    assert first.source_record_count == first.output_record_count == 2


@pytest.mark.parametrize("fmt,row", [
    ("alpaca-instruction-v1", {"instruction": "secret question"}),
    ("messages-role-content-v1", {"messages": [{"role": "tool", "content": "secret"}]}),
    ("sharegpt-conversations-v1", {"conversations": [{"from": "human", "value": "Q"}, {"from": "gpt", "value": ["image"]}]}),
])
def test_bad_records_fail_without_echoing_input(fmt, row):
    result = convert_sft_jsonl(io.BytesIO(_jsonl([row])), fmt)
    assert not result.valid
    assert result.errors[0].line == 1
    assert "secret" not in json.dumps(result.report(), ensure_ascii=False)


def test_no_automatic_format_sniffing_or_unknown_format():
    with pytest.raises(ValueError, match="unsupported source_format"):
        convert_sft_jsonl(io.BytesIO(_jsonl([{"instruction": "Q", "output": "A"}])), "auto")
    canonical = convert_sft_jsonl(io.BytesIO(_jsonl([{"instruction": "Q", "output": "A"}])), "canonical-v1")
    assert not canonical.valid


def test_mixed_format_fields_are_rejected_instead_of_silently_ignored():
    mixed = {"instruction": "Q", "output": "A", "conversations": []}
    result = convert_sft_jsonl(io.BytesIO(_jsonl([mixed])), "alpaca-instruction-v1")
    assert not result.valid
    assert result.errors[0].code == "conversion_error"


def test_external_format_registration_keeps_raw_and_canonical_with_separate_digests(tmp_path):
    raw = _jsonl([{"instruction": "Q", "output": "A"}])
    converted = convert_sft_jsonl(io.BytesIO(raw), "alpaca-instruction-v1")
    origin = DataSource(source_id="upload-pending", kind="local_upload", name="fixture",
                        revision_status="unknown", license_status="unknown", declared_by="user")
    config = {**converted.config, "source_record_count": converted.source_record_count,
              "output_record_count": converted.output_record_count}
    registered = register_jsonl(io.BytesIO(converted.canonical_bytes), workspace=tmp_path,
                                output_root=tmp_path / "data/processed/user-upload", stage="sft", origin=origin,
                                raw_source_bytes=raw, source_format="alpaca-instruction-v1", converter_config=config)
    assert (registered.path / "raw/source.jsonl").read_bytes() == raw
    assert hashlib.sha256((registered.path / "raw/source.jsonl").read_bytes()).hexdigest() == registered.original_sha256
    assert registered.dataset_id != f"ds-sft-upload-{registered.canonical_sha256[:20]}"


def test_external_format_identity_includes_source_and_converter_config(tmp_path):
    raw = _jsonl([{"instruction": "Q", "output": "A"}])
    converted = convert_sft_jsonl(io.BytesIO(raw), "alpaca-instruction-v1")
    origin = DataSource(source_id="upload-pending", kind="local_upload", name="fixture",
                        revision_status="unknown", license_status="unknown", declared_by="user")
    config = {**converted.config, "source_record_count": converted.source_record_count,
              "output_record_count": converted.output_record_count}
    kwargs = dict(workspace=tmp_path, output_root=tmp_path / "data/processed/user-upload", stage="sft",
                  origin=origin, raw_source_bytes=raw, source_format="alpaca-instruction-v1", converter_config=config)
    first = register_jsonl(io.BytesIO(converted.canonical_bytes), **kwargs)
    same = register_jsonl(io.BytesIO(converted.canonical_bytes), **kwargs)
    changed = register_jsonl(io.BytesIO(converted.canonical_bytes), **{**kwargs,
        "converter_config": {**config, "converter_version": "sft-source-format-v-next"}})
    assert first.dataset_id == same.dataset_id and same.reused
    assert first.dataset_id != changed.dataset_id
    assert (first.path / "raw/source.jsonl").read_bytes() == raw
    assert (changed.path / "raw/source.jsonl").read_bytes() == raw


def test_external_source_registration_failure_rolls_back_staging(tmp_path, monkeypatch):
    import onegpu_posttrain.data.dataset_import as dataset_import

    raw = _jsonl([{"instruction": "Q", "output": "A"}])
    converted = convert_sft_jsonl(io.BytesIO(raw), "alpaca-instruction-v1")
    origin = DataSource(source_id="upload-pending", kind="local_upload", name="fixture",
                        revision_status="unknown", license_status="unknown", declared_by="user")
    output = tmp_path / "data/processed/user-upload"
    output.mkdir(parents=True)
    monkeypatch.setattr(dataset_import.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("disk error")))
    with pytest.raises(OSError, match="disk error"):
        register_jsonl(io.BytesIO(converted.canonical_bytes), workspace=tmp_path, output_root=output,
                       stage="sft", origin=origin, raw_source_bytes=raw,
                       source_format="alpaca-instruction-v1", converter_config=converted.config)
    assert list(output.iterdir()) == []


def test_converted_dataset_is_consumed_by_read_only_sft_training_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.console_command_available", lambda _: True)
    recipe = tmp_path / "recipes/mini/23-sft-1024-fallback.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("training:\n  max_length: 1024\n  num_train_epochs: 1\n  learning_rate: 0.0001\n  gradient_accumulation_steps: 8\n", encoding="utf-8")
    raw = _jsonl([{"messages": [{"role": "user", "content": "Question"},
                                {"role": "assistant", "content": "Answer"}]}])
    conversion = convert_sft_jsonl(io.BytesIO(raw), "messages-role-content-v1")
    origin = DataSource(source_id="upload-pending", kind="local_upload", name="fixture",
                        revision_status="unknown", license_status="unknown", declared_by="user")
    config = {**conversion.config, "source_record_count": conversion.source_record_count,
              "output_record_count": conversion.output_record_count}
    registered = register_jsonl(io.BytesIO(conversion.canonical_bytes), workspace=tmp_path,
                                output_root=tmp_path / "data/processed/user-upload", stage="sft", origin=origin,
                                raw_source_bytes=raw, source_format="messages-role-content-v1", converter_config=config)
    model_id = "model-cpt-fixture"
    model_dir = tmp_path / "models" / model_id
    model_dir.mkdir(parents=True)
    (model_dir / "weights.txt").write_text("fixture", encoding="utf-8")
    model_digest = sha256_directory(model_dir)
    run_id = "run-cpt-fixture"
    artifacts = tmp_path / "runs" / run_id / "model_artifacts"
    artifacts.mkdir(parents=True)
    model = ModelLineage(model_id=model_id, kind="merged", path=str(model_dir),
                         digest={"value": model_digest, "status": "available"},
                         parent_model_ids=["base-fixture"], produced_by_run_id=run_id, stage="cpt")
    (artifacts / f"{model_id}.json").write_text(model.model_dump_json(), encoding="utf-8")
    run = RunRecord(run_id=run_id, run_type="train", status="completed", started_at="2026-01-01T00:00:00Z",
                    completed_at="2026-01-01T00:01:00Z",
                    outputs=[ArtifactRef(id=model_id, kind="model", path=str(model_dir), role="merged",
                                         digest={"value": model_digest, "status": "available"})])
    (tmp_path / "runs" / run_id / "lineage.json").write_text(run.model_dump_json(), encoding="utf-8")
    result = preflight_stage(tmp_path, StagePreflightRequest(
        stage="sft", profile_id="sft-mini-1024-v1", dataset_ids=[registered.dataset_id], parent_model_id=model_id,
    ), launcher_enabled=True)
    assert result["datasets"][0]["status"] == "valid"
    assert result["datasets"][0]["content_sha256"] == registered.canonical_sha256
    assert result["execution_started"] is False

