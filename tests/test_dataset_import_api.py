import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from onegpu_posttrain.ui import server
from onegpu_posttrain.ui.server import create_app
from onegpu_posttrain.lineage_validation import validate_lineage_root


def _record():
    return (json.dumps({
        "id": "api-sft-1", "raw_problem": "1+1?",
        "messages": [{"role": "user", "content": "1+1?"}, {"role": "assistant", "content": "2"}],
        "prompt_template_id": "chat-v1",
    }) + "\n").encode()


def test_import_api_registers_valid_dataset_and_exposes_capabilities(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        response = client.post(
            "/api/data/import?stage=sft&source_name=fixture&license_status=unknown&revision_status=unknown",
            content=_record(), headers={"content-type": "application/x-ndjson", "x-filename": "../../escape.jsonl"},
        )
        assert response.status_code == 201, response.text
        payload = response.json()
        assert payload["stage"] == "sft"
        assert payload["record_count"] == 1
        assert payload["quality_status"] == "NOT_EVALUATED"
        assert payload["manifest_path"].startswith("data/processed/user-upload/")
        assert not (tmp_path.parent / "escape.jsonl").exists()
        index = client.get("/api/data").json()
        assert any(dataset["dataset_id"] == payload["dataset_id"]
                   for build in index["builds"] for dataset in build["datasets"])


def test_import_api_rejects_bad_jsonl_without_echoing_raw_record(tmp_path):
    secret_record = b'{"id":"RAW_SENTINEL_DO_NOT_ECHO"}\n'
    with TestClient(create_app(tmp_path)) as client:
        response = client.post(
            "/api/data/import?stage=sft", content=secret_record,
            headers={"content-type": "application/x-ndjson"},
        )
        assert response.status_code == 422
        assert "RAW_SENTINEL_DO_NOT_ECHO" not in response.text
        assert "schema_validation" in response.text
        assert not list((tmp_path / "data/processed/user-upload").glob("ds-*"))


def test_import_api_rejects_unsupported_stage_and_content_type(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/api/data/import?stage=eval", content=b"{}\n",
                           headers={"content-type": "application/x-ndjson"}).status_code == 422
        assert client.post("/api/data/import?stage=sft", content=b"{}\n",
                           headers={"content-type": "application/json"}).status_code == 415


def test_local_upload_cannot_claim_independent_license_verification(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        response = client.post(
            "/api/data/import?stage=sft&license_status=verified", content=_record(),
            headers={"content-type": "application/x-ndjson"},
        )
    assert response.status_code == 422
    assert "不能自助标记为已独立核验" in response.text


def test_import_api_enforces_actual_stream_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MAX_IMPORT_BYTES", 8)
    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/api/data/import?stage=sft", content=b"0123456789",
                               headers={"content-type": "application/x-ndjson"})
    assert response.status_code == 413
    assert not list((tmp_path / "data/processed/user-upload").glob("ds-*"))


def test_source_format_preview_is_ephemeral_and_registration_retains_raw(tmp_path):
    raw = (json.dumps({"id": "upstream", "conversations": [
        {"from": "human", "value": "1+1?"}, {"from": "gpt", "value": "2"},
    ]}) + "\n").encode()
    headers = {"content-type": "application/x-ndjson"}
    with TestClient(create_app(tmp_path)) as client:
        preview = client.post("/api/data/import/preview?stage=sft&source_format=sharegpt-conversations-v1",
                              content=raw, headers=headers)
        assert preview.status_code == 200, preview.text
        data = preview.json()
        assert data["valid"] is True
        assert data["provider_calls"] == 0 if "provider_calls" in data else True
        assert data["samples"][0]["messages"][1]["content"] == "2"
        assert not list((tmp_path / "data/processed/user-upload").glob("ds-*"))
        registered = client.post(
            "/api/data/import?stage=sft&source_format=sharegpt-conversations-v1&license_status=unknown",
            content=raw, headers=headers,
        )
        assert registered.status_code == 201, registered.text
        result = registered.json()
        dataset = tmp_path / result["manifest_path"]
        artifact = dataset.parent.parent
        assert (artifact / "raw/source.jsonl").read_bytes() == raw
        assert json.loads((artifact / "canonical/sft.jsonl").read_text(encoding="utf-8"))["metadata"]["source_record_id"] == "upstream"
        audit = validate_lineage_root(artifact, require_complete=True)
        assert audit["valid"], audit
        manifest = json.loads((artifact / "registration.json").read_text(encoding="utf-8"))
        assert manifest["source_format"] == "sharegpt-conversations-v1"


def test_source_format_preview_reports_safe_line_errors_and_never_registers(tmp_path):
    raw = b'{"instruction":"DO_NOT_ECHO","output":""}\n'
    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/api/data/import/preview?stage=sft&source_format=alpaca-instruction-v1",
                               content=raw, headers={"content-type": "application/x-ndjson"})
    assert response.status_code == 200
    assert response.json()["valid"] is False
    assert response.json()["errors"][0]["line"] == 1
    assert "DO_NOT_ECHO" not in response.text
    assert not list((tmp_path / "data/processed/user-upload").glob("ds-*"))


@pytest.mark.parametrize(("source_format", "row"), [
    ("alpaca-instruction-v1", {"instruction": "Question", "input": "context", "output": "Answer"}),
    ("messages-role-content-v1", {"messages": [{"role": "user", "content": "Question"},
                                                  {"role": "assistant", "content": "Answer"}]}),
    ("sharegpt-conversations-v1", {"conversations": [{"from": "human", "value": "Question"},
                                                        {"from": "gpt", "value": "Answer"}]}),
])
def test_external_sft_formats_api_walkthrough_preview_then_registration(tmp_path, source_format, row):
    raw = (json.dumps(row, ensure_ascii=False) + "\n").encode()
    query = f"stage=sft&source_format={source_format}&license_status=not_provided"
    headers = {"content-type": "application/x-ndjson"}
    with TestClient(create_app(tmp_path)) as client:
        preview = client.post(f"/api/data/import/preview?stage=sft&source_format={source_format}", content=raw, headers=headers)
        assert preview.status_code == 200 and preview.json()["valid"] is True
        response = client.post(f"/api/data/import?{query}", content=raw, headers=headers)
        assert response.status_code == 201, response.text
        payload = response.json()
    artifact = tmp_path / payload["manifest_path"]
    artifact = artifact.parent.parent
    assert (artifact / "raw/source.jsonl").read_bytes() == raw
    canonical = json.loads((artifact / "canonical/sft.jsonl").read_text(encoding="utf-8"))
    assert canonical["messages"][-1]["role"] == "assistant"
    audit = validate_lineage_root(artifact, require_complete=True)
    assert audit["valid"], audit


def test_objective_eval_import_preview_partial_registration_and_role_isolation(tmp_path):
    valid = {"id": "exam-1", "questionType": "multiple_choice", "question": "Pick A and F",
             "options": '["A","B","C","D","E","F"]', "correctAnswer": '["A","F"]'}
    missing = {"id": "exam-2", "questionType": "single_choice", "question": "missing answer",
               "options": '["yes","no"]'}
    raw = (json.dumps(valid, ensure_ascii=False) + "\n" + json.dumps(missing, ensure_ascii=False) + "\n").encode()
    headers = {"content-type": "application/x-ndjson"}
    with TestClient(create_app(tmp_path)) as client:
        preview = client.post("/api/data/import/preview?stage=eval&source_format=objective-choice-jsonl-v1",
                              content=raw, headers=headers)
        assert preview.status_code == 200, preview.text
        report = preview.json()
        assert (report["source_record_count"], report["output_record_count"], report["rejected_record_count"]) == (2, 1, 1)
        assert report["errors"][0]["line"] == 2
        assert report["errors"][0]["field"] == "correctAnswer"
        assert not list((tmp_path / "data/processed/user-upload").glob("ds-*"))
        registered = client.post(
            "/api/data/import?stage=eval&source_format=objective-choice-jsonl-v1&license_status=not_provided",
            content=raw, headers=headers,
        )
        assert registered.status_code == 201, registered.text
        payload = registered.json()
        manifest_path = tmp_path / payload["manifest_path"]
        manifest = __import__("onegpu_posttrain.data.manifest", fromlist=["load_manifest"]).load_manifest(manifest_path)
        assert manifest.stage == "eval" and manifest.metadata["dataset_role"] == "evaluation_only"
        assert manifest.metadata["source_record_count"] == 2
        assert manifest.metadata["converter_config"]["rejected_record_count"] == 1
        options = client.get("/api/training/workbench-options").json()
        registered_option = next(item for item in options["datasets"] if item["dataset_id"] == payload["dataset_id"])
        assert registered_option["stage"] == "eval" and registered_option["role"] == "evaluation_only"
        assert client.post("/api/data/import/preview?stage=eval&source_format=objective-choice-jsonl-v1",
                           content=(json.dumps(missing) + "\n").encode(), headers=headers).json()["valid"] is False
        rejected = client.post(
            "/api/data/import?stage=eval&source_format=objective-choice-jsonl-v1",
            content=(json.dumps(missing) + "\n").encode(), headers=headers,
        )
        assert rejected.status_code == 422
        assert "correctAnswer" in rejected.text
        assert len(list((tmp_path / "data/processed/user-upload").glob("ds-*"))) == 1
