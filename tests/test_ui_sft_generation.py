from __future__ import annotations

import json
import hashlib
import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from test_sft_generation_workflow import _responses
from test_sft_workflow_bundle import _inputs

from onegpu_posttrain.synthetic.providers import FakeProvider
from onegpu_posttrain.synthetic.sft_workflow import execute_sft_generation
from onegpu_posttrain.synthetic.ordinary_bundle import load_ordinary_sft_bundle
from onegpu_posttrain.synthetic.workflow_bundle import register_sft_workflow_bundle
from onegpu_posttrain.ui.server import create_app
from onegpu_posttrain.ui.sft_generation_jobs import SFTGenerationJobs
from onegpu_posttrain.ui.sft_generation_setup import discover_sft_generation_setup


def _setup_workspace(root):
    manifest = root / "data/processed/upload/manifests/sft-v1.yaml"
    manifest.parent.mkdir(parents=True)
    canonical = manifest.parent.parent / "canonical/sft.jsonl"
    canonical.parent.mkdir()
    content = json.dumps({
        "id": "record-1", "raw_problem": "What is 2 + 2?",
        "messages": [{"role": "user", "content": "What is 2 + 2?"},
                     {"role": "assistant", "content": "Final answer: 4"}],
        "prompt_template_id": "fixture-source-template",
        "metadata": {"problem_id": "problem-1", "reference": "4"},
    }) + "\n"
    canonical.write_bytes(content.encode("utf-8"))
    digest = hashlib.sha256(canonical.read_bytes()).hexdigest()
    manifest.write_text(
        f"dataset_id: ds-sft-fixture\nstage: sft\nsample_count: 1\nhashes:\n  canonical_records_sha256: {digest}\n",
        encoding="utf-8")
    examples = root / "configs/synthetic/examples"
    examples.mkdir(parents=True)
    (examples / "provider.fake.yaml").write_text(
        "profile_id: fake-local\nadapter: fake\nmodel: deterministic-fixture\n", encoding="utf-8")
    (examples / "provider.local.yaml").write_text(
        "profile_id: remote-local\nadapter: openai_compatible\nendpoint: https://example.test/v1/chat/completions\n"
        "model: fixture-remote\napi_key_env: SFT_TEST_SECRET\nsupported_parameters: [temperature, max_output_tokens]\n",
        encoding="utf-8")
    (examples / "prompt.sft-math.yaml").write_text(
        "template_id: math-sft\nversion: '1'\ntemplate: 'Solve: {question}'\nallowed_variables: [question]\n",
        encoding="utf-8")
    (examples / "output.sft-candidate.yaml").write_text(
        "protocol_id: sft-candidate\nversion: '1'\nresponse_format: json_object\nrequired_fields: [problem_id, solution, final_answer]\n",
        encoding="utf-8")


def test_generation_api_disabled_by_default_and_requires_local_token(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/synthetic/sft/capabilities").json()["enabled"] is False
        assert client.post("/api/synthetic/sft/preflight", json={}).status_code == 403
        assert client.post("/api/synthetic/sft/start", json={}).status_code == 403


def test_legacy_sft_execution_and_manifest_remain_readable_without_rewriting(tmp_path, monkeypatch):
    _setup_workspace(tmp_path)
    execution_id = "sftgen-" + "a" * 20
    state_path = tmp_path / "runs/synthetic-workflow" / execution_id / "state.json"
    state_path.parent.mkdir(parents=True)
    legacy = {"execution_id": execution_id, "bundle_id": "plan-targeted-legacy",
              "status": "completed", "result": {"training_dataset_id": "ds-sft-fixture"}}
    state_path.write_text(json.dumps(legacy), encoding="utf-8")
    manifest_path = tmp_path / "data/processed/upload/manifests/sft-v1.yaml"
    original_state, original_manifest = state_path.read_bytes(), manifest_path.read_bytes()
    calls = []
    monkeypatch.setattr(FakeProvider, "generate", lambda *args, **kwargs: calls.append(1))

    with TestClient(create_app(tmp_path, enable_synthetic_generation=True)) as client:
        token = client.get("/api/synthetic/sft/capabilities").json()["token"]
        response = client.get(f"/api/synthetic/sft/runs/{execution_id}", headers={"X-Local-Synthetic-Token": token})
        assert response.status_code == 200
        assert all(response.json()[key] == value for key, value in legacy.items())
        assert response.json()["events"] == []
        stage_response = client.get(f"/api/synthetic/stage/runs/{execution_id}",
                                    headers={"X-Local-Synthetic-Token": token})
        assert stage_response.status_code == 200
        assert stage_response.json()["stage"] == "sft"
        assert stage_response.json()["source_mode"] is None  # legacy did not record a mode
        assert client.get("/api/data").status_code == 200
        assert client.get("/api/synthetic/sft/readiness", headers={"X-Local-Synthetic-Token": token}).status_code == 200
    assert state_path.read_bytes() == original_state
    assert manifest_path.read_bytes() == original_manifest
    assert b"lineage" not in original_manifest
    assert calls == []


def test_stage_preflight_is_read_only_and_checks_bundle_credential(tmp_path, monkeypatch):
    _setup_workspace(tmp_path)
    monkeypatch.setenv("SFT_TEST_SECRET", "secret-not-in-response")
    calls = []
    monkeypatch.setattr(FakeProvider, "generate", lambda *args, **kwargs: calls.append(1))
    from onegpu_posttrain.synthetic.providers import OpenAICompatibleProvider
    monkeypatch.setattr(OpenAICompatibleProvider, "generate", lambda *args, **kwargs: calls.append(1))
    with TestClient(create_app(tmp_path, enable_synthetic_generation=True)) as client:
        token = client.get("/api/synthetic/sft/capabilities").json()["token"]
        headers = {"X-Local-Synthetic-Token": token}
        catalog = client.get("/api/targeted/stages").json()
        assert catalog["read_only"] is True
        assert all(mode["provider_calls"] == 0
                   for stage_item in catalog["stages"]
                   for mode in stage_item["generation_modes"].values())
        assert calls == []
        created = client.post("/api/synthetic/sft/prepare", headers=headers, json={
            "mode": "prompt_only", "topic": "algebra", "skill": "equations",
            "difficulty": "introductory", "target_count": 1, "profile_id": "remote-local",
        })
        assert created.status_code == 200, created.text
        payload = {"stage": "sft", "mode": "ordinary", "bundle_id": created.json()["bundle_id"],
                   "usable_target": 1, "max_requests": 1, "target_candidates": 1}
        result = client.post("/api/synthetic/stage/preflight", headers=headers, json=payload)
        assert result.status_code == 200, result.text
        assert result.json()["status"] == "preflight_ready"
        assert result.json()["provider_calls"] == 0
        assert "secret-not-in-response" not in result.text
        monkeypatch.delenv("SFT_TEST_SECRET")
        result = client.post("/api/synthetic/stage/preflight", headers=headers, json=payload)
        assert result.json()["status"] == "blocked"
        assert any("凭据" in item for item in result.json()["blockers"])
        for stage in ("cpt", "preference", "grpo"):
            for mode in ("ordinary", "targeted"):
                result = client.post("/api/synthetic/stage/preflight", headers=headers,
                                     json={"stage": stage, "mode": mode})
                assert result.status_code == 200
                assert result.json()["status"] == "blocked"
                assert result.json()["provider_calls"] == 0
                assert result.json()["adapter_available"] is (stage in {"cpt", "grpo"})
                assert result.json()["provider_configured"] is False
                expected_protocol = {
                    "cpt": "available",
                    "grpo": "prompt_only_not_trainable_without_reference",
                    "preference": "not_connected_to_generation_dispatch",
                }[stage]
                assert result.json()["candidate_protocol"] == expected_protocol
                assert result.json()["domain_adapter"]
                if stage == "grpo":
                    handoff = result.json()["training_handoff"]
                    assert handoff["status"] == "blocked"
                    assert handoff["reference"]["status"] == "not_provided"
                    assert handoff["reward"]["status"] == "missing"
        plugin = tmp_path / "plugins" / "grpo_rewards.py"
        plugin.parent.mkdir(parents=True, exist_ok=True)
        plugin.write_text("# fixture, not validated\n", encoding="utf-8")
        result = client.post("/api/synthetic/stage/preflight", headers=headers,
                             json={"stage": "grpo", "mode": "ordinary"}).json()
        assert result["training_handoff"]["reward"]["status"] == "present_unverified"
        assert result["training_handoff"]["reference"]["status"] == "not_provided"
        assert client.post("/api/synthetic/stage/preflight", headers=headers,
                           json=payload | {"mode": "targeted"}).status_code == 422
        assert client.post("/api/synthetic/stage/preflight", headers=headers,
                           json=payload | {"stage": []}).status_code == 422
        assert client.post("/api/synthetic/stage/preflight", headers=headers,
                           json=payload | {"max_requests": True}).status_code == 422
        assert client.post("/api/synthetic/stage/preflight", headers=headers,
                           json=payload | {"provider_key": "ignored"}).status_code == 422
        assert client.post("/api/synthetic/stage/preflight", json=payload).status_code == 403
        assert calls == []


def test_stage_start_dispatch_requires_confirmation_and_supported_preflight(tmp_path, monkeypatch):
    _setup_workspace(tmp_path)
    monkeypatch.setenv("SFT_TEST_SECRET", "test-secret")
    app = create_app(tmp_path, enable_synthetic_generation=True)
    calls = []
    monkeypatch.setattr(app.state.sft_generation_jobs, "start",
                        lambda bundle_id, **kwargs: calls.append((bundle_id, kwargs)) or {"execution_id": "fixture"})
    with TestClient(app) as client:
        token = client.get("/api/synthetic/sft/capabilities").json()["token"]
        headers = {"X-Local-Synthetic-Token": token}
        prepared = client.post("/api/synthetic/sft/prepare", headers=headers, json={
            "mode": "prompt_only", "topic": "algebra", "skill": "equations",
            "difficulty": "introductory", "target_count": 1, "profile_id": "remote-local",
        }).json()
        payload = {"stage": "sft", "mode": "ordinary", "bundle_id": prepared["bundle_id"],
                   "usable_target": 1, "max_requests": 1, "target_candidates": 1}
        assert client.post("/api/synthetic/stage/start", headers=headers, json=payload).status_code == 422
        assert client.post("/api/synthetic/stage/start", headers=headers,
                           json={"stage": "grpo", "mode": "ordinary", "confirm_start": True}).status_code == 409
        assert calls == []
        started = client.post("/api/synthetic/stage/start", headers=headers, json=payload | {"confirm_start": True})
        assert started.status_code == 202
        assert calls[0][0] == prepared["bundle_id"] and calls[0][1]["max_requests"] == 1


def test_readiness_discovery_is_read_only_and_explains_missing_evidence(tmp_path, monkeypatch):
    _setup_workspace(tmp_path)
    monkeypatch.setenv("SFT_TEST_SECRET", "never-return-this")
    report = discover_sft_generation_setup(tmp_path)
    assert report["provider_calls"] == 0
    assert report["summary"] == {
        "dataset_count": 1,
        "ready_count": 0,
        "ordinary_ready_count": 1,
        "targeted_ready_count": 0,
        "scenario_report_count": 0,
    }
    assert report["datasets"][0]["eligible"] is True
    assert report["readiness"][0]["eligible"] is False
    assert report["readiness"][0]["ordinary_sft"]["eligible"] is True
    assert report["readiness"][0]["ordinary_sft"]["sample_count"] == 1
    assert report["readiness"][0]["targeted"]["eligible"] is False
    assert any("角色登记" in item for item in report["readiness"][0]["missing_reasons"])
    assert "never-return-this" not in json.dumps(report)
    remote_profile = next(item for item in report["profiles"] if item["profile_id"] == "remote-local")
    assert remote_profile["credential_env_configured"] is True

    manifest = tmp_path / "data/processed/upload/manifests/sft-v1.yaml"
    manifest.write_text(
        "dataset_id: ds-sft-fixture\nstage: sft\nsample_count: 1\nhashes:\n  canonical_records_sha256: "
        + "'" + "0" * 64 + "'\n", encoding="utf-8")
    stale = discover_sft_generation_setup(tmp_path)
    assert stale["datasets"][0]["eligible"] is False
    assert stale["readiness"][0]["ordinary_sft"]["eligible"] is False
    assert "摘要与 manifest 不一致" in stale["datasets"][0]["missing_reasons"][0]


def test_readiness_detects_local_credential_without_returning_secret(tmp_path, monkeypatch):
    _setup_workspace(tmp_path)
    monkeypatch.delenv("SFT_TEST_SECRET", raising=False)
    (tmp_path / ".env.local").write_text("SFT_TEST_SECRET=secret-not-to-return\n", encoding="utf-8")

    report = discover_sft_generation_setup(tmp_path)
    profile = next(item for item in report["profiles"] if item["profile_id"] == "remote-local")

    assert profile["credential_env_configured"] is True
    assert "secret-not-to-return" not in json.dumps(report)


@pytest.mark.parametrize(("content", "expected"), [
    ("SFT_TEST_SECRET=\"\"\n", False),
    ("SFT_TEST_SECRET='unterminated\n", False),
    ("SFT_TEST_SECRET=secret\nOTHER=value\n", False),
    ("SFT_TEST_SECRET=first\nSFT_TEST_SECRET=second\n", False),
    ("SFT_TEST_SECRET=secret\ninvalid line\n", False),
    ("export SFT_TEST_SECRET='secret'\n", True),
    ("SFT_TEST_SECRET=\nSFT_TEST_SECRET=secret\n", True),
])
def test_readonly_credential_check_matches_local_file_rules(tmp_path, monkeypatch, content, expected):
    from onegpu_posttrain.ui.sft_generation_setup import _credential_available

    monkeypatch.delenv("SFT_TEST_SECRET", raising=False)
    (tmp_path / ".env.local").write_text(content, encoding="utf-8")
    assert _credential_available(tmp_path, "SFT_TEST_SECRET") is expected


def test_prepare_ordinary_sft_bundle_uses_only_ids_and_needs_no_diagnostic_evidence(tmp_path, monkeypatch):
    _setup_workspace(tmp_path)
    monkeypatch.setenv("SFT_TEST_SECRET", "do-not-return")

    with TestClient(create_app(tmp_path, enable_synthetic_generation=True)) as client:
        token = client.get("/api/synthetic/sft/capabilities").json()["token"]
        headers = {"X-Local-Synthetic-Token": token}
        response = client.post("/api/synthetic/sft/prepare", json={
            "dataset_id": "ds-sft-fixture", "target_count": 1,
        }, headers=headers)
        assert response.status_code == 200, response.text
        prepared = response.json()
        assert prepared["mode"] == "ordinary_sft"
        assert prepared["status"] == "prepared_not_executed"
        assert prepared["provider_calls"] == 0
        assert prepared["target_count"] == 1
        assert "do-not-return" not in response.text
        bundle, files = load_ordinary_sft_bundle(tmp_path, prepared["bundle_id"])
        assert bundle.bundle_id == prepared["bundle_id"]
        assert set(files) == {"source_manifest", "source_data", "profile", "template", "protocol", "policy", "plan"}
        repeated = client.post("/api/synthetic/sft/prepare", json={
            "dataset_id": "ds-sft-fixture", "target_count": 1,
        }, headers=headers)
        assert repeated.status_code == 200
        assert repeated.json()["bundle_id"] == prepared["bundle_id"]
        changed = client.post("/api/synthetic/sft/prepare", json={
            "dataset_id": "ds-sft-fixture", "target_count": 2,
        }, headers=headers)
        assert changed.status_code == 200
        assert changed.json()["bundle_id"] != prepared["bundle_id"]
        assert client.post("/api/synthetic/sft/prepare", json={
            "dataset_id": "ds-sft-fixture", "target_count": 1, "path": "../../outside",
        }, headers=headers).status_code == 422
        canonical = tmp_path / "data/processed/upload/canonical/sft.jsonl"
        canonical.write_text('{"id":"tampered"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="input changed"):
            load_ordinary_sft_bundle(tmp_path, prepared["bundle_id"])


def test_ordinary_sft_stops_after_uncertain_request(tmp_path, monkeypatch):
    from onegpu_posttrain.data.tokenization import WhitespaceTokenizer
    from onegpu_posttrain.synthetic.providers import ProviderError
    from onegpu_posttrain.synthetic.ordinary_workflow import execute_ordinary_sft_generation
    import onegpu_posttrain.synthetic.ordinary_workflow as workflow

    _setup_workspace(tmp_path)
    monkeypatch.setenv("SFT_TEST_SECRET", "test-secret")
    canonical = tmp_path / "data/processed/upload/canonical/sft.jsonl"
    row = json.loads(canonical.read_text(encoding="utf-8"))
    second = dict(row, id="record-2")
    canonical.write_text(json.dumps(row) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
    digest = hashlib.sha256(canonical.read_bytes()).hexdigest()
    manifest = tmp_path / "data/processed/upload/manifests/sft-v1.yaml"
    manifest.write_text(f"dataset_id: ds-sft-fixture\nstage: sft\nsample_count: 2\nhashes:\n"
                        f"  canonical_records_sha256: {digest}\n", encoding="utf-8")
    monkeypatch.setattr(workflow, "_tokenizer", lambda *_: WhitespaceTokenizer())

    class Provider:
        calls = 0
        def generate(self, profile, request):
            self.calls += 1
            raise ProviderError("timeout", "private detail", retryable=True, uncertain=True)

    with TestClient(create_app(tmp_path, enable_synthetic_generation=True)) as client:
        token = client.get("/api/synthetic/sft/capabilities").json()["token"]
        prepared = client.post("/api/synthetic/sft/prepare", json={"dataset_id": "ds-sft-fixture", "target_count": 2},
                               headers={"X-Local-Synthetic-Token": token})
        assert prepared.status_code == 200, prepared.text
        provider = Provider()
        result = execute_ordinary_sft_generation(tmp_path, prepared.json()["bundle_id"],
                                                 execution_id="sftgen-" + "e" * 20, provider=provider)
        assert provider.calls == 1
        assert result["stop_reason"] == "uncertain_request_requires_explicit_review"
        assert "private detail" not in json.dumps(result)
        with pytest.raises(ValueError, match="结果不确定"):
            execute_ordinary_sft_generation(tmp_path, prepared.json()["bundle_id"],
                                            execution_id="sftgen-" + "f" * 20, provider=provider)
        assert provider.calls == 1
        bundle_id = prepared.json()["bundle_id"]
        headers = {"X-Local-Synthetic-Token": token}
        url = f"/api/synthetic/sft/request-review/{bundle_id}"
        pending = client.get(url, headers=headers).json()["unresolved"]
        assert len(pending) == 1 and pending[0]["state"] == "uncertain"
        attempt = pending[0]["attempt_id"]
        assert client.post(url, headers=headers, json={"attempt_id": attempt,
                           "confirm_billing_risk": False}).status_code == 422
        reviewed = client.post(url, headers=headers, json={"attempt_id": attempt,
                               "confirm_billing_risk": True})
        assert reviewed.status_code == 200 and reviewed.json()["unresolved"] == []
        assert "仍未知" in reviewed.json()["note"]
        assert provider.calls == 1
        ledger = (tmp_path / "artifacts/synthetic" / bundle_id / "generation-requests.jsonl").read_text(encoding="utf-8")
        assert "reviewed_unknown" in ledger and "private detail" not in ledger


def test_ordinary_sft_fake_execution_registers_dataset_with_source_lineage(tmp_path, monkeypatch):
    import time
    from onegpu_posttrain.data.manifest import load_manifest
    from onegpu_posttrain.data.tokenization import WhitespaceTokenizer
    from onegpu_posttrain.synthetic.providers import GenerationResponse
    from onegpu_posttrain.synthetic.ordinary_workflow import execute_ordinary_sft_generation
    from onegpu_posttrain.ui.server import create_app

    _setup_workspace(tmp_path)
    monkeypatch.setenv("SFT_TEST_SECRET", "never-return")
    import onegpu_posttrain.synthetic.ordinary_workflow as ordinary_module
    monkeypatch.setattr(ordinary_module, "_tokenizer", lambda root, policy: WhitespaceTokenizer())

    class Provider:
        calls = 0

        def generate(self, profile, request):
            self.calls += 1
            return GenerationResponse(
                text=json.dumps({"problem_id": "problem-1", "solution": "Final answer: 4", "final_answer": "4"}),
                finish_reason="stop",
            )

    provider = Provider()
    import onegpu_posttrain.ui.sft_generation_jobs as jobs_module
    monkeypatch.setattr(jobs_module, "execute_ordinary_sft_generation",
                        lambda root, bundle_id, **kwargs: execute_ordinary_sft_generation(
                            root, bundle_id, provider=provider, **kwargs))
    with TestClient(create_app(tmp_path, enable_synthetic_generation=True)) as client:
        token = client.get("/api/synthetic/sft/capabilities").json()["token"]
        headers = {"X-Local-Synthetic-Token": token}
        prepared = client.post("/api/synthetic/sft/prepare", json={
            "dataset_id": "ds-sft-fixture", "target_count": 1,
        }, headers=headers)
        assert prepared.status_code == 200
        bundle_id = prepared.json()["bundle_id"]
        assert prepared.json()["provider_calls"] == 0
        preflight_payload = {"bundle_id": bundle_id, "usable_target": 1,
                             "max_requests": 2, "target_candidates": 1}
        preflight = client.post("/api/synthetic/sft/preflight", json=preflight_payload, headers=headers)
        assert preflight.status_code == 200
        assert preflight.json()["provider_calls"] == 0
        assert preflight.json()["eligible_source_count"] == 1
        assert provider.calls == 0
        started = client.post("/api/synthetic/sft/start", json=preflight_payload | {"confirm_start": True},
                              headers=headers)
        assert started.status_code == 202
        execution_id = started.json()["execution_id"]
        for _ in range(100):
            state = client.get(f"/api/synthetic/sft/runs/{execution_id}", headers=headers).json()
            if state["status"] in {"completed", "partial", "failed", "cancelled"}:
                break
            time.sleep(0.02)
        assert state["status"] == "completed"
        assert state["mode"] == "ordinary_sft"
        assert provider.calls == 1
        reviewed = client.get(f"/api/synthetic/sft/runs/{execution_id}/preview?limit=1", headers=headers)
        assert reviewed.status_code == 200
        assert reviewed.json()["count"] == 1
        assert reviewed.json()["semantic_correctness"] == "not_independently_verified"
        registered_response = client.post(f"/api/synthetic/sft/runs/{execution_id}/register",
                                          json={"confirm_reviewed": True}, headers=headers)
        assert registered_response.status_code == 200
        registration = registered_response.json()["registration"]
        catalog = client.get("/api/data")
        assert catalog.status_code == 200
        registered_ids = {dataset["dataset_id"] for build in catalog.json()["builds"]
                          for dataset in build["datasets"]}
        assert registration["dataset_id"] in registered_ids
        result = state["result"]
    manifest_path = tmp_path / registration["manifest_path"]
    manifest = load_manifest(manifest_path)
    assert manifest.lineage.parents == ["ds-sft-fixture"]
    assert manifest.lineage.source_evidence["generation_execution_id"] == execution_id
    assert manifest.lineage.source_evidence["generation_mode"] == "ordinary_sft"
    assert manifest.lineage.source_evidence["bundle_id"] == prepared.json()["bundle_id"]
    assert manifest.lineage.source_evidence["requested_target_count"] == 1
    assert len(manifest.lineage.source_evidence["provider_profile_sha256"]) == 64
    assert "never-return" not in manifest_path.read_text(encoding="utf-8")
    artifact_root = manifest_path.parent.parent
    build_lineage = json.loads((artifact_root / "runs" / manifest.lineage.produced_by_run_id
                                / "lineage.json").read_text(encoding="utf-8"))
    input_ids = {item["id"] for item in build_lineage["inputs"]}
    output_ids = {item["id"] for item in build_lineage["outputs"]}
    assert {"ds-sft-fixture", execution_id}.issubset(input_ids)
    assert manifest.dataset_id in output_ids


def test_readiness_api_uses_local_generation_token_and_never_accepts_paths(tmp_path):
    _setup_workspace(tmp_path)
    with TestClient(create_app(tmp_path, enable_synthetic_generation=True)) as client:
        assert client.get("/api/synthetic/sft/readiness").status_code == 403
        token = client.get("/api/synthetic/sft/capabilities").json()["token"]
        headers = {"X-Local-Synthetic-Token": token}
        response = client.get("/api/synthetic/sft/readiness", headers=headers)
        assert response.status_code == 200, response.text
        assert response.json()["provider_calls"] == 0
        assert client.post("/api/synthetic/sft/readiness", json={"path": "../../outside"},
                           headers=headers).status_code == 405


def test_generation_api_is_loopback_same_origin_and_typed(tmp_path):
    with pytest.raises(ValueError, match="loopback"):
        create_app(tmp_path, enable_synthetic_generation=True, allow_remote=True)
    with TestClient(create_app(tmp_path, enable_synthetic_generation=True)) as client:
        token = client.get("/api/synthetic/sft/capabilities").json()["token"]
        headers = {"X-Local-Synthetic-Token": token}
        payload = {"bundle_id": "../outside", "usable_target": 1, "max_requests": 1,
                   "target_candidates": 1}
        assert client.post("/api/synthetic/sft/preflight", json=payload).status_code == 403
        assert client.post("/api/synthetic/sft/preflight", json=payload, headers=headers).status_code == 422
        assert client.post("/api/synthetic/sft/preflight", json=payload, headers=headers | {
            "Origin": "https://evil.example"}).status_code == 403
        assert client.post("/api/synthetic/sft/start", json=payload, headers=headers).status_code == 422
        assert client.get("/api/synthetic/sft/runs/../outside", headers=headers).status_code in {404, 405}


def test_generation_api_explicit_start_persists_status_and_recovers_after_refresh(tmp_path, monkeypatch):
    workspace, inputs = _inputs(tmp_path)
    bundle = register_sft_workflow_bundle(workspace, inputs=inputs)
    import onegpu_posttrain.ui.sft_generation_jobs as manager_module
    monkeypatch.setattr(manager_module, "execute_sft_generation", lambda root, bundle_id, **kwargs:
                        execute_sft_generation(root, bundle_id, provider=FakeProvider(_responses(inputs)), **kwargs))
    with TestClient(create_app(workspace, enable_synthetic_generation=True,
                               enable_experiment_launcher=True)) as client:
        caps = client.get("/api/synthetic/sft/capabilities").json()
        token = caps["token"]
        assert caps["bundles"][0]["bundle_id"] == bundle.bundle_id
        headers = {"X-Local-Synthetic-Token": token}
        payload = {"bundle_id": bundle.bundle_id, "usable_target": 1, "max_requests": 1,
                   "target_candidates": 1}
        preflight = client.post("/api/synthetic/sft/preflight", json=payload, headers=headers)
        assert preflight.status_code == 200
        assert preflight.json()["provider_calls"] == 0
        assert preflight.json()["network_used"] is False

        assert client.post("/api/synthetic/sft/start", json=payload | {"confirm_start": False},
                           headers=headers).status_code == 422
        started = client.post("/api/synthetic/sft/start", json=payload | {"confirm_start": True},
                              headers=headers)
        assert started.status_code == 202
        execution_id = started.json()["execution_id"]
        state = None
        for _ in range(100):
            response = client.get(f"/api/synthetic/sft/runs/{execution_id}", headers=headers)
            assert response.status_code == 200
            state = response.json()
            if state["status"] in {"completed", "partial", "failed", "cancelled"}:
                break
            time.sleep(0.05)
        assert state is not None and state["status"] in {"partial", "completed"}
        assert state["result"]["provider_calls"] == 1
        assert state["result"]["network_used"] is False
        assert state["events"]
        response_body = json.dumps(state)
        assert "api_key" not in response_body

        reviewed = client.get(f"/api/synthetic/sft/runs/{execution_id}/preview?limit=2", headers=headers)
        assert reviewed.status_code == 200, reviewed.text
        assert reviewed.json()["records"]
        assert reviewed.json()["semantic_correctness"] == "not_independently_verified"

        register = client.post(
            f"/api/synthetic/sft/runs/{execution_id}/register",
            json={"confirm_reviewed": True}, headers=headers,
        )
        assert register.status_code == 200, register.text
        dataset_id = register.json()["registration"]["dataset_id"]
        preview = client.get(f"/api/data/datasets/{dataset_id}/preview")
        assert preview.status_code == 200, preview.text
        assert preview.json()["dataset_id"] == dataset_id
        catalog = client.get("/api/data").json()
        catalog_rows = [item for build in catalog["builds"] for item in build["datasets"]]
        catalog_dataset = next(item for item in catalog_rows if item["dataset_id"] == dataset_id)
        assert catalog_dataset["content_sha256"] == register.json()["registration"]["canonical_sha256"]
        launcher_caps = client.get("/api/launcher/capabilities").json()
        registered = {item["dataset_id"]: item for item in launcher_caps["datasets"]}
        assert registered[dataset_id]["count"] == register.json()["registration"]["count"]
        assert registered[dataset_id]["quality_status"] == "report_only"


def test_job_manager_recovers_interrupted_jobs_without_restarting(tmp_path):
    jobs = SFTGenerationJobs(tmp_path)
    job_id = "sftgen-" + "a" * 20
    job_dir = jobs.root / job_id
    job_dir.mkdir()
    (job_dir / "state.json").write_text(json.dumps({
        "execution_id": job_id, "status": "running", "phase": "generation",
    }), encoding="utf-8")

    recovered = SFTGenerationJobs(tmp_path).get(job_id)
    assert recovered["status"] == "interrupted"
    assert "未自动重启" in recovered["detail"]


def test_job_manager_cancellation_waits_for_current_provider_call(tmp_path, monkeypatch):
    import onegpu_posttrain.ui.sft_generation_jobs as manager_module

    entered = threading.Event()
    release = threading.Event()

    def fake_execute(workspace, bundle_id, **kwargs):
        entered.set()
        assert release.wait(2)
        return {"status": "cancelled", "provider_calls": 1, "network_used": False}

    monkeypatch.setattr(manager_module, "execute_sft_generation", fake_execute)
    jobs = SFTGenerationJobs(tmp_path)
    jobs.preflight = lambda *args, **kwargs: {"status": "preflight_ready"}
    started = jobs.start("plan-targeted-" + "b" * 20, usable_target=1, max_requests=1,
                         target_candidates=1)
    assert entered.wait(2)
    cancelling = jobs.cancel(started["execution_id"])
    assert cancelling["cancel_requested"] is True
    release.set()
    for _ in range(100):
        state = jobs.get(started["execution_id"])
        if state["status"] == "cancelled":
            break
        time.sleep(0.01)
    assert state["status"] == "cancelled"


def test_job_manager_never_persists_exception_text(tmp_path, monkeypatch):
    import onegpu_posttrain.ui.sft_generation_jobs as manager_module

    def fail_with_secret(*args, **kwargs):
        raise RuntimeError("SECRET_SENTINEL")

    monkeypatch.setattr(manager_module, "execute_sft_generation", fail_with_secret)
    jobs = SFTGenerationJobs(tmp_path)
    jobs.preflight = lambda *args, **kwargs: {"status": "preflight_ready"}
    started = jobs.start("plan-targeted-" + "c" * 20, usable_target=1, max_requests=1,
                         target_candidates=1)
    for _ in range(100):
        state = jobs.get(started["execution_id"])
        if state["status"] == "failed":
            break
        time.sleep(0.01)
    assert state["status"] == "failed"
    assert "SECRET_SENTINEL" not in json.dumps(state)
