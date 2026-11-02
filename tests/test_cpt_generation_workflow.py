import json
import os
import threading
from io import BytesIO

import yaml
from fastapi.testclient import TestClient

from onegpu_posttrain.data.dataset_import import register_jsonl
from onegpu_posttrain.data.decontaminate import BenchmarkIndex
from onegpu_posttrain.data.registry import DataSource
from onegpu_posttrain.data.targeted_stage_plan import (
    TargetedStagePlan,
    register_targeted_stage_plan,
)
from onegpu_posttrain.synthetic.providers import GenerationResponse, ProviderError
from onegpu_posttrain.synthetic.cpt_generation import validate_cpt_candidate_payload
from onegpu_posttrain.ui import cpt_generation_workflow as cpt_workflow
from onegpu_posttrain.ui import server as server_module
from onegpu_posttrain.ui.cpt_generation_workflow import (
    CPTGenerationJobs,
    execute_cpt_generation,
    preflight_cpt_generation,
    prepare_cpt_generation,
    preview_cpt_generation,
    register_cpt_generation,
)
from onegpu_posttrain.ui.server import create_app


class _Tokenizer:
    name = "test-tokenizer-v1"

    @staticmethod
    def count_text(text):
        return len(text.split())


class _Provider:
    calls = 0
    last_timeout = None

    def generate(self, profile, request):
        self.calls += 1
        self.last_timeout = request.timeout_seconds
        return GenerationResponse(text=json.dumps({
            "text": "This extended educational passage describes cellular membranes, organelles, and energy transfer in biology.",
            "subtopic": "cell structure",
        }), finish_reason="stop")


class _CapturingProvider(_Provider):
    prompt = ""

    def generate(self, profile, request):
        self.prompt = request.prompt
        return super().generate(profile, request)


class _UncertainProvider:
    def generate(self, profile, request):
        raise ProviderError("timeout", "provider result unknown", retryable=False, uncertain=True)


def test_cpt_candidate_canonicalizes_provider_expanded_subtopic():
    candidate, token_count, rejection = validate_cpt_candidate_payload(
        {"text": "A sufficiently detailed educational passage about recursive self-improvement systems.",
         "subtopic": "递归自进化（RSI）的核心机制与实现路径"},
        expected_subtopic="RSI(递归自进化)", benchmark_index=BenchmarkIndex([]),
        fuzzy_threshold=0.85, count_tokens=lambda text: len(text.split()), max_output_tokens=64,
    )

    assert rejection is None
    assert candidate is not None and candidate.subtopic == "RSI(递归自进化)"
    assert token_count == 9


def _workspace(tmp_path, monkeypatch):
    (tmp_path / "configs" / "synthetic" / "local").mkdir(parents=True)
    (tmp_path / "configs" / "model").mkdir(parents=True)
    (tmp_path / "data" / "examples" / "policy").mkdir(parents=True)
    profile = {
        "profile_id": "local-test", "adapter": "openai_compatible", "endpoint": "http://localhost/v1",
        "model": "test-model", "api_key_env": "CPT_GENERATION_TEST_KEY", "revision": "test-r1",
        "supports_json_object": True, "supported_parameters": ["temperature", "seed"],
    }
    (tmp_path / "configs" / "synthetic" / "local" / "provider-test.yaml").write_text(
        yaml.safe_dump(profile), encoding="utf-8")
    (tmp_path / "configs" / "model" / "qwen3-0.6b-base.yaml").write_text(
        yaml.safe_dump({"model": {"repo": "offline-fixture", "revision": "fixture-r1"}}), encoding="utf-8")
    (tmp_path / "data" / "examples" / "policy" / "benchmark.jsonl").write_text(
        json.dumps({"id": "protected-1", "question": "fixture benchmark problem"}) + "\n", encoding="utf-8")
    monkeypatch.setenv("CPT_GENERATION_TEST_KEY", "not-a-real-secret")
    return tmp_path


def test_cpt_ordinary_prepare_and_preflight_are_explicit_and_provider_read_only(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    prepared = prepare_cpt_generation(
        root, mode="ordinary", domain="biology", topic="cell structure", target_token_budget=50,
        request_budget=2, candidate_budget=2, max_output_tokens=64, profile_id="local-test",
    )
    assert prepared["provider_calls"] == 0
    assert prepared["network_used"] is False
    report = preflight_cpt_generation(root, prepared["bundle_id"], tokenizer=_Tokenizer())
    assert report["status"] == "preflight_ready"
    assert report["provider_calls"] == 0
    assert report["tokenizer_id"] == "test-tokenizer-v1"
    assert report["contamination_scope"] == "example_fixture_only"
    assert "不构成正式污染排除证明。" in report["limitations"][0]


def test_cpt_preflight_detects_provider_profile_mutation(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    prepared = prepare_cpt_generation(
        root, mode="ordinary", domain="biology", topic="cell structure", target_token_budget=50,
        request_budget=2, candidate_budget=2, max_output_tokens=64, profile_id="local-test",
    )
    profile = root / "configs" / "synthetic" / "local" / "provider-test.yaml"
    changed = yaml.safe_load(profile.read_text(encoding="utf-8"))
    changed["model"] = "another-model"
    profile.write_text(yaml.safe_dump(changed), encoding="utf-8")
    try:
        preflight_cpt_generation(root, prepared["bundle_id"], tokenizer=_Tokenizer())
    except ValueError as exc:
        assert "Provider 配置已变化" in str(exc)
    else:
        raise AssertionError("modified provider profile must invalidate the immutable generation plan")


def test_cpt_api_requires_local_token_and_explicit_start_confirmation(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    app = create_app(root, enable_synthetic_generation=True)
    with TestClient(app) as client:
        capabilities = client.get("/api/synthetic/sft/capabilities").json()
        token = capabilities["token"]
        headers = {"X-Local-Synthetic-Token": token}
        payload = {
            "mode": "ordinary", "domain": "biology", "topic": "cell structure",
            "target_token_budget": 50, "request_budget": 2, "candidate_budget": 2,
            "max_output_tokens": 64, "profile_id": "local-test", "timeout_seconds": 180,
        }
        unauthorized = client.post("/api/synthetic/cpt/prepare", json=payload)
        assert unauthorized.status_code == 403
        prepared = client.post("/api/synthetic/cpt/prepare", json=payload, headers=headers)
        assert prepared.status_code == 200
        assert prepared.json()["timeout_seconds"] == 180
        bundle_id = prepared.json()["bundle_id"]

        def fake_preflight(_workspace, _bundle_id):
            return {"status": "preflight_ready", "read_only": True, "provider_calls": 0,
                    "bundle_id": bundle_id}

        monkeypatch.setattr(server_module, "preflight_cpt_generation", fake_preflight)
        report = client.post("/api/synthetic/cpt/preflight", json={"bundle_id": bundle_id}, headers=headers)
        assert report.status_code == 200
        assert report.json()["provider_calls"] == 0
        no_confirm = client.post("/api/synthetic/cpt/start", json={"bundle_id": bundle_id}, headers=headers)
        assert no_confirm.status_code == 422
        no_confirmation = client.post("/api/synthetic/cpt/start",
                                      json={"bundle_id": bundle_id, "confirm_start": False}, headers=headers)
        assert no_confirmation.status_code == 422
        app.state.cpt_generation_jobs.start = lambda selected, *, confirm_start: {
            "execution_id": "cptgenrun-" + "4" * 20, "bundle_id": selected,
            "status": "queued", "confirmed": confirm_start,
        }
        started = client.post("/api/synthetic/cpt/start",
                              json={"bundle_id": bundle_id, "confirm_start": True}, headers=headers)
        assert started.status_code == 202
        assert started.json()["confirmed"] is True


def test_cpt_generation_preview_requires_review_and_registration_writes_hash_bound_reports(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    prepared = prepare_cpt_generation(
        root, mode="ordinary", domain="biology", topic="cell structure", target_token_budget=50,
        request_budget=1, candidate_budget=1, max_output_tokens=64, profile_id="local-test",
    )
    provider = _Provider()
    result = execute_cpt_generation(root, prepared["bundle_id"], execution_id="cptgenrun-" + "1" * 20,
                                    provider=provider, tokenizer=_Tokenizer())
    assert provider.calls == 1
    assert provider.last_timeout == 120
    assert result["registered"] is False
    assert result["factuality"] == "not_verified"
    assert result["contamination_scope"] == "example_fixture_only"
    preview = preview_cpt_generation(root, result)
    assert preview["registered"] is False
    assert preview["candidate_count"] == 1
    registration = register_cpt_generation(root, result)
    assert registration["registered"] is True
    assert registration["record_count"] == 1
    artifact = root / "data" / "processed" / "generated" / registration["dataset_id"]
    manifest_path = artifact / "manifests" / "cpt-model-generated-v1.yaml"
    from onegpu_posttrain.data.manifest import load_manifest
    manifest = load_manifest(manifest_path)
    assert manifest.hashes["canonical_records_sha256"] == registration["canonical_sha256"]
    assert manifest.metadata["factuality"] == "not_verified"
    assert manifest.metadata["provider_identity_status"] == "profile_declared_not_independently_verified"
    assert manifest.metadata["registered_model_association_status"] == "not_proven"
    assert manifest.metadata["registered_model_id"] is None
    assert manifest.metadata["registered_model_sha256"] is None
    assert manifest.metadata["provider_model"] == "test-model"
    assert manifest.metadata["provider_revision"] == "test-r1"
    assert manifest.processing_report.report_id == registration["processing_report_id"]


def test_cpt_execution_loads_local_provider_credential_before_provider_creation(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    prepared = prepare_cpt_generation(
        root, mode="ordinary", domain="biology", topic="cell structure", target_token_budget=50,
        request_budget=1, candidate_budget=1, max_output_tokens=64, profile_id="local-test",
    )
    monkeypatch.delenv("CPT_GENERATION_TEST_KEY")
    (root / ".env.local").write_text("CPT_GENERATION_TEST_KEY=test-placeholder\n", encoding="utf-8")
    credential_loads = []

    def load_credential(env_name, env_file):
        credential_loads.append((env_name, env_file))
        os.environ[env_name] = "not-a-real-secret"

    monkeypatch.setattr(cpt_workflow, "load_local_provider_credential", load_credential)
    provider = _Provider()
    monkeypatch.setattr(cpt_workflow, "OpenAICompatibleProvider", lambda: provider)
    result = execute_cpt_generation(
        root, prepared["bundle_id"], execution_id="cptgenrun-" + "5" * 20, tokenizer=_Tokenizer(),
    )

    assert credential_loads == [("CPT_GENERATION_TEST_KEY", root / ".env.local")]
    assert provider.calls == 1
    assert result["candidate_count"] == 1


def test_cpt_job_surfaces_sanitized_provider_error_details(tmp_path, monkeypatch):
    jobs = CPTGenerationJobs(tmp_path)
    execution_id = "cptgenrun-" + "6" * 20
    jobs._dir(execution_id).mkdir(parents=True)

    def missing_credential(*args, **kwargs):
        raise ProviderError("missing_credential", "本地 Provider 凭据不可用；请求未发送",
                            retryable=False, uncertain=False, timeout_seconds=120)

    monkeypatch.setattr(cpt_workflow, "execute_cpt_generation", missing_credential)
    state = {"execution_id": execution_id, "bundle_id": "cptgen-" + "7" * 20,
             "status": "running", "progress": {}}
    jobs._run(state, threading.Event())

    assert state["status"] == "failed"
    assert state["provider_error"]["category"] == "missing_credential"
    assert "missing_credential" in state["detail"]


def test_targeted_cpt_plan_pins_seed_dataset_hash_and_never_uses_holdout_rows(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    rows = [
        {"id": "train-1", "text": "Trusted train seed text for algebraic reasoning.", "source": "fixture",
         "metadata": {"data_role": "train_seed"}},
        {"id": "holdout-1", "text": "DO NOT USE FINAL HOLDOUT SECRET CONTENT.", "source": "fixture",
         "metadata": {"data_role": "final_holdout"}},
    ]
    body = "".join(json.dumps(row) + "\n" for row in rows).encode()
    source = DataSource(source_id="local-cpt-fixture", kind="local_upload", name="CPT fixture",
                        license_status="unknown", declared_by="user")
    registration = register_jsonl(BytesIO(body), workspace=root,
                                  output_root=root / "data" / "processed" / "uploads",
                                  stage="cpt", origin=source)
    plan = TargetedStagePlan.model_validate({
        "target": {"stage": "cpt", "domain": "mathematics", "subtopics": ["algebra"]},
        "training_seed": {"dataset_id": registration.dataset_id,
                          "content_sha256": registration.canonical_sha256, "role": "train_seed"},
        "starting_model": {"model_id": "base-fixture", "content_sha256": "a" * 64},
        "diagnostic_evaluation": {"dataset_id": "diagnostic", "dataset_sha256": "d" * 64,
                                  "protocol_version": "eval-v1", "config_sha256": "e" * 64,
                                  "role": "diagnostic_dev"},
        "final_evaluation": {"dataset_id": "holdout", "dataset_sha256": "b" * 64,
                             "protocol_version": "eval-v1", "config_sha256": "c" * 64,
                             "role": "final_holdout"},
        "provider_profile_id": "local-test",
        "budget": {"unit": "tokens", "expected_usable_count": 50, "max_candidates": 1,
                   "max_requests": 1, "max_output_tokens": 64},
        "training_budget": 1, "seed": 42,
    })
    registered_plan = register_targeted_stage_plan(root, plan)
    prepared = prepare_cpt_generation(
        root, mode="targeted", domain="mathematics", topic="algebra", target_token_budget=50,
        request_budget=1, candidate_budget=1, max_output_tokens=64, profile_id="local-test",
        plan_id=registered_plan["plan_id"], plan_version=registered_plan["plan_version"],
    )
    provider = _CapturingProvider()
    generated = execute_cpt_generation(root, prepared["bundle_id"], execution_id="cptgenrun-" + "2" * 20,
                                       provider=provider, tokenizer=_Tokenizer())
    assert generated["targeted_plan_id"] == registered_plan["plan_id"]
    assert generated["targeted_plan_version"] == registered_plan["plan_version"]
    assert generated["parent_dataset_refs"] == [{"dataset_id": registration.dataset_id,
                                                  "content_sha256": registration.canonical_sha256}]
    assert "Trusted train seed text" in provider.prompt
    assert "DO NOT USE FINAL HOLDOUT" not in provider.prompt

    bad_plan = plan.model_copy(update={"version": 2, "training_seed": plan.training_seed.model_copy(
        update={"content_sha256": "f" * 64})})
    register_targeted_stage_plan(root, bad_plan)
    try:
        prepare_cpt_generation(
            root, mode="targeted", domain="mathematics", topic="algebra", target_token_budget=50,
            request_budget=1, candidate_budget=1, max_output_tokens=64, profile_id="local-test",
            plan_id=bad_plan.plan_id, plan_version=2,
        )
    except ValueError as exc:
        assert "train_seed" in str(exc) or "计划" in str(exc) or "hash" in str(exc)
    else:
        raise AssertionError("targeted generation must reject a seed dataset hash mismatch")


def test_unknown_provider_result_blocks_retry_until_explicit_request_review(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    prepared = prepare_cpt_generation(
        root, mode="ordinary", domain="biology", topic="cell structure", target_token_budget=50,
        request_budget=1, candidate_budget=1, max_output_tokens=64, profile_id="local-test",
    )
    try:
        execute_cpt_generation(root, prepared["bundle_id"], execution_id="cptgenrun-" + "3" * 20,
                               provider=_UncertainProvider(), tokenizer=_Tokenizer())
    except ProviderError:
        pass
    else:
        raise AssertionError("uncertain provider result must propagate to the supervisor")
    jobs = CPTGenerationJobs(root)
    pending = jobs.request_review(prepared["bundle_id"])
    assert len(pending) == 1 and pending[0]["state"] == "uncertain"
    report = preflight_cpt_generation(root, prepared["bundle_id"], tokenizer=_Tokenizer())
    assert report["status"] == "blocked"
    assert report["provider_calls"] == 0
    acknowledged = jobs.request_review(prepared["bundle_id"], attempt_id=pending[0]["attempt_id"])
    assert acknowledged == []
    assert preflight_cpt_generation(root, prepared["bundle_id"], tokenizer=_Tokenizer())["status"] == "preflight_ready"
