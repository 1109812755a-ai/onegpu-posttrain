from __future__ import annotations

import json
import threading
import time
from io import BytesIO

import yaml
from fastapi.testclient import TestClient

from onegpu_posttrain.data.dataset_import import register_jsonl
from onegpu_posttrain.data.registry import DataSource
from onegpu_posttrain.data.targeted_grpo import ExecutableRewardContract
from onegpu_posttrain.data.targeted_stage_plan import (
    TargetedStagePlan,
    register_targeted_stage_plan,
)
from onegpu_posttrain.synthetic.providers import GenerationResponse
from onegpu_posttrain.ui.grpo_generation_jobs import GRPOGenerationJobs
from onegpu_posttrain.ui.server import create_app


class FakeProvider:
    def __init__(self):
        self.calls = 0

    def generate(self, profile, request):
        self.calls += 1
        return GenerationResponse(text=json.dumps({"raw_problem": "A generated algebra prompt."}), finish_reason="stop")


def _profile(root):
    path = root / "configs" / "synthetic" / "local" / "provider-grpo-test.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"profile_id": "grpo-test", "adapter": "fake", "model": "fixture"}),
                    encoding="utf-8")
    return path


def _seed_and_plan(root):
    rows = [
        {"id": "seed-1", "raw_problem": "Solve 2x=4.",
         "messages": [{"role": "user", "content": "Solve 2x=4."}, {"role": "assistant", "content": "x=2"}],
         "prompt_template_id": "fixture", "metadata": {"data_role": "train_seed", "problem_id": "algebra-1"}},
        {"id": "holdout-1", "raw_problem": "SECRET HOLDOUT.",
         "messages": [{"role": "user", "content": "SECRET HOLDOUT."}, {"role": "assistant", "content": "x=9"}],
         "prompt_template_id": "fixture", "metadata": {"data_role": "final_holdout", "problem_id": "holdout-1"}},
    ]
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    registered = register_jsonl(BytesIO(payload), workspace=root,
        output_root=root / "data" / "processed" / "fixture", stage="sft",
        origin=DataSource(source_id="source-fixture", kind="local_upload", name="fixture",
                          license_status="user_declared_unverified", declared_by="user"))
    contract = ExecutableRewardContract.create()
    plan = TargetedStagePlan.model_validate({
        "target": {"stage": "grpo", "scenario_id": "algebra",
                   "reward_contract": {"evidence_id": contract.contract_id,
                                       "content_sha256": contract.protocol_sha256,
                                       "evidence_kind": "reward_contract"}},
        "training_seed": {"dataset_id": registered.dataset_id,
                          "content_sha256": registered.canonical_sha256, "role": "train_seed"},
        "starting_model": {"model_id": "model-fixture", "content_sha256": "a" * 64},
        "diagnostic_evaluation": {"dataset_id": "diag", "dataset_sha256": "b" * 64,
                                  "protocol_version": "eval-v1", "config_sha256": "c" * 64,
                                  "role": "diagnostic_dev"},
        "final_evaluation": {"dataset_id": "holdout", "dataset_sha256": "d" * 64,
                             "protocol_version": "eval-v1", "config_sha256": "e" * 64,
                             "role": "final_holdout"},
        "provider_profile_id": "grpo-test",
        "budget": {"unit": "prompts", "expected_usable_count": 1, "max_candidates": 2,
                   "max_requests": 2, "max_output_tokens": 128},
        "training_budget": 1, "seed": 42,
    })
    register_targeted_stage_plan(root, plan)
    return registered, plan


def test_ordinary_grpo_generation_is_confirmed_persisted_and_not_trainable(tmp_path):
    _profile(tmp_path)
    provider = FakeProvider()
    jobs = GRPOGenerationJobs(tmp_path, provider_factory=lambda: provider)
    payload = {"mode": "ordinary", "profile_id": "grpo-test", "prompts": ["linear equations"],
               "request_budget": 1, "candidate_budget": 1, "max_output_tokens": 64,
               "timeout_seconds": 199}
    prepared = jobs.prepare(payload)
    assert prepared["timeout_seconds"] == 199
    assert prepared["provider_calls"] == provider.calls == 0
    assert jobs.preflight(prepared["bundle_id"])["provider_calls"] == 0
    try:
        jobs.start(prepared["bundle_id"], confirm_start=False)
    except ValueError as exc:
        assert "确认" in str(exc)
    else:
        raise AssertionError("generation must require explicit confirmation")
    started = jobs.start(prepared["bundle_id"], confirm_start=True)
    for _ in range(200):
        state = jobs.get(started["execution_id"])
        if state and state["status"] in {"completed", "partial", "failed"}:
            break
        time.sleep(0.01)
    assert state["status"] == "completed"
    assert provider.calls == 1
    preview = jobs.preview(started["execution_id"])
    assert preview["registration_allowed"] is False and preview["trainable"] is False
    assert preview["reference_status"] == "missing_until_independently_bound"
    assert preview["candidates"][0]["reference_status"] == "missing"
    reloaded = GRPOGenerationJobs(tmp_path, provider_factory=lambda: provider)
    assert reloaded.get(started["execution_id"])["status"] == "completed"
    assert provider.calls == 1


def test_targeted_grpo_generation_binds_exact_seed_and_never_reads_holdout(tmp_path):
    _profile(tmp_path)
    registered, plan = _seed_and_plan(tmp_path)
    provider = FakeProvider()
    jobs = GRPOGenerationJobs(tmp_path, provider_factory=lambda: provider)
    listed = jobs.targeted_plans()
    assert listed[0]["plan_id"] == plan.plan_id
    payload = {"mode": "targeted", "profile_id": "grpo-test", "stage_plan_id": plan.plan_id,
               "plan_version": plan.version, "request_budget": 1, "candidate_budget": 1,
               "max_output_tokens": 64}
    prepared = jobs.prepare(payload)
    assert prepared["parent_dataset_ref"] == {"dataset_id": registered.dataset_id,
        "content_sha256": registered.canonical_sha256, "role": "train_seed"}
    assert jobs.preflight(prepared["bundle_id"])["provider_calls"] == 0
    started = jobs.start(prepared["bundle_id"], confirm_start=True)
    for _ in range(200):
        state = jobs.get(started["execution_id"])
        if state and state["status"] in {"completed", "partial", "failed"}:
            break
        time.sleep(0.01)
    assert state["status"] == "completed"
    assert state["source_dataset_id"] == registered.dataset_id
    assert state["source_dataset_sha256"] == registered.canonical_sha256
    assert state["targeted_plan_id"] == plan.plan_id
    assert state["targeted_plan_version"] == plan.version
    preview = jobs.preview(started["execution_id"])
    assert preview["candidates"][0]["source_record_id"] == "seed-1"
    assert "SECRET HOLDOUT" not in json.dumps(preview)
    assert preview["trainable"] is False


def test_targeted_grpo_refuses_seed_without_registered_processing_evidence(tmp_path):
    _profile(tmp_path)
    registered, plan = _seed_and_plan(tmp_path)
    manifest_path = next((tmp_path / "data" / "processed").rglob("manifests/*.yaml"))
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    raw["processing_report"] = None
    manifest_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    jobs = GRPOGenerationJobs(tmp_path, provider_factory=lambda: FakeProvider())
    try:
        jobs.prepare({"mode": "targeted", "profile_id": "grpo-test", "stage_plan_id": plan.plan_id,
            "plan_version": plan.version, "request_budget": 1, "candidate_budget": 1,
            "max_output_tokens": 64})
    except ValueError as exc:
        assert "processing report" in str(exc)
    else:
        raise AssertionError("targeted GRPO must require verified registration evidence")


def test_grpo_generation_api_is_token_gated_and_read_only_until_confirmed(tmp_path):
    _profile(tmp_path)
    app = create_app(tmp_path, enable_synthetic_generation=True)
    provider = FakeProvider()
    app.state.grpo_generation_jobs = GRPOGenerationJobs(tmp_path, provider_factory=lambda: provider)
    with TestClient(app) as client:
        token = app.state.synthetic_generation_token
        payload = {"mode": "ordinary", "profile_id": "grpo-test", "prompts": ["fractions"],
                   "request_budget": 1, "candidate_budget": 1, "max_output_tokens": 64}
        denied = client.post("/api/synthetic/grpo/prepare", json=payload)
        assert denied.status_code == 403
        headers = {"X-Local-Synthetic-Token": token}
        prepared = client.post("/api/synthetic/grpo/prepare", json=payload, headers=headers)
        assert prepared.status_code == 200
        preflight = client.post("/api/synthetic/grpo/preflight", json={"bundle_id": prepared.json()["bundle_id"]}, headers=headers)
        assert preflight.status_code == 200 and preflight.json()["provider_calls"] == 0
        denied_start = client.post("/api/synthetic/grpo/start",
                                   json={"bundle_id": prepared.json()["bundle_id"], "confirm_start": False}, headers=headers)
        assert denied_start.status_code == 422
        started = client.post("/api/synthetic/grpo/start",
                              json={"bundle_id": prepared.json()["bundle_id"], "confirm_start": True}, headers=headers)
        assert started.status_code == 202
        execution_id = started.json()["execution_id"]
        for _ in range(200):
            state = client.get(f"/api/synthetic/grpo/runs/{execution_id}", headers=headers).json()
            if state["status"] in {"completed", "partial", "failed"}:
                break
            time.sleep(0.01)
        snapshot = client.get(f"/api/synthetic/status/grpo/{execution_id}", headers=headers)
        assert snapshot.status_code == 200
        body = snapshot.json()
        assert body["inputs"]["provider_profile_id"] == "grpo-test"
        assert body["outputs"]["artifacts"] == {"candidate_preview": "result.json"}
        assert "trusted_reference" in body["missing_evidence"]
        assert provider.calls == 1  # Read-only refresh never issues a second provider request.


def test_running_grpo_generation_can_be_cancelled_without_replaying_requests(tmp_path):
    _profile(tmp_path)
    entered, release = threading.Event(), threading.Event()

    class SlowProvider(FakeProvider):
        def generate(self, profile, request):
            entered.set()
            assert release.wait(3)
            return super().generate(profile, request)

    provider = SlowProvider()
    jobs = GRPOGenerationJobs(tmp_path, provider_factory=lambda: provider)
    prepared = jobs.prepare({"mode": "ordinary", "profile_id": "grpo-test",
        "prompts": ["one equation", "another equation"], "request_budget": 2,
        "candidate_budget": 2, "max_output_tokens": 64})
    started = jobs.start(prepared["bundle_id"], confirm_start=True)
    assert entered.wait(2)
    requested = jobs.cancel(started["execution_id"])
    assert requested["cancel_requested"] is True
    release.set()
    for _ in range(200):
        state = jobs.get(started["execution_id"])
        if state and state["status"] in {"cancelled", "failed", "completed", "partial"}:
            break
        time.sleep(0.01)
    assert state["status"] == "cancelled"
    assert provider.calls == 1
