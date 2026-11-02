import json
import shutil
import time
from io import BytesIO
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from onegpu_posttrain.synthetic.providers import GenerationResponse
from onegpu_posttrain.data.dataset_import import register_jsonl
from onegpu_posttrain.data.registry import DataSource
from onegpu_posttrain.data.targeted_stage_plan import register_targeted_stage_plan
from onegpu_posttrain.lineage import ArtifactRef, ModelLineage, RunRecord, sha256_directory
from onegpu_posttrain.synthetic.verification import AnswerAssessment
from onegpu_posttrain.ui.preference_generation_jobs import PreferenceGenerationJobs
from onegpu_posttrain.ui.preference_stage_adapter import PreferenceStageAdapter, PreferenceStageRequest
from onegpu_posttrain.ui.server import create_app
from test_targeted_stage_plan import stage_plan


class _Provider:
    def __init__(self):
        self.calls = 0

    def generate(self, profile, request):
        self.calls += 1
        completion = "Final answer: 2" if self.calls % 2 else "Final answer: 3"
        return GenerationResponse(text=json.dumps({"completion": completion}),
                                  provider_request_id=f"request-{self.calls}")


def _profile(workspace):
    path = workspace / "configs" / "synthetic" / "local" / "provider.preference-test.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump({
        "profile_id": "preference-test", "adapter": "openai_compatible",
        "endpoint": "http://127.0.0.1:9/v1/chat/completions", "api_key_env": "PREFERENCE_TEST_KEY",
        "model": "fixture-model", "revision": "fixture-rev",
    }), encoding="utf-8")


def test_ordinary_preference_job_persists_candidates_reviews_and_registration(tmp_path, monkeypatch):
    monkeypatch.setenv("PREFERENCE_TEST_KEY", "not-returned")
    _profile(tmp_path)
    provider = _Provider()
    jobs = PreferenceGenerationJobs(tmp_path, provider_factory=lambda: provider)
    plan = jobs.prepare({
        "mode": "ordinary", "profile_id": "preference-test", "prompts": ["What is 1+1?"],
        "request_budget": 2, "candidate_budget": 2, "max_output_tokens": 64,
    })
    assert plan["status"] == "prepared_not_executed"
    assert plan["provider_calls"] == 0
    assert jobs.preflight(plan["bundle_id"])["provider_calls"] == 0
    assert provider.calls == 0

    started = jobs.start(plan["bundle_id"], confirm_start=True)
    for _ in range(100):
        state = jobs.get(started["execution_id"])
        if state["status"] in {"completed", "partial", "failed"}:
            break
        time.sleep(0.01)
    assert state["status"] == "completed"
    assert state["result"]["candidate_count"] == 2
    preview = jobs.preview(started["execution_id"])
    assert preview["trainable"] is False
    ids = [row["candidate_id"] for row in preview["candidates"]]
    problem_id = preview["candidates"][0]["problem_id"]
    reviewed = jobs.review(started["execution_id"], {
        "reviewer_id": "local-user",
        "reviews": [{"problem_id": problem_id, "chosen_candidate_id": ids[0],
                     "rejected_candidate_id": ids[1], "rationale": "答案表达更清楚且完整。"}],
    })
    assert reviewed["review"]["usable_pairs"] == 1
    registered = jobs.register(started["execution_id"])
    assert registered["registration"]["record_count"] == 1
    assert registered["registration"]["parent_dataset_refs"] == []
    assert provider.calls == 2

    repo = Path(__file__).resolve().parents[1]
    for relative in ("recipes/smoke/30-dpo.yaml", "recipes/smoke/31-simpo-original.yaml"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / relative, target)
    model_path = tmp_path / "models" / "fixture-sft-merged"
    model_path.mkdir(parents=True)
    (model_path / "weights.bin").write_bytes(b"fixture sft model")
    model_digest = sha256_directory(model_path)
    model_id = f"model-{model_digest[:20]}"
    parent_run_id = "run-sft-preference-fixture"
    run_dir = tmp_path / "runs" / parent_run_id
    (run_dir / "model_artifacts").mkdir(parents=True)
    model = ModelLineage(model_id=model_id, kind="merged", path=str(model_path),
                         digest={"value": model_digest, "status": "available"},
                         parent_model_ids=["model-cpt-fixture"], produced_by_run_id=parent_run_id, stage="sft")
    (run_dir / "model_artifacts" / f"{model_id}.json").write_text(model.model_dump_json(), encoding="utf-8")
    producing_run = RunRecord(run_id=parent_run_id, run_type="train", status="completed",
        started_at="2026-01-01T00:00:00Z", completed_at="2026-01-01T00:01:00Z",
        outputs=[ArtifactRef(id=model_id, kind="model", path=str(model_path), role="merged",
                            digest={"value": model_digest, "status": "available"})])
    (run_dir / "lineage.json").write_text(producing_run.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.shutil.which", lambda _: "swift")
    adapter = PreferenceStageAdapter(tmp_path)
    generated_dataset_id = registered["registration"]["dataset_id"]
    dpo_request = PreferenceStageRequest(profile_id="dpo-smoke-v1", dataset_id=generated_dataset_id, parent_model_id=model_id)
    simpo_request = PreferenceStageRequest(profile_id="simpo-smoke-v1", dataset_id=generated_dataset_id, parent_model_id=model_id)
    dpo = adapter.preflight(dpo_request)
    simpo = adapter.preflight(simpo_request)
    assert dpo["status"] == simpo["status"] == "ready_for_cli"
    assert dpo["datasets"][0]["dataset_id"] == simpo["datasets"][0]["dataset_id"] == generated_dataset_id
    assert dpo["parent_model"]["model_id"] == simpo["parent_model"]["model_id"] == model_id
    dpo_command = adapter.build_command(dpo_request, run_id="run-dpo-generated-pref")
    simpo_command = adapter.build_command(simpo_request, run_id="run-simpo-generated-pref")
    assert dpo_command != simpo_command
    assert dpo_command[dpo_command.index("--dataset-id") + 1] == simpo_command[simpo_command.index("--dataset-id") + 1] == generated_dataset_id


def test_preference_api_is_loopback_token_gated_and_preflight_is_read_only(tmp_path, monkeypatch):
    monkeypatch.setenv("PREFERENCE_TEST_KEY", "not-returned")
    _profile(tmp_path)
    app = create_app(tmp_path, enable_synthetic_generation=True)
    with TestClient(app) as client:
        capabilities = client.get("/api/synthetic/preference/capabilities").json()
        assert capabilities["enabled"] is True
        token = capabilities["token"]
        assert capabilities["profiles"][0]["profile_id"] == "preference-test"
        denied = client.post("/api/synthetic/preference/prepare", json={"mode": "ordinary"})
        assert denied.status_code == 403
        invalid = client.post("/api/synthetic/preference/prepare", headers={"X-Local-Synthetic-Token": token},
                              json={"mode": "ordinary"})
        assert invalid.status_code == 422


def test_targeted_preference_job_uses_exact_train_seed_and_registers_parent_identity(tmp_path, monkeypatch):
    monkeypatch.setattr("onegpu_posttrain.data.targeted_preference.assess_candidate_answers_batch", lambda answers, refs: [
        AnswerAssessment(status="correct" if answer.endswith("2") else "incorrect",
                         method="fixture-reference-check", coverage="reference_and_parser",
                         extracted_answer=answer[-1], reference_available=True, verifier_available=True,
                         evidence="test-only deterministic verifier") for answer, _ in zip(answers, refs, strict=True)
    ])
    dataset = register_jsonl(BytesIO((json.dumps({
        "id": "seed-row-1", "raw_problem": "What is 1+1?",
        "messages": [{"role": "user", "content": "What is 1+1?"}, {"role": "assistant", "content": "2"}],
        "prompt_template_id": "fixture-v1",
        "metadata": {"data_role": "train_seed", "reference": "2", "problem_id": "p-1"},
    }, ensure_ascii=False) + "\n").encode()), workspace=tmp_path,
        output_root=tmp_path / "data" / "processed" / "fixture",
        stage="sft", origin=DataSource(source_id="fixture-source", kind="local_upload",
        name="fixture", license_status="verified", declared_by="user"))
    plan = stage_plan("preference", training_seed={"dataset_id": dataset.dataset_id,
        "content_sha256": dataset.canonical_sha256, "role": "train_seed"},
        budget={"unit": "pairs", "expected_usable_count": 1, "max_candidates": 2,
                "max_requests": 2, "max_output_tokens": 64})
    register_targeted_stage_plan(tmp_path, plan)
    profile_path = tmp_path / "configs" / "synthetic" / "local" / "provider.preference-test.yaml"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(yaml.safe_dump({"profile_id": "preference-test", "adapter": "fake",
                                            "model": "fixture-model"}), encoding="utf-8")
    jobs = PreferenceGenerationJobs(tmp_path, provider_factory=lambda: _Provider())
    prepared = jobs.prepare({"mode": "targeted", "profile_id": "preference-test",
        "stage_plan_id": plan.plan_id, "plan_version": plan.version,
        "request_budget": 2, "candidate_budget": 2, "max_output_tokens": 64})
    assert prepared["parent_dataset_ref"] == {"dataset_id": dataset.dataset_id,
        "content_sha256": dataset.canonical_sha256, "role": "train_seed"}
    assert jobs.preflight(prepared["bundle_id"])["provider_calls"] == 0
    started = jobs.start(prepared["bundle_id"], confirm_start=True)
    for _ in range(200):
        state = jobs.get(started["execution_id"])
        if state is not None and state["status"] in {"completed", "partial", "failed"}:
            break
        time.sleep(0.01)
    assert state["status"] == "completed"
    registered = jobs.register(started["execution_id"])["registration"]
    assert registered["parent_dataset_refs"] == [{"dataset_id": dataset.dataset_id,
        "content_sha256": dataset.canonical_sha256, "role": "train_seed"}]
    manifest_path = tmp_path / registered["registration_path"] / "manifests" / "preference-model-generated-v1.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["metadata"]["targeted_plan_id"] == plan.plan_id
    assert manifest["metadata"]["parent_dataset_refs"] == registered["parent_dataset_refs"]
    context = manifest["lineage"]["source_evidence"]["registration_context"]
    assert context["generation_execution_id"] == started["execution_id"]
    assert context["generation_bundle_id"] == prepared["bundle_id"]
    assert context["provider_model"] == "fixture-model"
    assert context["provider_identity_status"] == "profile_declared_not_independently_verified"
    assert context["registered_model_association_status"] == "not_proven"
    assert context["registered_model_id"] is None and context["registered_model_sha256"] is None
