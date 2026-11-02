import json
import time

import pytest

from onegpu_posttrain.data.generated_sft_registration import (
    register_generated_sft_dataset,
)
from onegpu_posttrain.data.manifest import load_manifest
from onegpu_posttrain.data.tokenization import WhitespaceTokenizer
from onegpu_posttrain.synthetic.prompt_only import (
    load_prompt_only_bundle,
    prepare_prompt_only_bundle,
)
from onegpu_posttrain.synthetic.prompt_only_workflow import (
    execute_prompt_only_sft_generation,
)
from onegpu_posttrain.synthetic.providers import GenerationResponse
from onegpu_posttrain.ui.server import create_app

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient


def _workspace(root):
    config = root / "configs/synthetic/provider.local.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "profile_id: local-test\nadapter: openai_compatible\nendpoint: https://example.test/v1/chat/completions\n"
        "model: fixture-model\napi_key_env: FIXTURE_KEY\n"
        "supported_parameters: [temperature, max_output_tokens]\n",
        encoding="utf-8",
    )
    (root / ".env.local").write_text("FIXTURE_KEY=not-returned\n", encoding="utf-8")
    return root


def test_prompt_only_plan_has_no_seed_and_is_immutable(tmp_path):
    root = _workspace(tmp_path)
    prepared = prepare_prompt_only_bundle(root, topic="基础代数", skill="一次方程", difficulty="introductory",
                                         constraints="答案为整数", target_count=2)
    plan, files = load_prompt_only_bundle(root, prepared["bundle_id"])
    assert plan.mode == "prompt_only_sft"
    assert not hasattr(plan, "source_dataset_id")
    assert set(files) == {"plan", "policy", "profile"}
    repeated = prepare_prompt_only_bundle(root, topic="基础代数", skill="一次方程", difficulty="introductory",
                                          constraints="答案为整数", target_count=2)
    assert repeated["bundle_id"] == prepared["bundle_id"]
    changed = prepare_prompt_only_bundle(root, topic="基础代数", skill="一次方程", difficulty="advanced",
                                         constraints="答案为整数", target_count=2)
    assert changed["bundle_id"] != prepared["bundle_id"]
    with pytest.raises(ValueError):
        prepare_prompt_only_bundle(root, topic="x", skill="一次方程", difficulty="introductory", target_count=2)


def test_prompt_only_generation_registers_root_dataset_lineage(tmp_path, monkeypatch):
    root = _workspace(tmp_path)
    prepared = prepare_prompt_only_bundle(root, topic="基础代数", skill="一次方程", difficulty="introductory",
                                         constraints="答案为整数", target_count=1)
    import onegpu_posttrain.synthetic.prompt_only_workflow as workflow
    monkeypatch.setattr(workflow, "_tokenizer", lambda *_: WhitespaceTokenizer())

    class Provider:
        calls = 0

        def generate(self, profile, request):
            self.calls += 1
            assert "基础代数" in request.prompt
            return GenerationResponse(text=json.dumps({
                "question": "小明有 3 个苹果，又买了 2 个，现在有几个？",
                "solution": "将原有数量与新增数量相加：3+2=5。",
                "final_answer": "5",
            }, ensure_ascii=False), finish_reason="stop")

    provider = Provider()
    result = execute_prompt_only_sft_generation(root, prepared["bundle_id"], execution_id="sftgen-" + "a" * 20,
                                                provider=provider)
    assert provider.calls == 1
    assert result["usable_records"] == 1
    assert result["semantic_correctness"] == "not_independently_verified"
    registration = register_generated_sft_dataset(root, {
        "execution_id": result["execution_id"], "bundle_id": result["bundle_id"],
        "status": result["status"], "result": result,
    })
    manifest = load_manifest(root / registration["manifest_path"])
    assert manifest.lineage.parents == []
    assert manifest.lineage.source_evidence["generation_mode"] == "prompt_only_sft"
    assert manifest.lineage.source_evidence["semantic_correctness"] == "not_independently_verified"
    assert manifest.lineage.source_evidence["generation_execution_id"] == result["execution_id"]
    build_run = json.loads((root / "data/processed/model-generated" / registration["dataset_id"]
                            / "runs" / manifest.lineage.produced_by_run_id / "lineage.json").read_text(encoding="utf-8"))
    assert all(item.get("role") != "generation_seed" for item in build_run["inputs"])


def test_prompt_only_plan_rejects_unbounded_fields(tmp_path):
    root = _workspace(tmp_path)
    with pytest.raises(ValueError):
        prepare_prompt_only_bundle(root, topic="x" * 121, skill="equations", difficulty="introductory",
                                   constraints="", target_count=1)
    with pytest.raises(ValueError):
        prepare_prompt_only_bundle(root, topic="algebra", skill="equations", difficulty="expert",
                                   constraints="", target_count=1)


def test_prompt_only_api_preflight_is_read_only_and_requires_explicit_start(tmp_path, monkeypatch):
    root = _workspace(tmp_path)
    import onegpu_posttrain.synthetic.prompt_only_workflow as workflow
    import onegpu_posttrain.ui.sft_generation_jobs as jobs

    class Provider:
        calls = 0

        def generate(self, profile, request):
            self.calls += 1
            return GenerationResponse(text=json.dumps({"question": "Compute 2+3.", "solution": "2+3=5.",
                                                        "final_answer": "5"}), finish_reason="stop")

    provider = Provider()
    monkeypatch.setattr(workflow, "_tokenizer", lambda *_: WhitespaceTokenizer())
    monkeypatch.setattr(jobs, "execute_prompt_only_sft_generation",
                        lambda workspace, bundle_id, **kwargs: workflow.execute_prompt_only_sft_generation(
                            workspace, bundle_id, provider=provider, **kwargs))
    with TestClient(create_app(root, enable_synthetic_generation=True, enable_experiment_launcher=True)) as client:
        token = client.get("/api/synthetic/sft/capabilities").json()["token"]
        headers = {"X-Local-Synthetic-Token": token}
        prepared = client.post("/api/synthetic/sft/prepare", headers=headers, json={
            "mode": "prompt_only", "topic": "algebra", "skill": "addition",
            "difficulty": "introductory", "target_count": 1,
        })
        assert prepared.status_code == 200, prepared.text
        bundle_id = prepared.json()["bundle_id"]
        budget = {"bundle_id": bundle_id, "usable_target": 1, "max_requests": 2, "target_candidates": 1}
        assert client.post("/api/synthetic/sft/preflight", headers=headers, json=budget).status_code == 200
        assert provider.calls == 0
        assert client.post("/api/synthetic/sft/start", headers=headers,
                           json=budget | {"confirm_start": False}).status_code == 422
        started = client.post("/api/synthetic/sft/start", headers=headers,
                              json=budget | {"confirm_start": True})
        assert started.status_code == 202, started.text
        execution_id = started.json()["execution_id"]
        state = None
        for _ in range(100):
            state = client.get(f"/api/synthetic/sft/runs/{execution_id}", headers=headers).json()
            if state["status"] in {"completed", "partial", "failed"}:
                break
            time.sleep(0.02)
        assert state["status"] == "completed"
        assert provider.calls == 1
        preview = client.get(f"/api/synthetic/sft/runs/{execution_id}/preview?limit=1", headers=headers)
        assert preview.status_code == 200 and preview.json()["records"]
        registered = client.post(f"/api/synthetic/sft/runs/{execution_id}/register", headers=headers,
                                 json={"confirm_reviewed": True})
        assert registered.status_code == 200, registered.text
        assert provider.calls == 1
        dataset_id = registered.json()["registration"]["dataset_id"]
        catalog = client.get("/api/data").json()
        assert dataset_id in {dataset["dataset_id"] for build in catalog["builds"]
                              for dataset in build["datasets"]}
        launcher_datasets = client.get("/api/launcher/capabilities").json()["datasets"]
        assert dataset_id in {dataset["dataset_id"] for dataset in launcher_datasets}


def test_prompt_only_filters_duplicate_and_truncated_outputs_with_bounded_requests(tmp_path, monkeypatch):
    root = _workspace(tmp_path)
    prepared = prepare_prompt_only_bundle(root, topic="algebra", skill="addition", difficulty="introductory",
                                         target_count=2)
    import onegpu_posttrain.synthetic.prompt_only_workflow as workflow
    monkeypatch.setattr(workflow, "_tokenizer", lambda *_: WhitespaceTokenizer())

    def response(question, finish="stop"):
        return GenerationResponse(text=json.dumps({"question": question, "solution": "Show addition: 2+3=5.",
                                                  "final_answer": "5"}), finish_reason=finish)

    class Provider:
        def __init__(self):
            self.responses = [response("Compute 2+3."), response("Compute 2+3."),
                              response("Truncated example.", "length"), response("Compute 4+1.")]
            self.calls = 0

        def generate(self, profile, request):
            self.calls += 1
            return self.responses.pop(0)

    provider = Provider()
    result = execute_prompt_only_sft_generation(root, prepared["bundle_id"],
        execution_id="sftgen-" + "b" * 20, provider=provider)
    assert provider.calls == 4
    assert result["request_count"] == result["max_requests"] == 4
    assert result["usable_records"] == 2
    assert result["candidate_rejections"]["duplicate_question"] == 1
    assert result["candidate_rejections"]["truncated"] == 1


def test_prompt_only_exhausted_budget_returns_failure_evidence_instead_of_validation_error(tmp_path, monkeypatch):
    root = _workspace(tmp_path)
    prepared = prepare_prompt_only_bundle(root, topic="algebra", skill="addition", difficulty="introductory",
                                         target_count=1)
    import onegpu_posttrain.synthetic.prompt_only_workflow as workflow
    monkeypatch.setattr(workflow, "_tokenizer", lambda *_: WhitespaceTokenizer())

    class Provider:
        calls = 0

        def generate(self, profile, request):
            self.calls += 1
            # A valid JSON payload that fails the canonical SFT candidate schema.
            return GenerationResponse(text=json.dumps({"question": "Compute 2+3."}), finish_reason="stop")

    provider = Provider()
    execution_id = "sftgen-" + "c" * 20
    result = execute_prompt_only_sft_generation(root, prepared["bundle_id"], execution_id=execution_id,
                                                provider=provider)

    assert provider.calls == 2
    assert result["status"] == "failed"
    assert result["usable_records"] == 0
    assert result["shortfall"] == 1
    assert result["candidate_rejections"]["invalid_candidate_schema"] == 2
    report_path = (root / "artifacts/synthetic" / prepared["bundle_id"] / "builds" / execution_id
                   / "reports/processing_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["stop_reason"] == "request_budget_exhausted"
    assert report["processing_steps"][0]["first_rejection_reasons"] == {"invalid_candidate_schema": 2}


def test_prompt_only_stops_after_uncertain_provider_request(tmp_path, monkeypatch):
    from onegpu_posttrain.synthetic.providers import ProviderError
    import onegpu_posttrain.synthetic.prompt_only_workflow as workflow

    root = _workspace(tmp_path)
    prepared = prepare_prompt_only_bundle(root, topic="algebra", skill="addition", difficulty="introductory",
                                         target_count=2)
    monkeypatch.setattr(workflow, "_tokenizer", lambda *_: WhitespaceTokenizer())

    class Provider:
        calls = 0

        def generate(self, profile, request):
            self.calls += 1
            raise ProviderError("timeout", "secret detail", retryable=True, uncertain=True)

    provider = Provider()
    execution_id = "sftgen-" + "d" * 20
    result = execute_prompt_only_sft_generation(root, prepared["bundle_id"], execution_id=execution_id,
                                                provider=provider)
    assert provider.calls == 1
    assert result["status"] == "failed"
    assert result["stop_reason"] == "uncertain_request_requires_explicit_review"
    assert "secret detail" not in json.dumps(result)
    report_path = (root / "artifacts/synthetic" / prepared["bundle_id"] / "builds" / execution_id
                   / "reports/processing_report.json")
    assert json.loads(report_path.read_text(encoding="utf-8"))["stop_reason"] == result["stop_reason"]
    with pytest.raises(ValueError, match="结果不确定"):
        execute_prompt_only_sft_generation(root, prepared["bundle_id"], execution_id="sftgen-" + "e" * 20,
                                           provider=provider)
    assert provider.calls == 1
    ledger = (root / "artifacts/synthetic" / prepared["bundle_id"] / "generation-requests.jsonl")
    assert "secret detail" not in ledger.read_text(encoding="utf-8")


def test_prompt_only_respects_smaller_execution_budget(tmp_path, monkeypatch):
    root = _workspace(tmp_path)
    prepared = prepare_prompt_only_bundle(root, topic="algebra", skill="addition", difficulty="introductory",
                                         target_count=2)
    import onegpu_posttrain.synthetic.prompt_only_workflow as workflow
    monkeypatch.setattr(workflow, "_tokenizer", lambda *_: WhitespaceTokenizer())

    class Provider:
        calls = 0
        def generate(self, profile, request):
            self.calls += 1
            return GenerationResponse(text="{}", finish_reason="stop")

    provider = Provider()
    result = execute_prompt_only_sft_generation(root, prepared["bundle_id"], execution_id="sftgen-" + "1" * 20,
                                                provider=provider, max_requests=1)
    assert provider.calls == result["request_count"] == result["max_requests"] == 1
