import json

from onegpu_posttrain.ui.server import create_app


def endpoint(app, path):
    return next(route.endpoint for route in app.routes if getattr(route, "path", None) == path)


def test_targeted_overview_exposes_allowlisted_summaries_only(tmp_path):
    build = tmp_path / "artifacts" / "synthetic" / "plan-a"
    reports = build / "reports"
    private = build / "private"
    reports.mkdir(parents=True)
    private.mkdir()
    (reports / "scenario_report.json").write_text(json.dumps({
        "eval_id": "eval-1", "scenarios": [{
            "scenario_id": "algebra", "sample_count": 4, "denominator": 3,
            "correct": 1, "incorrect": 2, "accuracy": 1 / 3,
            "status": "insufficient_evidence", "question": "secret question",
        }, {
            "scenario_id": "unlabeled", "sample_count": 0, "denominator": None,
            "correct": 0, "incorrect": 0, "accuracy": None, "status": "not_evaluated",
        }],
    }), encoding="utf-8")
    (reports / "generation_validation_cost.json").write_text(json.dumps({
        "provider_profile_id": "profile-local", "provider_adapter": "fake", "model": "example-model",
        "prompt": "must not appear", "generation": {
            "execution_status": "partial_not_trained_not_reevaluated",
            "request_count": 2, "successful_candidates": 1, "target_candidate_count": 3,
            "ledger": {"cost_status": "unknown", "estimated_cost_usd": None},
        },
    }), encoding="utf-8")
    (reports / "registered_plan.json").write_text(json.dumps({
        "plan_id": "plan-safe", "status": "registered_not_executed", "private_note": "PLAN_SECRET_SENTINEL",
        "plan": {
            "version": 2, "target_scenario_id": "algebra", "stage": "sft",
            "provider_profile_id": "profile-local", "comparison_design": "before_after_pipeline",
            "budget": {"target_candidates": 7, "api_key": "PLAN_SECRET_SENTINEL"},
            "starting_model": {"model_id": "model-before", "path": "private/path"},
            "training_seed": {"dataset_id": "seed-train", "content_sha256": "c" * 64},
            "diagnostic": {"eval_id": "eval-diagnostic", "problem_text": "PLAN_SECRET_SENTINEL"},
            "final_evaluation": {"dataset_id": "holdout-final", "role": "final_holdout"},
        },
    }), encoding="utf-8")
    (private / "responses.jsonl").write_text('SECRET_SENTINEL {"text":"private response"}', encoding="utf-8")
    eval_report = tmp_path / "runs" / "evaluations" / "run-eval-grpo" / "effect_report.json"
    eval_report.parent.mkdir(parents=True)
    eval_report.write_text(json.dumps({
        "schema_version": "1.0", "status": "not_comparable", "target_state": "not_measured",
        "targeted_plan_id": "plan-grpo-safe", "targeted_plan_version": 3,
        "training_run_id": "run-grpo-1", "model_id": "model-grpo-1",
        "before_eval_id": None, "after_eval_id": "eval-after-1",
        "target_metric": None, "guardrail_metrics": {},
        "comparison_limitations": ["评测协议尚未匹配"], "private_prompt": "EFFECT_SECRET_SENTINEL",
    }), encoding="utf-8")

    app = create_app(tmp_path)
    payload = endpoint(app, "/api/targeted/overview")()
    body = json.dumps(payload)
    assert payload["scenarios"][0]["denominator"] == 3
    assert payload["scenarios"][0]["status"] == "insufficient_evidence"
    assert payload["scenarios"][1]["denominator"] is None
    assert payload["scenarios"][1]["accuracy"] is None
    assert payload["generations"][0]["execution_status"] == "partial_not_trained_not_reevaluated"
    assert payload["generations"][0]["provider_adapter"] == "fake"
    assert payload["plans"][0]["starting_model"]["model_id"] == "model-before"
    assert payload["plans"][0]["final_evaluation"]["dataset_id"] == "holdout-final"
    grpo_effect = next(item for item in payload["effects"] if item.get("stage") == "grpo")
    assert grpo_effect["target_state"] == "not_measured"
    assert grpo_effect["target_metric"] is None
    assert grpo_effect["training_run_id"] == "run-grpo-1"
    assert payload["generations"][0]["cost_status"] == "unknown"
    assert "SECRET_SENTINEL" not in body
    assert "private response" not in body
    assert "secret question" not in body
    assert "must not appear" not in body
    assert "PLAN_SECRET_SENTINEL" not in body
    assert "private/path" not in body
    assert "EFFECT_SECRET_SENTINEL" not in body


def test_targeted_capabilities_are_allowlisted_and_do_not_expose_secret_values(tmp_path, monkeypatch):
    examples = tmp_path / "configs" / "synthetic" / "examples"
    examples.mkdir(parents=True)
    (examples / "provider.fake.yaml").write_text(
        "profile_id: fake-local\nadapter: fake\nmodel: fixture-v1\napi_key: SECRET_SENTINEL\n",
        encoding="utf-8",
    )
    (examples / "provider.openai-compatible.example.yaml").write_text(
        "profile_id: remote-example\nadapter: openai_compatible\nmodel: m\n"
        "api_key_env: ONEGPU_KEY\nendpoint: https://api.example.invalid/v1/chat/completions\n",
        encoding="utf-8",
    )
    (examples / "prompt.sft-math.yaml").write_text(
        "template_id: prompt-v1\nversion: v1\ntemplate: 'Solve {question} for {topic}'\n"
        "allowed_variables: [question, topic]\n",
        encoding="utf-8",
    )
    (examples / "output.sft-candidate.yaml").write_text(
        "protocol_id: candidate-v1\nversion: v1\nresponse_format: json_object\n"
        "required_fields: [problem_id, solution, final_answer]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ONEGPU_KEY", "ANOTHER_SECRET_SENTINEL")

    app = create_app(tmp_path)
    payload = endpoint(app, "/api/targeted/capabilities")()
    body = json.dumps(payload)
    assert "SECRET_SENTINEL" not in body
    assert "ANOTHER_SECRET_SENTINEL" not in body
    assert "api_key" not in payload["profiles"][0]
    assert payload["profiles"][1]["api_key_env"] == "ONEGPU_KEY"
    targeted_routes = [route for route in app.routes if route.path.startswith("/api/targeted/")]
    assert {method for route in targeted_routes for method in route.methods or set()} == {"GET"}


def test_targeted_stage_capabilities_report_requirements_and_blockers_without_execution(tmp_path, monkeypatch):
    from onegpu_posttrain.ui import targeted_capabilities

    monkeypatch.setattr(targeted_capabilities, "_gpu_snapshot", lambda: {
        "available": False, "status": "unknown", "reason": "fixture"
    })
    for relative in (
        "recipes/smoke/10-peft-cpt.yaml",
        "recipes/mini/23-sft-1024-fallback.yaml",
        "recipes/smoke/30-dpo.yaml",
        "recipes/smoke/31-simpo-original.yaml",
        "recipes/smoke/40-grpo-standard.yaml",
        "recipes/smoke/41-grpo-lowmem.yaml",
        "configs/evaluation/math-v1.yaml",
        "configs/evaluation/math-v1-launcher-mini.yaml",
        "plugins/grpo_rewards.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")

    app = create_app(tmp_path)
    payload = endpoint(app, "/api/targeted/stages")()

    assert payload["read_only"] is True
    assert [item["stage"] for item in payload["stages"]] == ["cpt", "sft", "preference", "grpo"]
    cpt, sft, preference, grpo = payload["stages"]
    assert cpt["dataset_schema"] == "cpt-corpus-v1"
    assert cpt["parent_model_stage"] == "base"
    assert cpt["evaluation"]["available"] is True
    assert cpt["generation_ui_available"] is False
    assert any("--enable-synthetic-generation" in blocker for blocker in cpt["generation_blockers"])
    assert not any("生成适配器" in blocker for blocker in cpt["blockers"])
    assert sft["dataset_schema"] == "sft-v1"
    assert sft["parent_model"]["status"] == "select_registered_artifact"
    assert preference["verifier"]["status"] == "preflight_required"
    assert {profile["algorithm"] for profile in preference["profiles"]} == {"dpo", "simpo"}
    assert grpo["reward_contract"]["available"] is True
    assert all(item["hardware"]["status"] == "unknown" for item in payload["stages"])
    assert all(item["ready_for_execution"] is False for item in payload["stages"])


def test_stage_capabilities_separate_training_from_generation(tmp_path, monkeypatch):
    from onegpu_posttrain.ui import targeted_capabilities

    recipes = {
        "recipes/smoke/10-peft-cpt.yaml": "max_length: 1024\nmax_steps: 1\nlearning_rate: 0.0001\ngradient_accumulation_steps: 1\n",
        "recipes/mini/23-sft-1024-fallback.yaml": "max_length: 1024\nnum_train_epochs: 1\nlearning_rate: 0.0001\ngradient_accumulation_steps: 1\n",
        "recipes/smoke/30-dpo.yaml": "max_length: 1024\nmax_steps: 1\nlearning_rate: 0.0001\ngradient_accumulation_steps: 1\n",
        "recipes/smoke/31-simpo-original.yaml": "max_length: 1024\nmax_steps: 1\nlearning_rate: 0.0001\ngradient_accumulation_steps: 1\n",
        "recipes/smoke/40-grpo-standard.yaml": "max_prompt_length_candidate: 512\nmax_completion_length_candidate: 512\nmax_steps: 1\nlearning_rate: 0.00001\ngradient_accumulation_steps: 1\n",
        "recipes/smoke/41-grpo-lowmem.yaml": "max_prompt_length_candidate: 512\nmax_completion_length_candidate: 512\nmax_steps: 1\nlearning_rate: 0.00001\ngradient_accumulation_steps: 1\n",
    }
    for relative, training in recipes.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("training:\n" + "".join(f"  {line}\n" for line in training.splitlines()), encoding="utf-8")
    for relative in ("configs/evaluation/math-v1.yaml", "configs/evaluation/math-v1-launcher-mini.yaml",
                     "plugins/grpo_rewards.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(targeted_capabilities, "_gpu_snapshot", lambda: {"available": False, "status": "unknown"})

    payload = targeted_capabilities.stage_capabilities(
        tmp_path, synthetic_generation_enabled=False, experiment_launcher_enabled=True,
    )

    for stage in payload["stages"]:
        assert stage["training_ui_available"] is True
        assert stage["training_ready"] is True
        assert stage["ready_for_execution"] is True
        assert stage["generation_ui_available"] is False
        assert stage["generation_blockers"]
        assert not any("生成适配器" in item for item in stage["training_blockers"])


def test_generation_mode_catalog_reports_gaps_without_provider_calls(tmp_path, monkeypatch):
    from onegpu_posttrain.ui import targeted_capabilities

    monkeypatch.setattr(targeted_capabilities, "_gpu_snapshot", lambda: {"status": "unknown"})
    monkeypatch.setattr(targeted_capabilities, "_profiles", lambda root: [])
    monkeypatch.setattr(targeted_capabilities, "_validated_local_configs", lambda root: {
        "templates": [], "protocols": [], "policies": [],
    })
    stages = targeted_capabilities.stage_capabilities(
        tmp_path, synthetic_generation_enabled=True, experiment_launcher_enabled=False,
    )["stages"]
    for item in stages:
        assert set(item["generation_modes"]) == {"ordinary", "targeted"}
        for mode in item["generation_modes"].values():
            assert mode["provider_calls"] == 0
            assert mode["status"] == "blocked"
            assert any("Provider" in blocker for blocker in mode["blockers"])
    sft = next(item for item in stages if item["stage"] == "sft")
    assert any("协议" in check for check in sft["generation_modes"]["ordinary"]["pending_checks"])
    grpo = next(item for item in stages if item["stage"] == "grpo")
    assert any("可信 reference" in check and "reward" in check
               for check in grpo["generation_modes"]["targeted"]["pending_checks"])
    cpt = next(item for item in stages if item["stage"] == "cpt")
    assert cpt["generation_modes"]["targeted"]["domain_adapter"] == "shared_cpt_job"
    assert cpt["generation_modes"]["ordinary"]["domain_adapter"] == "shared_cpt_job"
    assert cpt["generation_modes"]["ordinary"]["quality_verifier"] == (
        "available_static_checks_factuality_not_verified"
    )
    preference = next(item for item in stages if item["stage"] == "preference")
    assert preference["generation_modes"]["targeted"]["domain_adapter"] == "targeted_pair_builder_only"
    assert "未接通" in " ".join(preference["generation_modes"]["ordinary"]["blockers"])
    assert grpo["generation_modes"]["targeted"]["domain_adapter"] == "prompt_only_candidate_job"
    assert grpo["reward_contract"]["status"] == "missing"


def test_generation_mode_catalog_does_not_claim_selected_inputs_are_ready(tmp_path, monkeypatch):
    from onegpu_posttrain.ui import targeted_capabilities

    monkeypatch.setattr(targeted_capabilities, "_gpu_snapshot", lambda: {"status": "unknown"})
    monkeypatch.setattr(targeted_capabilities, "_profiles", lambda root: [
        {"is_example": False, "credential_env_configured": True},
    ])
    monkeypatch.setattr(targeted_capabilities, "_validated_local_configs", lambda root: {
        "templates": [{"id": "template"}], "protocols": [{"id": "protocol"}], "policies": [],
    })
    stages = targeted_capabilities.stage_capabilities(
        tmp_path, synthetic_generation_enabled=True, experiment_launcher_enabled=False,
    )["stages"]
    sft = next(item for item in stages if item["stage"] == "sft")
    for mode in sft["generation_modes"].values():
        assert mode["status"] == "preflight_required"
        assert mode["provider_calls"] == 0
        assert mode["pending_checks"]
    assert next(item for item in stages if item["stage"] == "cpt")["generation_modes"]["ordinary"]["status"] == "preflight_required"
    assert next(item for item in stages if item["stage"] == "preference")["generation_modes"]["ordinary"]["status"] == "blocked"
    grpo = next(item for item in stages if item["stage"] == "grpo")
    assert grpo["generation_ui_available"] is True
    assert grpo["generation_modes"]["ordinary"]["domain_adapter"] == "prompt_only_candidate_job"
    assert grpo["generation_modes"]["targeted"]["candidate_protocol"] == "prompt_only_not_trainable_without_reference"
    assert any("可信 reference" in check for check in grpo["generation_modes"]["ordinary"]["pending_checks"])
