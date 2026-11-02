from __future__ import annotations

import json
from pathlib import Path

import yaml
from fastapi import HTTPException
from starlette.requests import Request

from onegpu_posttrain.data.io import write_jsonl
from onegpu_posttrain.data.policy import policy_templates
from onegpu_posttrain.data.policy_experiment import load_experiment
from onegpu_posttrain.ui.server import create_app


def endpoint(app, path):
    return next(route.endpoint for route in app.routes if getattr(route, "path", None) == path)


def request_context(headers=None):
    return Request({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
                    "scheme": "http", "path": "/", "raw_path": b"/", "query_string": b"", "headers":
                    [(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()],
                    "server": ("testserver", 80), "client": ("127.0.0.1", 1234), "root_path": ""})


def expect_http_error(call, status):
    try:
        call()
    except HTTPException as exc:
        assert exc.status_code == status
    else:
        raise AssertionError(f"expected HTTP {status}")


def test_policy_editor_is_read_only_unless_explicitly_enabled(tmp_path):
    app = create_app(tmp_path)
    assert endpoint(app, "/api/health")()["read_only"] is True
    capabilities = endpoint(app, "/api/policies/capabilities")()
    assert capabilities["enabled"] is False
    expect_http_error(lambda: endpoint(app, "/api/policies/save")(
        request_context(), {"policies": [policy_templates()[0]["config"]] * 2}, None), 403)
    assert not (tmp_path / "artifacts").exists()


def test_editor_rejects_symlinked_write_root(tmp_path, monkeypatch):
    source_dir = tmp_path / "configs" / "data" / "policy-sources"
    source_dir.mkdir(parents=True)
    (source_dir / "fixture.yaml").write_text(yaml.safe_dump({
        "title": "fixture", "input_jsonl": "raw.jsonl", "source_repo": "owned", "source_revision": "v1",
        "benchmark_jsonl": "benchmark.jsonl", "license_status": "declared", "license_note": "fixture",
    }), encoding="utf-8")
    app = create_app(tmp_path, enable_policy_editor=True)
    capabilities = endpoint(app, "/api/policies/capabilities")()
    policy = policy_templates()[0]["config"]
    real_is_symlink = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == tmp_path / "artifacts" or real_is_symlink(path))
    expect_http_error(lambda: endpoint(app, "/api/policies/save")(
        request_context({"X-Local-Editor-Token": capabilities["token"]}),
        {"source_id": capabilities["sources"][0]["id"], "policies": [policy, policy]}, capabilities["token"]), 422)
    assert not (tmp_path / "artifacts").exists()


def test_editor_requires_same_origin_and_opaque_token_and_writes_only_new_drafts(tmp_path):
    source_dir = tmp_path / "configs" / "data" / "policy-sources"
    source_dir.mkdir(parents=True)
    (source_dir / "fixture.yaml").write_text(yaml.safe_dump({
        "title": "fixture", "input_jsonl": "raw.jsonl", "source_repo": "owned", "source_revision": "v1",
        "benchmark_jsonl": "benchmark.jsonl", "license_status": "declared", "license_note": "fixture",
        "tokenizer": {"test_only": True},
    }), encoding="utf-8")
    app = create_app(tmp_path, enable_policy_editor=True)
    capabilities = endpoint(app, "/api/policies/capabilities")()
    token = capabilities["token"]
    policy = policy_templates()[0]["config"]
    headers = {"X-Local-Editor-Token": token}
    payload = {"source_id": capabilities["sources"][0]["id"],
               "policies": [policy, {**policy, "title": "B", "rules": {**policy["rules"], "completeness": "report_only"}}]}

    save = endpoint(app, "/api/policies/save")
    expect_http_error(lambda: save(request_context({"X-Local-Editor-Token": "wrong"}), payload, "wrong"), 403)
    expect_http_error(lambda: save(request_context({**headers, "Origin": "https://attacker.invalid"}), payload, token), 403)
    result = save(request_context(headers), payload, token)
    folder = tmp_path / result["path"]
    assert (folder / "policy-a.yaml").is_file()
    assert (folder / "policy-b.yaml").is_file()
    assert result["status"] == "saved_policy_draft"
    assert save(request_context(headers), payload, token)["draft_id"] != result["draft_id"]
    assert len(list((tmp_path / "artifacts" / "data-experiments").iterdir())) == 2

    experiment = {
        "schema_version": "1.0", "experiment_id": "exp-test-policy-ui", "question": "fixture",
        "evidence_kind": "fixture", "experiment_type": "pipeline_effect", "independent_variable": None,
        "source_pool_id": "pool-fixture", "split_snapshot_id": "split-fixture",
        "controls": {"parent_model_id": "fixture-parent", "parent_model_sha256": "0" * 64,
                      "recipe_sha256": "1" * 64, "training": {"max_length": 128}, "seeds": [42]},
        "evaluation": {"dataset_ids": ["fixture-eval"], "dataset_sha256": {"fixture-eval": "2" * 64},
                       "roles": {"fixture-eval": "diagnostic_dev"}, "protocol_version": "fixture-v1",
                       "config_sha256": "3" * 64},
        "primary_metrics": [{"name": "internal_math.accuracy", "unit": "ratio", "direction": "higher"}],
        "guardrail_metrics": [], "budget": {"unit": "examples", "value": 2},
    }
    save_plan = endpoint(app, "/api/policies/save-experiment")
    plan_result = save_plan(request_context(headers), {"draft_id": result["draft_id"],
                                                       "source_id": payload["source_id"], "experiment": experiment}, token)
    plan_path = tmp_path / plan_result["plan_path"]
    saved_experiment = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    typed_experiment = load_experiment(plan_path)
    assert typed_experiment.experiment_id == plan_result["experiment_id"]
    assert saved_experiment["a"]["manifest"].endswith("/data-a/manifests/sft-policy-v1.yaml")
    assert saved_experiment["b"]["manifest"].endswith("/data-b/manifests/sft-policy-v1.yaml")
    assert saved_experiment["evidence_kind"] == "fixture"
    expect_http_error(lambda: save_plan(request_context(headers), {"draft_id": result["draft_id"],
        "source_id": payload["source_id"], "experiment": experiment}, token), 409)
    expect_http_error(lambda: save_plan(request_context(headers), {"draft_id": result["draft_id"],
        "source_id": payload["source_id"], "experiment": {**experiment, "a": {"manifest": "../../secret"}}}, token), 422)


def test_preview_rejects_arbitrary_source_and_bounds_rows(tmp_path):
    app = create_app(tmp_path, enable_policy_editor=True)
    token = endpoint(app, "/api/policies/capabilities")()["token"]
    headers = {"X-Local-Editor-Token": token}
    policy = policy_templates()[0]["config"]
    preview = endpoint(app, "/api/policies/preview")
    expect_http_error(lambda: preview(request_context(headers), {"source_id": "../../secret", "policy": policy}, token), 422)
    capabilities = endpoint(app, "/api/policies/capabilities")()
    if capabilities["sources"]:
        source_id = capabilities["sources"][0]["id"]
        expect_http_error(lambda: preview(request_context(headers), {"source_id": source_id, "rows": 1001, "policy": policy}, token), 422)
    assert not (tmp_path / "secret").exists()


def test_preview_uses_only_operator_registered_local_files_and_redacts_records(tmp_path):
    source_root = tmp_path / "fixture"
    source_root.mkdir()
    rows = [{"problem": f"What is {i} plus 2?", "answer": str(i + 2), "uuid": f"fixture-{i}",
             "source": "project-owned", "generations": [f"Answer is \\boxed{{{i + 2}}}.", f"Wrong: \\boxed{{{i + 3}}}"],
             "correctness_math_verify": [True, False], "is_reasoning_complete": [True, True],
             "finish_reasons": ["stop", "stop"]} for i in range(30)]
    write_jsonl(source_root / "raw.jsonl", rows)
    write_jsonl(source_root / "benchmark.jsonl", [{"problem": "An unrelated evaluation-only benchmark question."}])
    config_dir = tmp_path / "configs" / "data" / "policy-sources"
    config_dir.mkdir(parents=True)
    config = {"title": "local UI fixture", "input_jsonl": str(source_root / "raw.jsonl"),
              "source_repo": "project-owned-fixture", "source_revision": "v1",
              "benchmark_jsonl": str(source_root / "benchmark.jsonl"), "license_status": "declared",
              "license_note": "project-owned fixture", "tokenizer": {"test_only": True},
              "split": {"sft_weight": 1, "alignment_weight": 1, "holdout_weight": 1}}
    (config_dir / "fixture.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    app = create_app(tmp_path, enable_policy_editor=True)
    capabilities = endpoint(app, "/api/policies/capabilities")()
    source = capabilities["sources"][0]
    policy = dict(policy_templates()[0]["config"])
    policy["resources"] = {"max_length": 128, "target_count": 2}
    result = endpoint(app, "/api/policies/preview")(
        request_context({"X-Local-Editor-Token": capabilities["token"]}),
        {"source_id": source["id"], "policy": policy, "rows": 20}, capabilities["token"])
    assert result["report"]["preview"] is True
    assert result["report"]["counters"]["raw_rows"] <= 20
    assert "generations" not in json.dumps(result)
    assert result["source_pool_id"] and result["split_snapshot_id"]
    app.state.policy_preview_lock.acquire()
    try:
        expect_http_error(lambda: endpoint(app, "/api/policies/preview")(
            request_context({"X-Local-Editor-Token": capabilities["token"]}),
            {"source_id": source["id"], "policy": policy, "rows": 20}, capabilities["token"]), 409)
    finally:
        app.state.policy_preview_lock.release()


def test_reports_endpoint_returns_only_allowlisted_bounded_comparison_fields(tmp_path):
    report_dir = tmp_path / "artifacts" / "data-experiments" / "demo"
    report_dir.mkdir(parents=True)
    (report_dir / "data_effect_report.json").write_text(json.dumps({
        "report_id": "der-fixture", "comparison_status": "NOT_EVALUATED", "experiment_id": "exp-fixture",
        "created_at": "2026-10-18T00:00:00Z", "experiment_type": "pipeline_effect",
        "evidence_kind": "fixture", "compatibility_issues": [], "policy_changes": [], "data": {},
        "model_metrics": [], "limitations": ["no training"], "raw_rows": [{"secret": "do-not-expose"}],
    }), encoding="utf-8")
    result = endpoint(create_app(tmp_path), "/api/policies/reports")()
    assert len(result["reports"]) == 1
    assert result["reports"][0]["report_id"] == "der-fixture"
    assert "raw_rows" not in result["reports"][0]
