from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from onegpu_posttrain.ui.run_workspace import MAX_RECORD_BYTES, RunWorkspace
from onegpu_posttrain.assistant.contracts import EvidenceQuery
from onegpu_posttrain.assistant.evidence_reader import WorkspaceEvidenceReader
from onegpu_posttrain.ui.server import create_app


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_run_discovery_normalizes_supported_completed_failed_legacy_and_launcher_records(tmp_path):
    root = tmp_path.resolve()
    put(root / "runs/mini/sft/run.json", {"run_id": "run-sft-one", "stage": "sft", "run_label": "mini-sft",
        "experiment_id": "exp-mini", "train": {"status": "PASS", "returncode": 0}})
    put(root / "runs/mini/eval/evaluation_run.json", {"run_id": "run-eval-one", "status": "partial",
        "started_at": "2026-10-20T00:00:00Z", "parent_run_id": "run-sft-one"})
    put(root / "runs/legacy/run.json", {"run_id": "run-legacy"})
    put(root / "runs/failed/run.json", {"run_id": "run-failed", "status": "failed", "stage": "cpt"})
    put(root / ".onegpu-posttrain/launcher/runs/run-interrupted.json", {"run_id": "run-interrupted",
        "status": "interrupted", "run_kind": "training", "output": "runs/launcher/run-interrupted"})
    result = RunWorkspace(root).list_runs()
    by_id = {row["run_id"]: row for row in result["runs"]}
    assert by_id["run-sft-one"]["status"] == "completed"
    assert by_id["run-sft-one"]["run_label"] == "mini-sft"
    assert by_id["run-eval-one"]["run_kind"] == "evaluation"
    assert by_id["run-failed"]["status"] == "failed"
    assert by_id["run-interrupted"]["status"] == "interrupted"
    assert by_id["run-legacy"]["stage"] is None
    assert by_id["run-legacy"]["status"] == "unknown"


def test_run_discovery_deduplicates_by_id_and_reports_status_conflict(tmp_path):
    root = tmp_path.resolve()
    put(root / "runs/first/run.json", {"run_id": "run-duplicate", "status": "completed", "stage": "sft"})
    put(root / ".onegpu-posttrain/launcher/runs/run-duplicate.json", {"run_id": "run-duplicate",
        "status": "interrupted", "output": "runs/first"})
    result = RunWorkspace(root).list_runs()
    matches = [row for row in result["runs"] if row["run_id"] == "run-duplicate"]
    assert len(matches) == 1
    assert matches[0]["status"] == "interrupted"
    assert matches[0]["status_conflict"] is True
    assert len(matches[0]["source_refs"]) == 2


def test_run_discovery_skips_malformed_oversized_symlink_and_bounds_scan(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    put(root / "runs/malformed/run.json", {"no_id": True})
    big = root / "runs/oversized/run.json"
    big.parent.mkdir(parents=True)
    big.write_text("{}".ljust(MAX_RECORD_BYTES + 1), encoding="utf-8")
    target = root / "outside/run.json"
    put(target, {"run_id": "run-outside"})
    link = root / "runs/symlink"
    try:
        link.symlink_to(target.parent, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")
    result = RunWorkspace(root).list_runs()
    assert result["runs"] == []
    monkeypatch.setattr("onegpu_posttrain.ui.run_workspace.MAX_RUN_RECORDS", 1)
    put(root / "runs/ok/run.json", {"run_id": "run-ok", "status": "failed"})
    assert RunWorkspace(root).list_runs()["partial_scan"] is True


def test_run_filters_detail_and_launcher_output_are_bounded_and_read_only(tmp_path):
    root = tmp_path.resolve()
    output = root / "runs/launcher/run-real"
    put(output / "run.json", {"run_id": "run-real", "status": "completed", "stage": "sft", "run_label": "pilot"})
    put(output / "lineage.json", {"run_id": "run-real", "lineage_status": "complete",
        "inputs": [{"id": "ds-1", "kind": "dataset", "digest": {"algorithm": "sha256", "status": "available", "value": "a" * 64}}],
        "outputs": [{"id": "model-one", "kind": "model", "role": "merged",
                     "digest": {"algorithm": "sha256", "status": "available", "value": "b" * 64}}]})
    put(output / "resolved_config.json", {"max_length": 1024})
    (output / "launcher.log").write_text("training complete\n", encoding="utf-8")
    api = create_app(root)
    with TestClient(api) as client:
        listing = client.get("/api/runs?stage=sft&status=completed&q=pilot&limit=10")
        assert listing.status_code == 200
        assert listing.json()["total"] == 1
        detail = client.get("/api/runs/run-real")
        assert detail.status_code == 200
        assert detail.json()["config"] == {"max_length": 1024}
        assert detail.json()["provenance"][0]["id"] == "ds-1"
        assert detail.json()["provenance"][0]["digest"]["value"] == "a" * 64
        assert detail.json()["produced_artifacts"][0]["id"] == "model-one"
        assert detail.json()["lineage"]["lineage_status"] == "complete"
        assert "training complete" in detail.json()["logs"]["text"]
        assert client.get("/api/runs/..%2Foutside").status_code == 404
        assert client.get("/api/runs?limit=invalid").status_code == 422
        assert client.get("/api/runs/run-unknown").status_code == 404
        artifact = client.get("/api/runs/run-real/artifacts/lineage.json")
        assert artifact.status_code == 200 and artifact.json()["run_id"] == "run-real"
        assert client.get("/api/runs/run-real/artifacts/secrets.txt").status_code == 404
    assert output.joinpath("run.json").is_file()


def test_completed_training_run_resolves_model_handoff_through_verified_merge_lineage(tmp_path):
    root = tmp_path.resolve()
    source_id, merge_id = "run-sft-source", "run-sft-merge"
    source_output = {"id": "model-sft-adapter", "kind": "model", "role": "adapter",
        "digest": {"algorithm": "sha256", "status": "available", "value": "a" * 64}}
    source = root / "runs" / "training" / source_id
    put(source / "run.json", {"run_id": source_id, "status": "completed", "stage": "sft"})
    put(source / "lineage.json", {"run_id": source_id, "run_type": "train", "status": "completed",
        "lineage_status": "complete", "inputs": [], "outputs": [source_output]})

    merged_path = root / "runs" / "merges" / merge_id / "merged-model"
    merged_path.mkdir(parents=True)
    (merged_path / "weights.bin").write_bytes(b"merged-model")
    from onegpu_posttrain.lineage import sha256_directory
    merged_digest = sha256_directory(merged_path)
    merged_id = "model-sft-merged"
    merge = root / "runs" / "merges" / merge_id
    put(merge / "lineage.json", {"run_id": merge_id, "run_type": "merge", "status": "completed",
        "inputs": [source_output], "outputs": [{"id": merged_id, "kind": "model", "role": "merged",
            "path": str(merged_path), "digest": {"algorithm": "sha256", "status": "available", "value": merged_digest}}]})
    put(merge / "model_artifacts" / f"{merged_id}.json", {"model_id": merged_id, "kind": "merged", "stage": "sft",
        "path": str(merged_path), "produced_by_run_id": merge_id, "parent_model_ids": ["model-base"],
        "adapter_id": source_output["id"],
        "digest": {"algorithm": "sha256", "status": "available", "value": merged_digest}})

    detail = RunWorkspace(root).run_detail(source_id)

    assert detail["produced_artifacts"][0]["id"] == source_output["id"]
    assert detail["handoff_models"] == [{"model_id": merged_id, "stage": "sft", "kind": "merged",
        "produced_by_run_id": merge_id, "source_run_id": source_id,
        "digest": {"algorithm": "sha256", "status": "available", "value": merged_digest},
        "path_available": True, "registry_count": 1}]
    with TestClient(create_app(root)) as client:
        response = client.get(f"/api/runs/{source_id}")
        assert response.status_code == 200
        assert response.json()["handoff_models"][0]["model_id"] == merged_id

    put(merge / "lineage.json", {"run_id": merge_id, "run_type": "merge", "status": "failed",
        "inputs": [source_output], "outputs": [{"id": merged_id, "kind": "model", "role": "merged",
            "path": str(merged_path), "digest": {"algorithm": "sha256", "status": "available", "value": merged_digest}}]})
    assert RunWorkspace(root).run_detail(source_id)["handoff_models"] == []


def test_runs_api_pagination_is_clamped_and_uses_stable_order(tmp_path):
    root = tmp_path.resolve()
    for index in range(4):
        put(root / f"runs/{index}/run.json", {"run_id": f"run-item-{index}", "status": "failed", "stage": "sft"})
    with TestClient(create_app(root)) as client:
        first = client.get("/api/runs?limit=2&offset=0").json()
        second = client.get("/api/runs?limit=2&offset=2").json()
        assert first["total"] == second["total"] == 4
        assert len(first["runs"]) == len(second["runs"]) == 2
        assert {row["run_id"] for row in first["runs"]}.isdisjoint({row["run_id"] for row in second["runs"]})
        assert client.get("/api/runs?limit=900").json()["limit"] == 200


def test_run_detail_marks_evaluation_without_scores_and_unresolved_parent(tmp_path):
    root = tmp_path.resolve()
    eval_dir = root / "runs/evaluations/run-eval-failed"
    put(eval_dir / "evaluation_run.json", {"run_id": "run-eval-failed", "status": "failed", "parent_run_id": "run-parent-missing"})
    put(eval_dir / "evaluation_lineage.json", {"produced_by_run_id": "run-eval-failed", "inputs": []})
    result = RunWorkspace(root).run_detail("run-eval-failed")
    assert result["status"] == "failed"
    assert result["metrics"] == []
    assert result["parent_run_id"] == "run-parent-missing"
    assert result["parent_available"] is False


def test_assistant_evaluation_evidence_exposes_sanitized_failure_cause(tmp_path):
    run_id = "run-base-choice-accuracy-v1-fixture"
    output = tmp_path / "runs/evaluations" / run_id
    put(output / "evaluation_run.json", {
        "run_id": run_id, "status": "failed", "errors": [
            "ValueError: model type qwen3_5 is not recognized. API_KEY=secret-value "
            + "Transformers suggested updating the installed package because this model is newer."
        ],
    })
    put(output / "evaluation_lineage.json", {
        "run_id": run_id, "status": "failed", "lineage_schema_version": "1.0",
        "protocol_version": "choice-accuracy-v1", "metrics": {},
    })

    refs = WorkspaceEvidenceReader(tmp_path).search(EvidenceQuery(
        kind="general", query=run_id, selected_entity_ids=[run_id], limit=1,
    ))
    evaluation = next(item for item in refs if item.kind == "evaluation")
    failure = next(fact for fact in evaluation.facts if fact.key == "failure_reason")
    assert "qwen3_5" in failure.value
    assert "secret-value" not in failure.value
    assert "updating the installed package" in failure.value
    assert "qwen3_5" in evaluation.summary


def test_training_and_evaluation_details_link_only_recorded_parent_and_show_real_denominators(tmp_path):
    root = tmp_path.resolve()
    put(root / "runs/train/run.json", {"run_id": "run-train", "status": "completed", "stage": "sft"})
    eval_dir = root / "runs/evaluations/run-eval"
    put(eval_dir / "evaluation_run.json", {"run_id": "run-eval", "status": "completed", "parent_run_id": "run-train"})
    put(eval_dir / "evaluation_lineage.json", {"produced_by_run_id": "run-eval", "metrics": {
        "gsm8k.accuracy": {"value": 0.6, "denominator": 5}}})
    adapter = RunWorkspace(root)
    training = adapter.run_detail("run-train")
    evaluation = adapter.run_detail("run-eval")
    assert [row["run_id"] for row in training["evaluations"]] == ["run-eval"]
    assert evaluation["parent_available"] is True
    assert evaluation["metrics"] == [{"name": "gsm8k.accuracy", "metric": "accuracy", "value": 0.6, "denominator": 5}]


def test_evaluation_detail_prefers_launcher_parent_when_cli_record_omits_it(tmp_path):
    root = tmp_path.resolve()
    parent_id, eval_id = "run-launcher-training", "run-launcher-eval"
    put(root / "runs/train/run.json", {"run_id": parent_id, "status": "completed", "stage": "sft"})
    output = root / "runs/evaluations" / eval_id
    put(output / "evaluation_run.json", {"run_id": eval_id, "status": "completed"})
    put(root / ".onegpu-posttrain/launcher/runs" / f"{eval_id}.json", {
        "run_id": eval_id, "run_kind": "evaluation", "status": "completed",
        "parent_run_id": parent_id, "output": output.relative_to(root).as_posix(),
    })
    detail = RunWorkspace(root).run_detail(eval_id)
    assert detail["parent_run_id"] == parent_id
    assert detail["parent_available"] is True


def test_run_detail_preserves_unmeasured_and_partial_metric_statuses_and_denominators(tmp_path):
    root = tmp_path.resolve()
    eval_dir = root / "runs/evaluations/run-status-eval"
    put(eval_dir / "evaluation_run.json", {"run_id": "run-status-eval", "status": "partial"})
    put(eval_dir / "eval.json", {"metric_statuses": {
        "preference.accuracy": {"status": "partial", "value": 0.75, "denominator": 3,
                                 "denominator_unit": "pairs", "reason": "仅完成 3/4 pairs"},
        "gsm8k.accuracy": {"status": "not_measured", "value": None, "denominator": 0,
                            "reason": "未运行共享任务评测"},
    }})
    detail = RunWorkspace(root).run_detail("run-status-eval")
    by_name = {item["name"]: item for item in detail["metrics"]}
    assert by_name["preference.accuracy"]["status"] == "partial"
    assert by_name["preference.accuracy"]["denominator"] == 3
    assert by_name["preference.accuracy"]["denominator_unit"] == "pairs"
    assert by_name["gsm8k.accuracy"]["status"] == "not_measured"
    assert by_name["gsm8k.accuracy"]["value"] is None


def test_runs_endpoints_are_read_only_and_reject_traversal(tmp_path):
    root = tmp_path.resolve()
    put(root / "runs/x/run.json", {"run_id": "run-x", "status": "failed"})
    with TestClient(create_app(root)) as client:
        assert client.get("/api/runs").status_code == 200
        assert client.get("/api/runs/%2e%2e%2fsecret").status_code == 404
        assert client.post("/api/runs", json={}).status_code == 405
