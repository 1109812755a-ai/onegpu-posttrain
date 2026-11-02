import json
from io import BytesIO
from types import SimpleNamespace

from onegpu_posttrain.data.dataset_import import register_jsonl
from onegpu_posttrain.data.objective_evaluation_import import (
    OBJECTIVE_EVALUATION_FORMAT,
    convert_objective_evaluation_jsonl,
)
from onegpu_posttrain.data.registry import DataSource
from onegpu_posttrain.lineage import ArtifactRef, Digest, ModelLineage, RunRecord, sha256_directory, sha256_file
from onegpu_posttrain.ui.stage_evaluation import StageEvaluationAdapter, StageEvaluationRequest
from onegpu_posttrain.ui.stage_profiles import stage_profile_catalog
from onegpu_posttrain.ui.training_preflight import _inspect_dataset


def _register_eval_dataset(workspace, *, supplied_id=None):
    row = {"questionType": "single_choice", "question": "1+1?", "options": '["1", "2", "3"]',
           "correctAnswer": "B"}
    if supplied_id:
        row["id"] = supplied_id
    source = (json.dumps(row, ensure_ascii=False) + "\n").encode()
    conversion = convert_objective_evaluation_jsonl(BytesIO(source))
    origin = DataSource(source_id="upload-pending", kind="local_upload", name="eval suite",
                        revision_status="not_provided", license_status="not_provided", declared_by="user")
    result = register_jsonl(
        BytesIO(conversion.canonical_bytes), workspace=workspace,
        output_root=workspace / "data" / "processed" / "user-upload", stage="eval", origin=origin,
        raw_source_bytes=source, source_format=OBJECTIVE_EVALUATION_FORMAT,
        converter_config={"source_format": OBJECTIVE_EVALUATION_FORMAT,
                          "converter_version": "objective-choice-converter-v1",
                          "source_record_count": 1, "output_record_count": 1},
        registration_metadata={"dataset_role": "evaluation_only", "evaluation_protocol": "choice-accuracy-v1"},
    )
    return result, source


def _bind_training_run(adapter, workspace, *, training_dataset_id="ds-training", training_hash="0" * 64):
    model_dir = workspace / "models" / "merged"
    model_dir.mkdir(parents=True)
    (model_dir / "weights.bin").write_bytes(b"fixture model weights")
    model = ModelLineage(model_id="model-fixture", kind="merged", path=str(model_dir),
                         digest=Digest(value=sha256_directory(model_dir), status="available"),
                         produced_by_run_id="run-train-fixture", stage="sft")
    run = RunRecord(
        run_id="run-train-fixture", run_type="train", status="completed", started_at="2026-01-01T00:00:00Z",
        inputs=[ArtifactRef(id=training_dataset_id, kind="dataset", role="train",
                            digest=Digest(value=training_hash, status="available"))],
    )
    adapter._model_for_training_run = lambda run_id: (model, run)


def test_choice_eval_preflight_selects_eval_only_dataset_and_emits_fixed_command(tmp_path):
    registered, _ = _register_eval_dataset(tmp_path)
    manifest = next((registered.path / "manifests").glob("*.yaml"))
    from onegpu_posttrain.data.manifest import load_manifest

    manifest_data = load_manifest(manifest)
    adapter = StageEvaluationAdapter(tmp_path)
    _bind_training_run(adapter, tmp_path, training_hash="f" * 64)
    request = StageEvaluationRequest(training_run_id="run-train-fixture", profile_id="choice-accuracy-v1",
                                     dataset_id=manifest_data.dataset_id)
    result = adapter.preflight(request)
    assert result["ok"] is True
    assert result["dataset"]["role"] == "evaluation_only"
    assert result["dataset"]["record_count"] == 1
    assert result["dataset"]["comparability_key"]
    command = adapter.build_command(request, result["evaluation_run_id"])
    assert "run-registered-choice-eval" in command
    assert "--dataset-id" in command and manifest_data.dataset_id in command


def test_choice_eval_preflight_blocks_dataset_id_or_hash_overlap(tmp_path):
    registered, _ = _register_eval_dataset(tmp_path, supplied_id="shared-data-id")
    manifest = next((registered.path / "manifests").glob("*.yaml"))
    from onegpu_posttrain.data.manifest import load_manifest

    manifest_data = load_manifest(manifest)
    dataset_hash = manifest_data.hashes["canonical_records_sha256"]
    adapter = StageEvaluationAdapter(tmp_path)
    _bind_training_run(adapter, tmp_path, training_dataset_id=manifest_data.dataset_id,
                       training_hash=dataset_hash)
    result = adapter.preflight(StageEvaluationRequest(
        training_run_id="run-train-fixture", profile_id="choice-accuracy-v1",
        dataset_id=manifest_data.dataset_id,
    ))
    assert result["ok"] is False
    assert any("泄漏" in blocker for blocker in result["blockers"])


def test_evaluation_only_dataset_is_rejected_by_training_preflight(tmp_path):
    registered, _ = _register_eval_dataset(tmp_path)
    manifest_path = next((registered.path / "manifests").glob("*.yaml"))
    profile = next(item for item in stage_profile_catalog(tmp_path).profiles if item.stage == "sft")
    import pytest

    with pytest.raises(ValueError, match="数据阶段为 eval"):
        _inspect_dataset(tmp_path, manifest_path, __import__(
            "onegpu_posttrain.data.manifest", fromlist=["load_manifest"]
        ).load_manifest(manifest_path), profile)


def test_objective_converter_rejects_duplicate_explicit_ids_with_line_evidence():
    row = {"id": "same", "questionType": "single_choice", "question": "q", "options": '["a","b"]',
           "correctAnswer": "A"}
    payload = (json.dumps(row) + "\n" + json.dumps(row) + "\n").encode()
    converted = convert_objective_evaluation_jsonl(BytesIO(payload))
    assert converted.source_record_count == 2
    assert converted.output_record_count == 1
    assert converted.report()["errors"][0]["code"] == "duplicate_record_id"
    assert converted.report()["errors"][0]["line"] == 2


def test_registered_choice_eval_persists_suite_metrics_and_lineage(tmp_path, monkeypatch, capsys):
    from onegpu_posttrain import cli
    from onegpu_posttrain.evaluation import choice_accuracy
    from onegpu_posttrain.evaluation.choice_accuracy import choice_comparability_key

    registered, _ = _register_eval_dataset(tmp_path)
    manifest_path = next((registered.path / "manifests").glob("*.yaml"))
    from onegpu_posttrain.data.manifest import load_manifest

    manifest = load_manifest(manifest_path)
    model_dir = tmp_path / "models" / "merged"
    model_dir.mkdir(parents=True)
    (model_dir / "weights.bin").write_bytes(b"fixture model weights")
    model = ModelLineage(model_id="model-fixture", kind="merged", path=str(model_dir),
                         digest=Digest(value=sha256_directory(model_dir), status="available"),
                         produced_by_run_id="run-merge-fixture", stage="sft")
    training_run = RunRecord(
        run_id="run-train-fixture", run_type="train", status="completed", started_at="2026-01-01T00:00:00Z",
        inputs=[ArtifactRef(id="ds-training", kind="dataset", role="train",
                            digest=Digest(value="0" * 64, status="available"))],
    )
    monkeypatch.setattr(StageEvaluationAdapter, "_model_for_training_run",
                        lambda self, run_id: (model, training_run))
    monkeypatch.setattr(StageEvaluationAdapter, "_model_path", lambda self, item: model_dir)
    monkeypatch.setattr(choice_accuracy, "evaluate_registered_choice_dataset", lambda **kwargs: {
        "protocol": "choice-accuracy-v1", "sample_count": 1,
        "metrics": {"single_choice": {"correct": 1, "denominator": 1, "parse_failures": 0,
                                       "accuracy": 1.0, "status": "measured"},
                    "overall": {"correct": 1, "denominator": 1, "parse_failures": 0,
                                "accuracy": 1.0, "status": "measured"}},
    })
    args = SimpleNamespace(workspace=str(tmp_path), run_id="run-choice-fixture",
                           output="runs/evaluations/run-choice-fixture", training_run_id=training_run.run_id,
                           profile_id="choice-accuracy-v1", dataset_id=manifest.dataset_id)

    assert cli.cmd_run_registered_choice_eval(args) == 0
    payload = json.loads((tmp_path / args.output / "eval.json").read_text(encoding="utf-8"))
    evaluation_lineage = json.loads((tmp_path / args.output / "evaluation_lineage.json").read_text(encoding="utf-8"))
    evaluation_run = json.loads((tmp_path / args.output / "evaluation_run.json").read_text(encoding="utf-8"))
    assert payload["evaluation_suite"] == {
        "dataset_id": manifest.dataset_id,
        "dataset_sha256": manifest.hashes["canonical_records_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "sample_count": 1,
        "protocol_version": "choice-accuracy-v1",
    }
    assert payload["comparability"]["key"] == choice_comparability_key(
        manifest.hashes["canonical_records_sha256"], 1)
    assert payload["metric_statuses"]["choice_accuracy.single_choice"]["correct"] == 1
    assert evaluation_lineage["dataset_artifacts"][0]["role"] == "evaluation_only"
    assert evaluation_lineage["protocol_version"] == "choice-accuracy-v1"
    assert evaluation_run["run_type"] == "evaluate" and evaluation_run["status"] == "completed"
    capsys.readouterr()
