import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from onegpu_posttrain import cli
from onegpu_posttrain.lineage import ArtifactRef, Digest, sha256_directory, sha256_json


def _args(tmp_path, *, dry_run=False):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "weights.bin").write_bytes(b"fixture-model")
    return Namespace(
        config="recipes/smoke/00-base-eval.yaml", model=str(model), revision=None,
        output=str(tmp_path / "eval-out"), run_label="base-fixture", stage="base",
        limit=3, dry_run=dry_run, internal_holdout=None, internal_limit=20,
        max_new_tokens=32, domain_holdout=None, domain_limit=20,
        ppl_max_length=128, dtype="float16",
    )


def test_completed_evaluation_writes_separate_run_and_eval_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_evalscope", lambda cmd, dry_run=False: {"returncode": 0})
    monkeypatch.setattr(cli, "collect_evalscope_reports", lambda path: {
        "gsm8k": {"score": 0.5, "accuracy": 0.5, "samples": 3},
        "math_500": {"score": 0.6, "accuracy": 0.6, "samples": 3}})
    args = _args(tmp_path)

    assert cli.cmd_run_model_eval(args) == 0
    result = json.loads((tmp_path / "eval-out" / "eval.json").read_text(encoding="utf-8"))
    lineage = json.loads((tmp_path / "eval-out" / "evaluation_lineage.json").read_text(encoding="utf-8"))
    run = json.loads((tmp_path / "eval-out" / "evaluation_run.json").read_text(encoding="utf-8"))
    assert result["run_label"] == "base-fixture"
    assert run["run_id"].startswith("run-base-fixture-")
    assert lineage["produced_by_run_id"] == run["run_id"]
    assert lineage["eval_id"].startswith("eval-")
    assert lineage["protocol_version"] == "math-eval-v1"
    assert lineage["config_sha256"] == sha256_json(result["resolved_evaluation"])
    assert result["resolved_evaluation"]["internal_math"] is None
    assert result["resolved_evaluation"]["protocol_version"] == "math-eval-v1"
    assert result["resolved_evaluation"]["profile_id"] == "base-math-eval-v1"
    assert lineage["metrics"]["evalscope.gsm8k.accuracy"] == {"value": 0.5, "denominator": 3}
    assert result["metric_statuses"]["gsm8k.accuracy"] == {
        "status": "measured", "value": 0.5, "denominator": 3,
        "denominator_unit": "samples", "reason": "使用评测报告中的实际完成样本数。",
    }
    assert result["metric_statuses"]["internal_holdout.accuracy"]["status"] == "not_measured"
    # A local checkpoint is hashed; an unknown HF revision remains explicitly absent.
    assert run["inputs"][0]["digest"]["status"] == "available"
    assert run["inputs"][0]["revision"] is None
    assert run["inputs"][1]["availability"] == "unavailable"


def test_evaluation_dry_run_does_not_claim_execution_lineage(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_evalscope", lambda cmd, dry_run=False: {"dry_run": True})
    args = _args(tmp_path, dry_run=True)

    assert cli.cmd_run_model_eval(args) == 0
    output = tmp_path / "eval-out"
    assert (output / "eval.json").is_file()
    assert not (output / "evaluation_lineage.json").exists()
    assert not (output / "evaluation_run.json").exists()


def test_failed_evaluation_records_failed_run_without_fake_metrics(tmp_path, monkeypatch):
    def fail(cmd, dry_run=False):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(cli, "run_evalscope", fail)
    args = _args(tmp_path)

    assert cli.cmd_run_model_eval(args) == 4
    output = tmp_path / "eval-out"
    lineage = json.loads((output / "evaluation_lineage.json").read_text(encoding="utf-8"))
    run = json.loads((output / "evaluation_run.json").read_text(encoding="utf-8"))
    assert lineage["status"] == "failed"
    assert run["status"] == "failed"
    assert run["outputs"] == []
    assert run["errors"] == ["fixture failure"]


def test_optional_evaluation_failure_is_recorded_as_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_evalscope", lambda cmd, dry_run=False: {"returncode": 0})
    monkeypatch.setattr(cli, "collect_evalscope_reports", lambda path: {
        "gsm8k": {"score": 0.5, "accuracy": 0.5, "samples": 2},
        "math_500": {"score": 0.6, "accuracy": 0.6, "samples": 2}})
    monkeypatch.setattr(cli, "evaluate_cpt_perplexity", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("PPL fixture failure")))
    args = _args(tmp_path)
    args.domain_holdout = str(tmp_path / "domain.jsonl")

    assert cli.cmd_run_model_eval(args) == 4
    output = tmp_path / "eval-out"
    lineage = json.loads((output / "evaluation_lineage.json").read_text(encoding="utf-8"))
    run = json.loads((output / "evaluation_run.json").read_text(encoding="utf-8"))
    assert lineage["status"] == "partial"
    assert run["status"] == "partial"
    assert any("PPL fixture failure" in error for error in run["errors"])
    result = json.loads((output / "eval.json").read_text(encoding="utf-8"))
    assert result["metric_statuses"]["domain_ppl.perplexity"]["status"] == "not_measured"


def test_cpt_domain_ppl_uses_actual_predicted_token_denominator(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_evalscope", lambda cmd, dry_run=False: {"returncode": 0})
    monkeypatch.setattr(cli, "collect_evalscope_reports", lambda path: {
        "gsm8k": {"score": 0.5, "accuracy": 0.5, "samples": 4},
        "math_500": {"score": 0.25, "accuracy": 0.25, "samples": 4},
    })
    monkeypatch.setattr(cli, "evaluate_cpt_perplexity", lambda **kwargs: SimpleNamespace(to_dict=lambda: {
        "samples": 3, "predicted_tokens": 77, "total_nll": 90.0, "mean_nll": 90 / 77,
        "perplexity": 3.22, "truncated_samples": 1, "wall_seconds": 1.0,
    }))
    args = _args(tmp_path)
    args.stage = "cpt"
    args.domain_holdout = str(tmp_path / "cpt-holdout.jsonl")

    assert cli.cmd_run_model_eval(args) == 0
    output = tmp_path / "eval-out"
    result = json.loads((output / "eval.json").read_text(encoding="utf-8"))
    lineage = json.loads((output / "evaluation_lineage.json").read_text(encoding="utf-8"))
    assert result["resolved_evaluation"]["profile_id"] == "cpt-math-v1"
    assert result["metric_statuses"]["domain_ppl.perplexity"]["status"] == "measured"
    assert result["metric_statuses"]["domain_ppl.perplexity"]["denominator"] == 77
    assert result["metric_statuses"]["domain_ppl.perplexity"]["denominator_unit"] == "predicted_tokens"
    assert lineage["metrics"]["domain_perplexity.perplexity"] == {"value": 3.22, "denominator": 77}
    holdout = next(item for item in lineage["dataset_artifacts"] if item["path"] == str(args.domain_holdout))
    assert holdout["role"] == "cpt_holdout"
    run = json.loads((output / "evaluation_run.json").read_text(encoding="utf-8"))
    run_holdout = next(item for item in run["inputs"] if item.get("path") == str(args.domain_holdout))
    assert run_holdout["role"] == "cpt_holdout"


def test_existing_evaluation_output_is_not_overwritten(tmp_path, monkeypatch):
    args = _args(tmp_path)
    output = tmp_path / "eval-out"
    output.mkdir()
    (output / "eval.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "run_evalscope", lambda cmd, dry_run=False: {})

    with pytest.raises(FileExistsError):
        cli.cmd_run_model_eval(args)


def test_eval_command_splits_per_benchmark_limits_and_balances_math_levels(tmp_path):
    eval_config = {
        "benchmarks": ["gsm8k", "math_500"],
        "benchmark_execution": [
            {"dataset": "gsm8k", "limit": 5},
            {"dataset": "math_500", "limit": 1,
             "dataset_args": {"subset_list": ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]}},
        ],
        "generation": {"max_tokens": 64},
    }
    commands = cli._eval_commands(cfg={"seed": 42}, model="local-model", revision=None,
                                  eval_cfg=eval_config, output=tmp_path, limit=5)
    assert len(commands) == 2
    gsm = commands[0]
    math = commands[1]
    assert gsm[gsm.index("--datasets") + 1] == "gsm8k"
    assert gsm[gsm.index("--limit") + 1] == "5"
    assert math[math.index("--datasets") + 1] == "math_500"
    assert math[math.index("--limit") + 1] == "1"
    assert json.loads(math[math.index("--dataset-args") + 1]) == {
        "math_500": {"subset_list": ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]}}


def test_evalscope_report_collector_reads_dataset_scoped_workdirs(tmp_path):
    from onegpu_posttrain.evaluation.evalscope_reports import collect_evalscope_reports

    gsm_report = tmp_path / "gsm8k" / "reports" / "gsm8k" / "report.json"
    math_report = tmp_path / "math_500" / "reports" / "math_500" / "report.json"
    for path, dataset in ((gsm_report, "gsm8k"), (math_report, "math_500")):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"dataset_name": dataset, "score": 0.5, "num_samples": 5}), encoding="utf-8")
    reports = collect_evalscope_reports(tmp_path)
    assert reports["gsm8k"]["samples"] == 5
    assert reports["math_500"]["samples"] == 5


def test_evaluation_lineage_links_registered_merged_model_producer(tmp_path, monkeypatch):
    args = _args(tmp_path)
    model = tmp_path / "model"
    digest = sha256_directory(model)
    model_id = f"model-{digest[:20]}"
    registry = tmp_path / "model_artifacts" / f"{model_id}.json"
    registry.parent.mkdir()
    registry.write_text(json.dumps({"model_id": model_id, "path": str(model), "kind": "merged",
                                    "digest": {"value": digest}, "produced_by_run_id": "run-merge-parent-123"}), encoding="utf-8")
    monkeypatch.setattr(cli, "run_evalscope", lambda cmd, dry_run=False: {"returncode": 0})
    monkeypatch.setattr(cli, "collect_evalscope_reports", lambda path: {
        "gsm8k": {"score": 0.4, "accuracy": 0.4, "samples": 2},
        "math_500": {"score": 0.8, "accuracy": 0.8, "samples": 3}})
    assert cli.cmd_run_model_eval(args) == 0
    lineage = json.loads((tmp_path / "eval-out" / "evaluation_lineage.json").read_text(encoding="utf-8"))
    assert lineage["model_id"] == model_id
    assert lineage["model_produced_by_run_id"] == "run-merge-parent-123"
    assert lineage["status"] == "completed"


def test_registered_model_producer_resolves_workspace_relative_registry_path(tmp_path):
    model = tmp_path / "runs" / "sft" / "run-sft-fixture" / "merged" / "model"
    model.mkdir(parents=True)
    (model / "weights.bin").write_bytes(b"fixture-model")
    digest = sha256_directory(model)
    model_id = f"model-{digest[:20]}"
    registry = tmp_path / "runs" / "sft" / "run-sft-fixture" / "model_artifacts" / f"{model_id}.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({
        "model_id": model_id,
        "path": "runs/sft/run-sft-fixture/merged/model",
        "digest": {"value": digest},
        "produced_by_run_id": "run-sft-fixture",
    }), encoding="utf-8")

    model_ref = ArtifactRef(
        id=model_id,
        kind="model",
        path=str(model),
        digest=Digest(value=digest, status="available"),
    )
    assert cli._registered_model_producer(model, model_ref) == "run-sft-fixture"


def test_missing_benchmark_report_is_partial_and_never_fabricates_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_evalscope", lambda cmd, dry_run=False: {"returncode": 0})
    monkeypatch.setattr(cli, "collect_evalscope_reports", lambda path: {"gsm8k": {"score": 0.4, "samples": 2}})
    args = _args(tmp_path)
    assert cli.cmd_run_model_eval(args) == 4
    output = tmp_path / "eval-out"
    result = json.loads((output / "eval.json").read_text(encoding="utf-8"))
    lineage = json.loads((output / "evaluation_lineage.json").read_text(encoding="utf-8"))
    assert lineage["status"] == "partial"
    assert "math_500" not in result["evalscope_reports"]
    assert any("missing benchmark score: math_500" in error for error in json.loads((output / "evaluation_run.json").read_text(encoding="utf-8"))["errors"])


def test_launcher_stream_mode_uses_supervisor_run_id_and_preserves_eval_id(tmp_path, monkeypatch):
    seen = {}

    def stream(command, dry_run=False, stream_output=False):
        seen["stream_output"] = stream_output
        return {"returncode": 0}

    monkeypatch.setattr(cli, "run_evalscope", stream)
    monkeypatch.setattr(cli, "collect_evalscope_reports", lambda path: {
        "gsm8k": {"score": 0.5, "accuracy": 0.5, "samples": 5},
        "math_500": {"score": 0.4, "accuracy": 0.4, "samples": 5}})
    args = _args(tmp_path)
    args.run_id = "run-launcher-eval-fixture"
    args.stream_output = True
    assert cli.cmd_run_model_eval(args) == 0
    run = json.loads((tmp_path / "eval-out" / "evaluation_run.json").read_text(encoding="utf-8"))
    lineage = json.loads((tmp_path / "eval-out" / "evaluation_lineage.json").read_text(encoding="utf-8"))
    assert seen["stream_output"] is True
    assert run["run_id"] == lineage["produced_by_run_id"] == args.run_id
    assert lineage["eval_id"].startswith("eval-")

