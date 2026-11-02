from __future__ import annotations

import json

import pytest
import yaml

from onegpu_posttrain.reporting.evaluation_matrix import build_evaluation_matrix


def _put(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _put_eval_sources(root, name, *, status="completed", run_id=None, eval_id=None,
                     lineage_run_id=None, lineage_model_id=None, output_path="eval.json"):
    run_id = run_id or f"eval-{name.lower()}"
    eval_id = eval_id or f"id-{name.lower()}"
    model_id = f"model-{name.lower()}"
    _put(root / name / "evaluation_run.json", {
        "run_id": run_id, "run_type": "evaluate", "status": status,
        "inputs": [{"id": model_id, "kind": "model", "path": f"models/{name}"}],
        "outputs": [{"id": eval_id, "kind": "evaluation", "path": output_path}],
    })
    _put(root / name / "evaluation_lineage.json", {
        "eval_id": eval_id, "produced_by_run_id": lineage_run_id or run_id,
        "model_id": lineage_model_id or model_id,
        "dataset_ids": ["benchmark-gsm8k", "benchmark-math_500", "ds-holdout"],
        "protocol_version": "math-eval-v1", "config_sha256": "same",
        "status": {"completed": "completed", "partial": "partial", "failed": "failed"}[status],
    })


def test_matrix_preserves_statuses_missing_metrics_and_lineage_consistency(tmp_path):
    models = []
    for name in ("M2", "DPO", "SimPO", "GRPO-standard", "GRPO-lowmem"):
        eval_path = tmp_path / name / "eval.json"
        models.append({"model_id": name, "model_path": f"models/{name}", "eval_path": str(eval_path)})

    _put(tmp_path / "M2" / "eval.json", {
        "evalscope_reports": {
            "gsm8k": {"score": 0.6, "report_path": str(tmp_path / "M2" / "gsm8k.json")},
            "math_500": {"score": 0.4, "report_path": str(tmp_path / "M2" / "math_500.json")},
        },
        "run_id": "eval-m2", "eval_id": "id-m2",
        "internal_math": {"samples": 20, "accuracy": 0.2},
        "domain_perplexity": {"perplexity": 6.5},
    })
    _put(tmp_path / "M2" / "gsm8k.json", {"metrics": [{"num": 5, "score": 0.6}]})
    _put(tmp_path / "M2" / "math_500.json", {"metrics": [{"num": 25, "score": 0.4}]})
    _put_eval_sources(tmp_path, "M2", run_id="eval-m2", eval_id="id-m2")
    _put(tmp_path / "DPO" / "eval.json", {
        "run_id": "eval-dpo", "eval_id": "id-dpo",
        "evalscope_reports": {"gsm8k": {"score": 0.8}}, "internal_math": {"samples": 20, "accuracy": 0.25},
    })
    _put_eval_sources(tmp_path, "DPO", status="partial", run_id="eval-dpo", eval_id="id-dpo")
    _put(tmp_path / "SimPO" / "eval.json", {})
    _put_eval_sources(tmp_path, "SimPO", status="failed", run_id="eval-simpo", eval_id="id-simpo")

    config = tmp_path / "matrix.yaml"
    config.write_text(yaml.safe_dump({
        "baseline_model_id": "M2",
        "expected_protocol_version": "math-eval-v1",
        "expected_metrics": ["gsm8k", "math_500", "internal_acc", "domain_ppl"],
        "models": models,
    }), encoding="utf-8")

    result = build_evaluation_matrix(config, output_dir=tmp_path / "matrix")

    assert [m["status"] for m in result["models"]] == ["completed", "partial", "failed", "missing", "missing"]
    assert result["models"][1]["metrics"]["math_500"] is None
    assert abs(result["models"][1]["delta_vs_baseline"]["gsm8k"] - 0.2) < 1e-9
    assert result["models"][3]["delta_vs_baseline"]["internal_acc"] is None
    assert result["comparability"]["status"] == "partial_or_inconsistent"
    assert result["models"][0]["lineage"]["produced_by_run_id"] == "eval-m2"
    assert result["models"][0]["provenance"]["status"] == "verified"
    assert result["models"][0]["samples"]["gsm8k"]["observed"] == 5
    assert result["models"][0]["samples"]["math_500"]["observed"] == 25
    assert (tmp_path / "matrix" / "evaluation_matrix.md").is_file()


@pytest.mark.parametrize(("changed", "expected_code"), [
    ("run_id", "run_id_mismatch"),
    ("eval_id", "metrics_eval_id_mismatch"),
    ("model_id", "model_id_mismatch"),
    ("model_path", "model_path_mismatch"),
    ("output_path", "evaluation_output_mismatch"),
])
def test_matrix_marks_contradictory_source_identity_without_dropping_metrics(tmp_path, changed, expected_code):
    eval_path = tmp_path / "M2" / "eval.json"
    _put(eval_path, {
        "run_id": "eval-m2", "eval_id": "id-m2",
        "evalscope_reports": {"gsm8k": {"score": 0.5}},
        "internal_math": {"accuracy": 0.25}, "domain_perplexity": {"perplexity": 4.0},
    })
    kwargs = {}
    if changed == "run_id":
        kwargs["lineage_run_id"] = "eval-other"
    elif changed == "eval_id":
        kwargs["eval_id"] = "id-other"
    elif changed == "model_id":
        kwargs["lineage_model_id"] = "model-other"
    elif changed == "model_path":
        kwargs["model_path"] = "models/other"
    elif changed == "output_path":
        kwargs["output_path"] = "elsewhere/eval.json"
    spec_model_path = kwargs.pop("model_path", "models/M2")
    _put_eval_sources(tmp_path, "M2", **kwargs)
    config = tmp_path / "matrix.yaml"
    config.write_text(yaml.safe_dump({
        "baseline_model_id": "M2", "expected_metrics": ["gsm8k", "internal_acc", "domain_ppl"],
        "models": [{"model_id": "M2", "model_path": spec_model_path, "eval_path": str(eval_path)}],
    }), encoding="utf-8")

    result = build_evaluation_matrix(config, output_dir=tmp_path / "out")

    model = result["models"][0]
    assert model["status"] == "partial"
    assert model["metrics"]["gsm8k"] == 0.5
    assert model["provenance"]["status"] == "mismatch"
    assert expected_code in {issue["code"] for issue in model["provenance"]["issues"]}
    assert result["comparability"]["status"] == "partial_or_inconsistent"


def test_matrix_keeps_legacy_metrics_but_marks_provenance_incomplete(tmp_path):
    eval_path = tmp_path / "M2" / "eval.json"
    _put(eval_path, {"evalscope_reports": {"gsm8k": {"score": 0.5}}})
    config = tmp_path / "matrix.yaml"
    config.write_text(yaml.safe_dump({"baseline_model_id": "M2", "expected_metrics": ["gsm8k"],
        "models": [{"model_id": "M2", "model_path": "models/M2", "eval_path": str(eval_path)}]}), encoding="utf-8")

    result = build_evaluation_matrix(config, output_dir=tmp_path / "out")

    model = result["models"][0]
    assert model["metrics"]["gsm8k"] == 0.5
    assert model["status"] == "partial"
    assert model["provenance"]["status"] == "incomplete"
    assert result["comparability"]["status"] == "partial_or_inconsistent"


def test_matrix_flags_protocol_config_or_dataset_drift(tmp_path):
    models = []
    for name, protocol in (("M2", "math-eval-v1"), ("DPO", "math-eval-v2")):
        path = tmp_path / name / "eval.json"
        models.append({"model_id": name, "model_path": name, "eval_path": str(path)})
        _put(path, {
            "evalscope_reports": {"gsm8k": {"score": 0.5}, "math_500": {"score": 0.4}},
            "internal_math": {"accuracy": 0.2},
        })
        _put(path.with_name("evaluation_run.json"), {"run_id": f"eval-{name}", "status": "completed"})
        _put(path.with_name("evaluation_lineage.json"), {
            "eval_id": f"id-{name}", "produced_by_run_id": f"eval-{name}", "model_id": f"model-{name}",
            "dataset_ids": ["benchmark-gsm8k", "benchmark-math_500", f"holdout-{name}"],
            "protocol_version": protocol, "config_sha256": f"hash-{name}", "status": "completed",
        })
    config = tmp_path / "matrix.yaml"
    config.write_text(yaml.safe_dump({
        "baseline_model_id": "M2", "expected_protocol_version": "math-eval-v1",
        "expected_metrics": ["gsm8k", "math_500", "internal_acc"], "models": models,
    }), encoding="utf-8")

    result = build_evaluation_matrix(config, output_dir=tmp_path / "out")

    checks = result["comparability"]["protocol_consistency"]
    assert not checks["protocol_version_consistent"]
    assert not checks["config_hash_consistent"]
    assert not checks["dataset_ids_consistent"]
