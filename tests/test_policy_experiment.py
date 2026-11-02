import json
from datetime import datetime, timedelta, timezone

import pytest
from policy_support import policy_fixture

from onegpu_posttrain.data.io import write_jsonl
from onegpu_posttrain.data.policy import digest_json
from onegpu_posttrain.data.policy_build import build_policy_artifacts
from onegpu_posttrain.data.policy_experiment import (
    DataEffectExperiment,
    compare_experiment,
    definition_issues,
    register_experiment,
)
from onegpu_posttrain.lineage import sha256_file
from onegpu_posttrain.reporting.data_report import write_report


def experiment_fixture(root):
    a, source, _ = policy_fixture(root / "inputs")
    b = a.model_copy(deep=True)
    b.rules.correctness = "report_only"
    data_a = build_policy_artifacts(a, source, output_dir=root / "a")
    data_b = build_policy_artifacts(b, source, output_dir=root / "b")
    (root / "recipe.yaml").write_text("training: {max_length: 64, max_steps: 1}\n", encoding="utf-8")
    write_jsonl(root / "eval.jsonl", [{"id": str(i), "problem": f"Evaluation-only fixture question {i}."} for i in range(10)])
    plan = DataEffectExperiment.model_validate({
        "experiment_id": "exp-fixture-policy", "question": "CPU simulated evidence, not model results",
        "evidence_kind": "fixture", "experiment_type": "controlled_quality_ablation",
        "independent_variable": "rules.correctness", "source_pool_id": data_a["report"]["source_pool"]["pool_id"],
        "split_snapshot_id": source.split.snapshot_id,
        "a": {"policy": a.model_dump(), "manifest": "a/manifests/sft-policy-v1.yaml", "budget": {"unit": "examples", "value": 3}},
        "b": {"policy": b.model_dump(), "manifest": "b/manifests/sft-policy-v1.yaml", "budget": {"unit": "examples", "value": 3}},
        "controls": {"parent_model_id": "model-fixture-parent", "parent_model_sha256": "1" * 64,
                     "recipe_sha256": sha256_file(root / "recipe.yaml"), "training": {"max_length": 64, "max_steps": 1}, "seeds": [42]},
        "evaluation": {"dataset_ids": ["ds-fixture-eval"], "dataset_sha256": {"ds-fixture-eval": sha256_file(root / "eval.jsonl")},
                       "roles": {"ds-fixture-eval": "diagnostic_dev"}, "protocol_version": "fixture-eval-v1", "config_sha256": "2" * 64},
    })
    registration = register_experiment(plan, root=root, output=root / "registered.json")
    selections = {"a": [], "b": []}
    started = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()
    for label, data, score in (("a", data_a, 0.5), ("b", data_b, 0.6)):
        run_id, model_id, eval_id, eval_run_id = f"run-{label}-fixture", f"model-{label}", f"eval-{label}", f"run-eval-{label}"
        resolved = {"training": {**plan.controls.training, "seed": 42}}
        write_report(root / label / "run/resolved_config.json", resolved)
        write_report(root / label / "run/lineage.json", {
            "lineage_schema_version": "1.0", "run_id": run_id, "run_type": "train", "status": "completed",
            "run_label": "simulated-fixture", "experiment_id": plan.experiment_id, "started_at": started,
            "recipe_path": "recipe.yaml", "resolved_config_sha256": digest_json(resolved),
            "inputs": [{"id": data["dataset_id"], "kind": "dataset", "role": "train",
                        "digest": {"value": data["report"]["source_pool"]["content_sha256"], "status": "available"}},
                       {"id": "model-fixture-parent", "kind": "model", "role": "parent",
                        "digest": {"value": "1" * 64, "status": "available"}}],
            "outputs": [{"id": model_id, "kind": "model"}],
        })
        # The consumed dataset digest is the canonical output, not its raw source hash.
        payload = json.loads((root / label / "run/lineage.json").read_text(encoding="utf-8"))
        payload["inputs"][0]["digest"]["value"] = sha256_file(root / label / "canonical/sft.jsonl")
        write_report(root / label / "run/lineage.json", payload)
        write_report(root / label / "eval/evaluation_lineage.json", {
            "lineage_schema_version": "1.0", "eval_id": eval_id, "produced_by_run_id": eval_run_id,
            "model_id": model_id, "dataset_ids": ["ds-fixture-eval"], "protocol_version": "fixture-eval-v1",
            "config_sha256": "2" * 64, "status": "completed",
            "metrics": {"internal_math.accuracy": {"value": score, "denominator": 10}},
        })
        write_report(root / label / "eval/evaluation_run.json", {
            "lineage_schema_version": "1.0", "run_id": eval_run_id, "run_type": "evaluate", "status": "completed",
            "started_at": started, "inputs": [{"id": model_id, "kind": "model"},
                {"id": "ds-fixture-eval", "kind": "dataset", "digest": {
                    "value": sha256_file(root / "eval.jsonl"), "status": "available"}}],
        })
        selections[label].append({"run_id": run_id, "eval_id": eval_id})
    return plan, registration, selections


def test_comparison_only_uses_explicit_valid_lineage_and_never_mutates(tmp_path):
    plan, registration, selections = experiment_fixture(tmp_path)
    before = {p: sha256_file(p) for p in tmp_path.rglob("*.json")}
    empty = compare_experiment(plan, root=tmp_path)
    assert empty["comparison_status"] == "NOT_EVALUATED"
    assert empty["model_metrics"][0]["delta"] is None
    report = compare_experiment(plan, root=tmp_path, selections=selections, registration=registration)
    assert report["comparison_status"] == "COMPARABLE", report["compatibility_issues"]
    assert report["model_metrics"][0]["delta"] == pytest.approx(0.1)
    assert report["model_metrics"][0]["per_seed"][0]["denominators"] == [10, 10]
    assert report["evidence_kind"] == "fixture"
    assert before == {p: sha256_file(p) for p in before}


@pytest.mark.parametrize("mutation", ["parent", "dataset", "protocol", "denominator", "status", "budget", "model", "recipe", "seed"])
def test_incompatible_evidence_keeps_values_without_delta(tmp_path, mutation):
    plan, registration, selections = experiment_fixture(tmp_path)
    run_path = tmp_path / "b/run/lineage.json"
    eval_path = tmp_path / "b/eval/evaluation_lineage.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    if mutation == "parent":
        run["inputs"][1]["id"] = "different-parent"
    elif mutation == "dataset":
        run["inputs"][0]["id"] = "unrelated-dataset"
    elif mutation == "protocol":
        evaluation["protocol_version"] = "different"
    elif mutation == "denominator":
        evaluation["metrics"]["internal_math.accuracy"]["denominator"] = 1
    elif mutation == "status":
        run["status"] = "oom"
    elif mutation == "model":
        evaluation["model_id"] = "unrelated-model"
    elif mutation == "recipe":
        run["recipe_path"] = "missing.yaml"
    elif mutation == "budget":
        plan.a.budget.value = plan.b.budget.value = 999
    elif mutation == "seed":
        resolved = {"training": {**plan.controls.training, "seed": 43}}
        write_report(tmp_path / "b/run/resolved_config.json", resolved)
        run["resolved_config_sha256"] = digest_json(resolved)
    write_report(run_path, run)
    write_report(eval_path, evaluation)
    report = compare_experiment(plan, root=tmp_path, selections=selections, registration=registration)
    assert report["comparison_status"] == "NOT_COMPARABLE"
    if mutation == "seed":
        # Preserve the observed value on its raw run, but never relabel seed 43
        # as the planned seed 42 in the comparable per-seed summary.
        assert report["runs"]["b"][0]["metrics"]["internal_math.accuracy"]["value"] == pytest.approx(0.6)
        assert report["model_metrics"][0]["b"] is None
    else:
        assert report["model_metrics"][0]["b"] == pytest.approx(0.6)
    assert report["model_metrics"][0]["delta"] is None
    assert report["compatibility_issues"]


def test_definition_rejects_multi_variable_and_budget_units(tmp_path):
    plan, _, selections = experiment_fixture(tmp_path)
    plan.b.policy.resources.max_length = 128
    plan.b.budget.unit = "optimizer_steps"
    issues = definition_issues(plan)
    assert any("一个" in error for error in issues)
    assert any("预算" in error for error in issues)
    with pytest.raises(ValueError, match="登记"):
        register_experiment(plan, root=tmp_path, output=tmp_path / "other.json")
    report = compare_experiment(plan, root=tmp_path, selections=selections)
    assert any("训练前实验登记" in error for error in report["compatibility_issues"])
