"""Build and register an explicitly labeled, CPU-only policy experiment fixture."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from onegpu_posttrain.data.policy import digest_json, load_policy
from onegpu_posttrain.data.policy_build import build_policy_artifacts, load_source
from onegpu_posttrain.data.policy_experiment import (
    Budget,
    DataEffectExperiment,
    EvaluationContract,
    ExperimentArm,
    ExperimentControls,
    compare_experiment,
    register_experiment,
)
from onegpu_posttrain.lineage import sha256_file


def main() -> int:
    root = Path.cwd().resolve()
    if not (root / "pyproject.toml").is_file():
        raise SystemExit("请从项目根目录运行此示例")
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / "artifacts" / "data-experiments" / f"offline-example-{suffix}"
    source = load_source(root / "configs/data/policy-sources/example.yaml")
    baseline = load_policy(root / "configs/data/policies/sft-example.yaml")
    relaxed = baseline.model_copy(deep=True, update={
        "title": "B · 保留不完整推理并标记",
        "rules": baseline.rules.model_copy(update={"completeness": "report_only"}),
    })

    built = {}
    for label, policy in (("a", baseline), ("b", relaxed)):
        built[label] = build_policy_artifacts(policy, source, output_dir=output / f"data-{label}")
    a_report, b_report = built["a"]["report"], built["b"]["report"]
    if a_report["source_pool"]["pool_id"] != b_report["source_pool"]["pool_id"]:
        raise SystemExit("A/B 来源池不同，不能作为对照")
    recipe = root / "recipes/smoke/20-sft.yaml"
    placeholder_hash = "0" * 64
    evaluation_hash = digest_json({"example": "no evaluation has been run"})
    plan = DataEffectExperiment(
        question="项目自有样例中，报告完整性规则会如何改变可用数据？",
        evidence_kind="fixture",
        experiment_type="controlled_quality_ablation",
        independent_variable="rules.completeness",
        source_pool_id=a_report["source_pool"]["pool_id"],
        split_snapshot_id=a_report["split_snapshot"]["snapshot_id"],
        a=ExperimentArm(policy=baseline, manifest=(Path(f"artifacts/data-experiments/{output.name}/data-a/manifests/sft-policy-v1.yaml").as_posix()),
                        budget=Budget(unit="examples", value=2)),
        b=ExperimentArm(policy=relaxed, manifest=(Path(f"artifacts/data-experiments/{output.name}/data-b/manifests/sft-policy-v1.yaml").as_posix()),
                        budget=Budget(unit="examples", value=2)),
        controls=ExperimentControls(parent_model_id="fixture-parent-model", parent_model_sha256=placeholder_hash,
                                    recipe_sha256=sha256_file(recipe), training={"seed": 42, "max_length": 128}, seeds=[42]),
        evaluation=EvaluationContract(dataset_ids=["fixture-eval-not-run"],
                                      dataset_sha256={"fixture-eval-not-run": placeholder_hash},
                                      roles={"fixture-eval-not-run": "diagnostic_dev"},
                                      protocol_version="fixture-not-run-v1", config_sha256=evaluation_hash),
    )
    registration = register_experiment(plan, root=root, output=output / "registration.json")
    (output / "experiment.yaml").write_text(yaml.safe_dump(plan.model_dump(mode="json"), allow_unicode=True, sort_keys=False), encoding="utf-8")
    report = compare_experiment(plan, root=root, registration=registration, output=output / "data_effect_report.json")
    print(json.dumps({"status": report["comparison_status"], "evidence_kind": report["evidence_kind"],
                      "experiment_id": report["experiment_id"], "registration": str(output / "registration.json"),
                      "report": str(output / "data_effect_report.json"),
                      "data_records": {key: value.get("records") for key, value in report["data"].items()},
                      "model_effect": "NOT_EVALUATED"}, ensure_ascii=False, indent=2))
    return 0 if report["comparison_status"] == "NOT_EVALUATED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
