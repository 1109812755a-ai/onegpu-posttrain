"""CPU-only project-owned fixture walkthrough for targeted-data planning."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import yaml

from onegpu_posttrain.cli import build_parser
from onegpu_posttrain.data.benchmarks import benchmark_index_from_jsonl
from onegpu_posttrain.data.roles import (
    register_dataset,
    save_role_registry,
    select_training_seeds,
)
from onegpu_posttrain.data.targeted import (
    DataIdentity,
    EvaluationContract,
    EvaluationIdentity,
    GenerationBudget,
    MetricTarget,
    ModelIdentity,
    ScenarioDefinition,
    ScenarioLabel,
    TargetedGenerationPlan,
)
from onegpu_posttrain.data.targeted_plan import load_targeted_plan
from onegpu_posttrain.evaluation.scenarios import (
    build_scenario_report,
    write_scenario_report,
)
from onegpu_posttrain.lineage import sha256_file, sha256_json

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "targeted_synthetic"
SHA = "a" * 64


def main() -> int:
    train = FIXTURES / "train_seed.jsonl"
    diagnostic = FIXTURES / "diagnostic_dev.jsonl"
    holdout = FIXTURES / "final_holdout.jsonl"
    predictions = FIXTURES / "diagnostic_predictions.jsonl"
    definition = ScenarioDefinition(
        taxonomy_id="fixture-math-scenarios",
        taxonomy_version="v1",
        dataset_id="fixture-diagnostic-dev-v1",
        dataset_sha256=sha256_file(diagnostic),
        label_source="metadata",
        label_field="topic",
        scenarios=[
            ScenarioLabel(scenario_id="algebra", title="代数（示例标签）", values=["algebra"]),
            ScenarioLabel(scenario_id="arithmetic", title="算术（示例标签）", values=["arithmetic"]),
        ],
    )
    report = build_scenario_report(
        definition=definition,
        dataset_path=diagnostic,
        predictions_path=predictions,
        eval_id="eval-fixture-diagnostic-v1",
        model_id="model-fixture-sft-v1",
        protocol_version="internal-math-fixture-v1",
        threshold_source="本示例登记的演示阈值；不是推荐门槛",
        minimum_samples_required=2,
    )

    roles = register_dataset(None, train, dataset_id="fixture-train-seeds-v1", role="train_seed")
    roles = register_dataset(roles, diagnostic, dataset_id="fixture-diagnostic-dev-v1", role="diagnostic_dev")
    roles = register_dataset(roles, holdout, dataset_id="fixture-final-holdout-v1", role="final_holdout")
    index = benchmark_index_from_jsonl([diagnostic, holdout])
    protected_hash = sha256_json([sha256_file(diagnostic), sha256_file(holdout)])
    selection = select_training_seeds(
        train,
        dataset_id="fixture-train-seeds-v1",
        registry=roles,
        definition=definition,
        scenario_id="algebra",
        target_count=3,
        seed=42,
        protected_index=index,
        protected_index_id="fixture-diagnostic-and-holdout-v1",
        protected_index_sha256=protected_hash,
    )

    plan = TargetedGenerationPlan(
        scenario_report_id=report.report_id,
        scenario_report_sha256=report.report_sha256,
        scenario_definition_id=definition.definition_id,
        scenario_definition_sha256=definition.config_sha256,
        role_registry_sha256=roles.registry_sha256,
        target_scenario_id="algebra",
        target=MetricTarget(
            name="accuracy",
            direction="higher",
            threshold=0.66,
            threshold_source="本示例登记的演示阈值；不是推荐门槛",
            minimum_samples=2,
        ),
        diagnostic=EvaluationIdentity(
            eval_id="eval-fixture-diagnostic-v1",
            model_id="model-fixture-sft-v1",
            dataset_id="fixture-diagnostic-dev-v1",
            dataset_sha256=sha256_file(diagnostic),
            protocol_version="internal-math-fixture-v1",
            config_sha256=SHA,
            role="diagnostic_dev",
        ),
        training_seed=DataIdentity(
            dataset_id="fixture-train-seeds-v1",
            content_sha256=sha256_file(train),
            role="train_seed",
        ),
        starting_model=ModelIdentity(model_id="model-fixture-sft-v1", content_sha256=SHA),
        final_evaluation=EvaluationContract(
            dataset_id="fixture-final-holdout-v1",
            dataset_sha256=sha256_file(holdout),
            protocol_version="internal-math-fixture-v1",
            config_sha256=SHA,
            role="final_holdout",
        ),
        stage="sft",
        comparison_design="before_after_pipeline",
        provider_profile_id="fake-local",
        prompt_template_sha256=hashlib.sha256(b"fixture prompt template v1").hexdigest(),
        budget=GenerationBudget(max_requests=5, max_output_tokens=256, target_candidates=3),
        training_budget=2,
        seed=42,
    )

    with tempfile.TemporaryDirectory(prefix="onegpu-targeted-") as temp:
        temp_root = Path(temp)
        report_path = temp_root / "scenario-report.json"
        roles_path = temp_root / "role-registry.json"
        plan_yaml = temp_root / "plan.yaml"
        registered_path = temp_root / "registered-plan.json"
        write_scenario_report(report, report_path)
        save_role_registry(roles, roles_path)
        plan_yaml.write_text(yaml.safe_dump(plan.model_dump(mode="json"), allow_unicode=True), encoding="utf-8")
        cli_args = build_parser().parse_args([
            "register-targeted-plan", "--plan", str(plan_yaml), "--output", str(registered_path),
        ])
        cli_args.func(cli_args)
        registered = load_targeted_plan(registered_path)
        result = {
            "evidence_kind": "fixture",
            "scenario_report_id": report.report_id,
            "scenario_status": {row.scenario_id: row.status for row in report.scenarios},
            "selected_training_seed_ids": [row["id"] for row in selection.records],
            "training_seed_shortfall": selection.report["shortfall"],
            "role_registry_id": roles.registry_id,
            "registered_plan_id": registered.plan_id,
            "execution_status": "not_generated_not_trained_not_reevaluated",
            "provider_calls": 0,
            "network_used": False,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
