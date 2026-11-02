"""Identity checks connecting the shared stage plan to the proven SFT workflow."""

from __future__ import annotations

from pathlib import Path

from .targeted import TargetedGenerationPlan
from .targeted_plan import load_targeted_plan
from .targeted_stage_plan import TargetedStagePlan
from ..synthetic.workflow_bundle import load_sft_workflow_bundle


def validate_targeted_sft_bridge(
    workspace: str | Path,
    *,
    stage_plan: TargetedStagePlan,
    bundle_id: str,
) -> dict[str, str | int]:
    """Require all legacy SFT execution inputs to exactly match the shared plan.

    Generation stays in the already-tested targeted SFT engine. This adapter
    prevents it from silently executing a different scenario/data/holdout plan.
    """
    if stage_plan.target.stage != "sft":
        raise ValueError("shared stage plan is not SFT")
    bundle, files = load_sft_workflow_bundle(workspace, bundle_id)
    legacy: TargetedGenerationPlan = load_targeted_plan(files["plan"])
    target = stage_plan.target
    evidence = target.evidence
    checks = {
        "scenario_id": target.scenario_id == legacy.target_scenario_id,
        "training_seed_id": stage_plan.training_seed.dataset_id == legacy.training_seed.dataset_id,
        "training_seed_sha256": stage_plan.training_seed.content_sha256 == legacy.training_seed.content_sha256,
        "starting_model_id": stage_plan.starting_model.model_id == legacy.starting_model.model_id,
        "starting_model_sha256": stage_plan.starting_model.content_sha256 == legacy.starting_model.content_sha256,
        "diagnostic_dataset_id": stage_plan.diagnostic_evaluation.dataset_id == legacy.diagnostic.dataset_id,
        "diagnostic_dataset_sha256": stage_plan.diagnostic_evaluation.dataset_sha256 == legacy.diagnostic.dataset_sha256,
        "diagnostic_protocol": stage_plan.diagnostic_evaluation.protocol_version == legacy.diagnostic.protocol_version,
        "final_holdout_id": stage_plan.final_evaluation.dataset_id == legacy.final_evaluation.dataset_id,
        "final_holdout_sha256": stage_plan.final_evaluation.dataset_sha256 == legacy.final_evaluation.dataset_sha256,
        "provider_profile_id": stage_plan.provider_profile_id == legacy.provider_profile_id,
        "seed": stage_plan.seed == legacy.seed,
        "training_budget": stage_plan.training_budget == legacy.training_budget,
        "generation_candidate_budget": stage_plan.budget.max_candidates == legacy.budget.target_candidates,
        "generation_request_budget": stage_plan.budget.max_requests == legacy.budget.max_requests,
        "generation_output_token_budget": stage_plan.budget.max_output_tokens == legacy.budget.max_output_tokens,
        "scenario_report_evidence": bool(evidence and evidence.evidence_kind == "scenario_report"
                                           and evidence.evidence_id == legacy.scenario_report_id
                                           and evidence.content_sha256 == legacy.scenario_report_sha256),
    }
    failures = [key for key, passed in checks.items() if not passed]
    if failures:
        raise ValueError("shared targeted SFT plan does not match executable bundle: " + ", ".join(failures))
    return {
        "stage_plan_id": stage_plan.plan_id,
        "stage_plan_version": stage_plan.version,
        "bundle_id": bundle.bundle_id,
        "bundle_sha256": bundle.bundle_sha256,
        "legacy_plan_id": legacy.plan_id,
        "scenario_id": target.scenario_id,
        "checks_passed": len(checks),
    }
