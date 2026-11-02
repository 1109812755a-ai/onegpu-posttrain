"""Plan-bound dry-run and explicit targeted generation entry points."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import yaml
from pydantic import ValidationError

from ..data.benchmarks import benchmark_index_from_jsonl
from ..data.capabilities import check_source_stage
from ..data.roles import SeedSelection, load_role_registry, select_training_seeds
from ..data.targeted import (
    DataRoleRegistry,
    ScenarioDefinition,
    ScenarioReport,
    TargetedGenerationPlan,
)
from ..data.targeted_plan import load_targeted_plan
from ..evaluation.scenarios import load_scenario_definition
from ..lineage import sha256_file, sha256_json
from .cache import load_response_cache, load_response_cache_by_key, write_response_cache
from .ledger import RequestLedger
from .locking import PlanExecutionLock
from .providers import (
    FakeProvider,
    GenerationRequest,
    GenerationResponse,
    OpenAICompatibleProvider,
    OutputProtocol,
    PromptTemplate,
    ProviderError,
    ProviderProfile,
    require_provider_credentials,
    validate_request_support,
)


@dataclass(frozen=True)
class GenerationSetup:
    plan: TargetedGenerationPlan
    profile: ProviderProfile
    template: PromptTemplate
    protocol: OutputProtocol
    definition: ScenarioDefinition
    report: ScenarioReport
    registry: DataRoleRegistry
    selection: SeedSelection
    protected_index_id: str
    protected_index_sha256: str

    def dry_run_summary(self) -> dict[str, Any]:
        return {
            "status": "dry_run_ready",
            "plan_id": self.plan.plan_id,
            "provider_profile_id": self.profile.profile_id,
            "provider_adapter": self.profile.adapter,
            "model": self.profile.model,
            "model_revision": self.profile.revision or "unavailable",
            "scenario_id": self.plan.target_scenario_id,
            "selected_training_seed_count": len(self.selection.records),
            "training_seed_shortfall": self.selection.report["shortfall"],
            "planned_request_count": min(
                len(self.selection.records), self.plan.budget.max_requests,
            ),
            "max_requests": self.plan.budget.max_requests,
            "max_output_tokens_per_request": self.plan.budget.max_output_tokens,
            "temperature": self.plan.generation_parameters.temperature,
            "provider_seed": self.plan.generation_parameters.provider_seed,
            "contamination_index_id": self.protected_index_id,
            "contamination_index_sha256": self.protected_index_sha256,
            "provider_calls": 0,
            "network_used": False,
            "execution_status": "not_generated_not_trained_not_reevaluated",
        }


def _load_yaml_model(path: str | Path, model_type: type):
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return model_type.model_validate(payload)
    except (OSError, yaml.YAMLError, ValidationError, TypeError, ValueError):
        # Validation errors can include the original input value (including a
        # forbidden secret URL); never forward their details to CLI/UI output.
        raise ValueError(f"Invalid {model_type.__name__} configuration; sensitive input redacted") from None


def load_provider_profile(path: str | Path) -> ProviderProfile:
    return _load_yaml_model(path, ProviderProfile)


def load_prompt_template(path: str | Path) -> PromptTemplate:
    return _load_yaml_model(path, PromptTemplate)


def load_output_protocol(path: str | Path) -> OutputProtocol:
    return _load_yaml_model(path, OutputProtocol)


def load_scenario_report(path: str | Path) -> ScenarioReport:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        report_payload = {key: value for key, value in payload.items() if key not in {"report_id", "report_sha256"}}
        report = ScenarioReport.model_validate(report_payload)
        if payload.get("report_id") != report.report_id or payload.get("report_sha256") != report.report_sha256:
            raise ValueError
        return report
    except (OSError, json.JSONDecodeError, ValidationError, TypeError, ValueError):
        raise ValueError("Invalid scenario report or mismatched report identity") from None


def prepare_generation(
    *,
    plan_path: str | Path,
    profile_path: str | Path,
    template_path: str | Path,
    protocol_path: str | Path,
    definition_path: str | Path,
    report_path: str | Path,
    registry_path: str | Path,
    train_seed_path: str | Path,
    protected_dataset_paths: list[str | Path],
) -> GenerationSetup:
    """Validate all identities and select only isolated train_seed rows; no I/O to providers."""
    plan = load_targeted_plan(plan_path)
    supported, reason = check_source_stage("llm-generation", plan.stage)
    if not supported:
        raise ValueError(reason or "模型生成来源不支持所选训练阶段")
    profile = load_provider_profile(profile_path)
    template = load_prompt_template(template_path)
    protocol = load_output_protocol(protocol_path)
    definition = load_scenario_definition(definition_path)
    report = load_scenario_report(report_path)
    registry = load_role_registry(registry_path)

    if profile.profile_id != plan.provider_profile_id:
        raise ValueError("provider profile ID does not match preregistered plan")
    if hashlib.sha256(template.template.encode("utf-8")).hexdigest() != plan.prompt_template_sha256:
        raise ValueError("prompt template hash does not match preregistered plan")
    if definition.definition_id != plan.scenario_definition_id or definition.config_sha256 != plan.scenario_definition_sha256:
        raise ValueError("scenario definition does not match preregistered plan")
    if report.report_id != plan.scenario_report_id or report.report_sha256 != plan.scenario_report_sha256:
        raise ValueError("scenario report does not match preregistered plan")
    if report.scenario_definition_id != definition.definition_id:
        raise ValueError("scenario report and definition do not match")
    if registry.registry_sha256 != plan.role_registry_sha256:
        raise ValueError("role registry does not match preregistered plan")
    validate_request_support(
        profile,
        GenerationRequest(
            prompt="validation-only",
            max_output_tokens=plan.budget.max_output_tokens,
            temperature=plan.generation_parameters.temperature,
            seed=plan.generation_parameters.provider_seed,
            output_protocol=protocol,
        ),
    )
    if sha256_file(train_seed_path) != plan.training_seed.content_sha256:
        raise ValueError("training seed dataset hash does not match preregistered plan")
    if not protected_dataset_paths:
        raise ValueError("at least one diagnostic/final protected dataset is required")

    protected_hashes = [sha256_file(path) for path in protected_dataset_paths]
    protected_hash = sha256_json(protected_hashes)
    protected_index_id = f"protected-index-{protected_hash[:16]}"
    protected_index = benchmark_index_from_jsonl(protected_dataset_paths)
    selection = select_training_seeds(
        train_seed_path,
        dataset_id=plan.training_seed.dataset_id,
        registry=registry,
        definition=definition,
        scenario_id=plan.target_scenario_id,
        target_count=plan.budget.target_candidates,
        seed=plan.seed,
        protected_index=protected_index,
        protected_index_id=protected_index_id,
        protected_index_sha256=protected_hash,
    )
    return GenerationSetup(
        plan=plan, profile=profile, template=template, protocol=protocol,
        definition=definition, report=report, registry=registry,
        selection=selection, protected_index_id=protected_index_id,
        protected_index_sha256=protected_hash,
    )


def _render_seed_prompt(setup: GenerationSetup, row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    question = row.get("raw_problem") or row.get("question") or row.get("problem")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("selected train_seed is missing problem text")
    topic = metadata.get(setup.definition.label_field)
    if isinstance(topic, list):
        topic = ", ".join(item for item in topic if isinstance(item, str))
    if not isinstance(topic, str):
        topic = setup.plan.target_scenario_id
    prompt = setup.template.render({"question": question, "topic": topic})
    required = ", ".join(setup.protocol.required_fields)
    return f"{prompt}\n\nReturn one JSON object with these required fields: {required}."


def _execute_targeted_generation_locked(
    setup: GenerationSetup,
    *,
    output_path: str | Path,
    ledger_path: str | Path | None = None,
    provider: Any | None = None,
    retry_uncertain: bool = False,
    run_id: str | None = None,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Explicitly execute selected requests and store raw responses in a private path."""
    output = Path(output_path)
    if len(output.parents) < 2 or output.parent.name.lower() != "private" or output.parent.parent.name != setup.plan.plan_id:
        raise ValueError("raw generation responses must be written to artifacts/synthetic/<plan_id>/private/")
    if output.exists():
        raise FileExistsError("refusing to overwrite an existing private response artifact")
    plan_dir = output.parent.parent
    expected_ledger = plan_dir / "requests.jsonl"
    if ledger_path is not None and Path(ledger_path).resolve() != expected_ledger.resolve():
        raise ValueError("request ledger must stay beside the registered plan to prevent resetting its budget")
    ledger_path = expected_ledger
    ledger = RequestLedger(ledger_path)
    for row in ledger._latest_by_attempt.values():
        if row.get("plan_id") != setup.plan.plan_id:
            raise ValueError("request ledger contains attempts from a different plan")
    if provider is None:
        provider = FakeProvider() if setup.profile.adapter == "fake" else OpenAICompatibleProvider()
    max_requests = setup.plan.budget.max_requests
    available = ledger.remaining_requests(max_requests)
    selected = setup.selection.records[: setup.plan.budget.target_candidates]
    if not selected:
        summary = ledger.summary()
        return {
            "status": "budget_exhausted" if available == 0 else "no_eligible_training_seeds",
            "plan_id": setup.plan.plan_id,
            "request_count": 0,
            "target_candidate_count": setup.plan.budget.target_candidates,
            "shortfall": setup.plan.budget.target_candidates,
            "provider_calls": 0,
            "network_used": False,
            "ledger": summary,
            "execution_status": "not_generated_not_trained_not_reevaluated",
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    run_id = run_id or f"synthetic-run-{uuid4().hex}"
    written = 0
    cache_reused = 0
    cache_recovered = 0
    provider_attempts_this_run = 0
    stop_reason: str | None = None
    profile_sha256 = sha256_json(setup.profile.model_dump(mode="json"))
    try:
        with output.open("x", encoding="utf-8") as stream:
            for slot, row in enumerate(selected):
                if stop_requested and stop_requested():
                    stop_reason = "cancelled_before_next_request"
                    break
                request = GenerationRequest(
                    prompt=_render_seed_prompt(setup, row),
                    max_output_tokens=setup.plan.budget.max_output_tokens,
                    temperature=setup.plan.generation_parameters.temperature,
                    seed=setup.plan.generation_parameters.provider_seed,
                    timeout_seconds=setup.plan.budget.timeout_seconds,
                    output_protocol=setup.protocol,
                )
                metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                problem_id = metadata.get("problem_id")
                request_sha256 = sha256_json({
                    "plan_id": setup.plan.plan_id,
                    "profile_sha256": profile_sha256,
                    "prompt_sha256": hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
                    "template_sha256": setup.plan.prompt_template_sha256,
                    "model": setup.profile.model,
                    "endpoint": setup.profile.endpoint,
                    "revision": setup.profile.revision,
                    "record_id": row["id"],
                    "problem_id": problem_id,
                    "dataset_id": setup.plan.training_seed.dataset_id,
                    "dataset_sha256": setup.plan.training_seed.content_sha256,
                    "role_registry_sha256": setup.registry.registry_sha256,
                    "max_output_tokens": request.max_output_tokens,
                    "temperature": request.temperature,
                    "seed": request.seed,
                    "timeout_seconds": request.timeout_seconds,
                    "output_protocol": setup.protocol.model_dump(mode="json"),
                    "candidate_slot": slot,
                })
                completed = ledger.completed_cache(request_sha256)
                cache_hit = completed is not None
                unresolved = ledger.unresolved_attempt(request_sha256)
                candidate_response: GenerationResponse | None = None
                if completed is not None:
                    cache_path = completed.get("response_cache_path")
                    response_digest = completed.get("response_sha256")
                    if not isinstance(cache_path, str) or not isinstance(response_digest, str):
                        raise ValueError("completed request ledger entry has no response cache identity")
                    candidate_response = load_response_cache(
                        plan_dir, cache_path, request_sha256, response_digest,
                    )
                    cache_reused += 1
                else:
                    orphan = load_response_cache_by_key(plan_dir, request_sha256)
                    if orphan is not None:
                        if unresolved is None:
                            raise ValueError("response cache has no matching pending ledger attempt")
                        candidate_response, cache_path, response_digest = orphan
                        estimated_cost = None
                        if (
                            candidate_response.prompt_tokens is not None
                            and candidate_response.completion_tokens is not None
                            and setup.profile.input_cost_per_million_tokens is not None
                            and setup.profile.output_cost_per_million_tokens is not None
                        ):
                            estimated_cost = (
                                candidate_response.prompt_tokens * setup.profile.input_cost_per_million_tokens
                                + candidate_response.completion_tokens * setup.profile.output_cost_per_million_tokens
                            ) / 1_000_000
                        ledger.append({
                            "event": "request_attempt_result",
                            "attempt_id": unresolved["attempt_id"],
                            "plan_id": setup.plan.plan_id,
                            "run_id": run_id,
                            "record_id": row["id"],
                            "candidate_slot": slot,
                            "attempt_number": unresolved.get("attempt_number", 1),
                            "request_sha256": request_sha256,
                            "state": "completed",
                            "prompt_tokens": candidate_response.prompt_tokens,
                            "completion_tokens": candidate_response.completion_tokens,
                            "estimated_cost_usd": estimated_cost,
                            "cost_status": "estimate" if estimated_cost is not None else "unknown",
                            "response_cache_path": cache_path,
                            "response_sha256": response_digest,
                            "recovered_from_atomic_cache": True,
                        })
                        cache_hit = True
                        cache_recovered += 1
                        cache_reused += 1
                unresolved = None if cache_hit else unresolved
                if unresolved is not None and not retry_uncertain:
                    stop_reason = "uncertain_request_requires_explicit_retry"
                    break
                if unresolved is not None and retry_uncertain:
                    stop_reason = "retry_uncertain_may_duplicate_billing"
                if not cache_hit and ledger.remaining_requests(max_requests) == 0:
                    stop_reason = "max_requests_exhausted"
                    break
                candidate_attempts = 0
                while not cache_hit and ledger.remaining_requests(max_requests) > 0:
                    if stop_requested and stop_requested():
                        stop_reason = "cancelled_before_next_request"
                        break
                    candidate_attempts += 1
                    attempt_id = uuid4().hex
                    require_provider_credentials(setup.profile)
                    ledger.append({
                        "event": "request_attempt",
                        "attempt_id": attempt_id,
                        "plan_id": setup.plan.plan_id,
                        "run_id": run_id,
                        "record_id": row["id"],
                        "candidate_slot": slot,
                        "attempt_number": candidate_attempts,
                        "request_sha256": request_sha256,
                        "profile_sha256": profile_sha256,
                        "uncertain_retry_authorized": bool(unresolved is not None and retry_uncertain),
                        "state": "pending",
                    })
                    provider_attempts_this_run += 1
                    try:
                        candidate_response = provider.generate(setup.profile, request)
                    except ProviderError as exc:
                        ledger.append({
                            "event": "request_attempt_result",
                            "attempt_id": attempt_id,
                            "plan_id": setup.plan.plan_id,
                            "run_id": run_id,
                            "record_id": row["id"],
                            "candidate_slot": slot,
                            "attempt_number": candidate_attempts,
                            "request_sha256": request_sha256,
                            "state": "uncertain" if exc.uncertain else "failed",
                            "error_category": exc.category,
                            "status_code": exc.status_code,
                            "retryable": exc.retryable,
                        })
                        if (
                            exc.retryable
                            and not exc.uncertain
                            and candidate_attempts <= setup.plan.budget.max_retries
                            and ledger.remaining_requests(max_requests) > 0
                        ):
                            continue
                        stop_reason = (
                            "uncertain_request_requires_explicit_retry" if exc.uncertain else "provider_error"
                        )
                        break
                    estimated_cost = None
                    if (
                        candidate_response.prompt_tokens is not None
                        and candidate_response.completion_tokens is not None
                        and setup.profile.input_cost_per_million_tokens is not None
                        and setup.profile.output_cost_per_million_tokens is not None
                    ):
                        estimated_cost = (
                            candidate_response.prompt_tokens * setup.profile.input_cost_per_million_tokens
                            + candidate_response.completion_tokens * setup.profile.output_cost_per_million_tokens
                        ) / 1_000_000
                    cache_path, response_digest = write_response_cache(plan_dir, request_sha256, candidate_response)
                    ledger.append({
                        "event": "request_attempt_result",
                        "attempt_id": attempt_id,
                        "plan_id": setup.plan.plan_id,
                        "run_id": run_id,
                        "record_id": row["id"],
                        "candidate_slot": slot,
                        "attempt_number": candidate_attempts,
                        "request_sha256": request_sha256,
                        "state": "completed",
                        "prompt_tokens": candidate_response.prompt_tokens,
                        "completion_tokens": candidate_response.completion_tokens,
                        "estimated_cost_usd": estimated_cost,
                        "cost_status": "estimate" if estimated_cost is not None else "unknown",
                        "response_cache_path": cache_path,
                        "response_sha256": response_digest,
                        "model_revision": setup.profile.revision,
                        "model_revision_status": "available" if setup.profile.revision else "unavailable",
                        "provider_request_id": candidate_response.provider_request_id,
                    })
                    break

                if candidate_response is None:
                    if stop_reason is None:
                        stop_reason = "max_requests_exhausted"
                    break
                record_id = row["id"]
                response_sha256 = sha256_json(candidate_response.model_dump(mode="json"))
                result = {
                    "record_id": record_id,
                    "problem_id": problem_id,
                    "candidate_slot": slot,
                    "request_sha256": request_sha256,
                    "response_source": "cache" if cache_hit else "provider",
                    "response": candidate_response.text,
                    "response_sha256": response_sha256,
                    "finish_reason": candidate_response.finish_reason,
                    "prompt_tokens": candidate_response.prompt_tokens,
                    "completion_tokens": candidate_response.completion_tokens,
                    "provider_request_id": candidate_response.provider_request_id,
                }
                stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                written += 1
                if progress_callback:
                    progress_callback({"completed_candidates": written, "attempted_requests": provider_attempts_this_run,
                                       "candidate_target": setup.plan.budget.target_candidates,
                                       "request_budget": max_requests})
    except Exception:
        if written == 0:
            output.unlink(missing_ok=True)
        raise
    if written == 0:
        output.unlink(missing_ok=True)
    summary = ledger.summary()
    return {
        "status": "partial" if written < setup.plan.budget.target_candidates else "generated",
        "plan_id": setup.plan.plan_id,
        "request_count": provider_attempts_this_run,
        "successful_candidates": written,
        "target_candidate_count": setup.plan.budget.target_candidates,
        "shortfall": max(0, setup.plan.budget.target_candidates - written),
        "stop_reason": stop_reason or ("target_met" if written >= setup.plan.budget.target_candidates else "source_shortfall"),
        "output_path": str(output),
        "provider_calls": provider_attempts_this_run,
        "cache_reused_count": cache_reused,
        "cache_recovered_count": cache_recovered,
        "ledger": summary,
        "network_used": provider_attempts_this_run > 0 and setup.profile.adapter != "fake",
        "execution_status": "generated_not_trained_not_reevaluated" if written else "partial_not_trained_not_reevaluated",
    }


def execute_targeted_generation(
    setup: GenerationSetup,
    *,
    output_path: str | Path,
    ledger_path: str | Path | None = None,
    provider: Any | None = None,
    retry_uncertain: bool = False,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    output = Path(output_path)
    if len(output.parents) < 2 or output.parent.name.lower() != "private" or output.parent.parent.name != setup.plan.plan_id:
        raise ValueError("raw generation responses must be written to artifacts/synthetic/<plan_id>/private/")
    plan_dir = output.parent.parent
    plan_dir.mkdir(parents=True, exist_ok=True)
    with PlanExecutionLock(plan_dir) as lock:
        return _execute_targeted_generation_locked(
            setup,
            output_path=output,
            ledger_path=ledger_path,
            provider=provider,
            retry_uncertain=retry_uncertain,
            run_id=lock.run_id,
            progress_callback=progress_callback,
            stop_requested=stop_requested,
        )
