from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import onegpu_posttrain.synthetic.generation as generation_module
import onegpu_posttrain.cli as cli_module
from onegpu_posttrain.cli import build_parser
from onegpu_posttrain.data.roles import register_dataset, save_role_registry
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
from onegpu_posttrain.data.targeted_plan import save_targeted_plan
from onegpu_posttrain.evaluation.scenarios import (
    build_scenario_report,
    write_scenario_report,
)
from onegpu_posttrain.lineage import (
    sha256_file,
)
from onegpu_posttrain.synthetic.generation import (
    execute_targeted_generation,
    load_provider_profile,
    prepare_generation,
)
from onegpu_posttrain.synthetic.ledger import RequestLedger
from onegpu_posttrain.synthetic.locking import PlanExecutionLock
from onegpu_posttrain.synthetic.providers import (
    FakeProvider,
    GenerationResponse,
    ProviderError,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "targeted_synthetic"
SHA = "a" * 64


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")


def _prepare_files(tmp_path: Path) -> dict[str, Path]:
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
            ScenarioLabel(scenario_id="algebra", title="代数", values=["algebra"]),
            ScenarioLabel(scenario_id="arithmetic", title="算术", values=["arithmetic"]),
        ],
    )
    report = build_scenario_report(
        definition=definition,
        dataset_path=diagnostic,
        predictions_path=predictions,
        eval_id="eval-fixture-diagnostic-v1",
        model_id="model-fixture-sft-v1",
        protocol_version="fixture-v1",
        threshold_source="test fixture only",
        minimum_samples_required=2,
    )
    registry = register_dataset(None, train, dataset_id="fixture-train-seeds-v1", role="train_seed")
    registry = register_dataset(registry, diagnostic, dataset_id="fixture-diagnostic-dev-v1", role="diagnostic_dev")
    registry = register_dataset(registry, holdout, dataset_id="fixture-final-holdout-v1", role="final_holdout")

    profile_path = ROOT / "configs" / "synthetic" / "examples" / "provider.fake.yaml"
    template_path = ROOT / "configs" / "synthetic" / "examples" / "prompt.sft-math.yaml"
    protocol_path = ROOT / "configs" / "synthetic" / "examples" / "output.sft-candidate.yaml"
    profile_payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    template_payload = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    template_text = template_payload["template"]
    plan = TargetedGenerationPlan(
        scenario_report_id=report.report_id,
        scenario_report_sha256=report.report_sha256,
        scenario_definition_id=definition.definition_id,
        scenario_definition_sha256=definition.config_sha256,
        role_registry_sha256=registry.registry_sha256,
        target_scenario_id="algebra",
        target=MetricTarget(
            name="accuracy", direction="higher", threshold=0.5,
            threshold_source="test fixture only", minimum_samples=2,
        ),
        diagnostic=EvaluationIdentity(
            eval_id=report.eval_id, model_id=report.model_id, dataset_id=report.dataset_id,
            dataset_sha256=report.dataset_sha256, protocol_version=report.protocol_version,
            config_sha256=SHA, role="diagnostic_dev",
        ),
        training_seed=DataIdentity(
            dataset_id="fixture-train-seeds-v1", content_sha256=sha256_file(train), role="train_seed",
        ),
        starting_model=ModelIdentity(model_id="model-fixture-sft-v1", content_sha256=SHA),
        final_evaluation=EvaluationContract(
            dataset_id="fixture-final-holdout-v1", dataset_sha256=sha256_file(holdout),
            protocol_version="fixture-v1", config_sha256=SHA, role="final_holdout",
        ),
        stage="sft",
        comparison_design="before_after_pipeline",
        provider_profile_id=profile_payload["profile_id"],
        prompt_template_sha256=hashlib.sha256(template_text.encode()).hexdigest(),
        budget=GenerationBudget(max_requests=5, max_output_tokens=128, target_candidates=3),
        training_budget=10,
        seed=42,
    )

    paths = {
        "plan": tmp_path / "plan.json",
        "profile": profile_path,
        "template": template_path,
        "protocol": protocol_path,
        "definition": tmp_path / "definition.yaml",
        "report": tmp_path / "report.json",
        "registry": tmp_path / "registry.json",
        "train": train,
        "diagnostic": diagnostic,
        "holdout": holdout,
    }
    save_targeted_plan(plan, paths["plan"])
    _write_yaml(paths["definition"], definition.model_dump(mode="json"))
    write_scenario_report(report, paths["report"])
    save_role_registry(registry, paths["registry"])
    return paths


def _prepare(tmp_path: Path):
    paths = _prepare_files(tmp_path)
    setup = prepare_generation(
        plan_path=paths["plan"], profile_path=paths["profile"], template_path=paths["template"],
        protocol_path=paths["protocol"], definition_path=paths["definition"], report_path=paths["report"],
        registry_path=paths["registry"], train_seed_path=paths["train"],
        protected_dataset_paths=[paths["diagnostic"], paths["holdout"]],
    )
    return paths, setup


def _private_output(tmp_path: Path, setup, name: str) -> Path:
    return tmp_path / setup.plan.plan_id / "private" / name


def test_dry_run_validates_inputs_without_credentials_or_provider_calls(tmp_path, monkeypatch):
    paths, setup = _prepare(tmp_path)
    monkeypatch.delenv("MISSING_FIXTURE_KEY", raising=False)
    summary = setup.dry_run_summary()
    assert summary["selected_training_seed_count"] == 2
    assert summary["planned_request_count"] == 2
    assert summary["provider_calls"] == 0
    assert summary["network_used"] is False

    output = _private_output(tmp_path, setup, "responses.jsonl")
    provider = FakeProvider()
    no_execute_args = build_parser().parse_args([
        "generate-targeted-data", "--plan", str(paths["plan"]), "--profile", str(paths["profile"]),
        "--template", str(paths["template"]), "--output-protocol", str(paths["protocol"]),
        "--definition", str(paths["definition"]), "--report", str(paths["report"]),
        "--role-registry", str(paths["registry"]), "--train-seeds", str(paths["train"]),
        "--protected-jsonl", str(paths["diagnostic"]), "--protected-jsonl", str(paths["holdout"]),
        "--output", str(output),
    ])
    with pytest.raises(SystemExit, match="Pass --execute"):
        no_execute_args.func(no_execute_args)
    assert provider.calls == 0
    assert not output.exists()


def test_unsupported_generation_stage_is_rejected_before_provider_calls(tmp_path):
    paths = _prepare_files(tmp_path)
    payload = json.loads(paths["plan"].read_text(encoding="utf-8"))
    payload["plan"]["stage"] = "grpo"
    paths["plan"].write_text(json.dumps(payload), encoding="utf-8")
    provider = FakeProvider()
    with pytest.raises(ValidationError):
        prepare_generation(
            plan_path=paths["plan"], profile_path=paths["profile"], template_path=paths["template"],
            protocol_path=paths["protocol"], definition_path=paths["definition"], report_path=paths["report"],
            registry_path=paths["registry"], train_seed_path=paths["train"],
            protected_dataset_paths=[paths["diagnostic"], paths["holdout"]],
        )
    assert provider.calls == 0


def test_explicit_execution_uses_only_train_seed_and_writes_private_responses(tmp_path):
    _, setup = _prepare(tmp_path)
    output = _private_output(tmp_path, setup, "responses.jsonl")
    provider = FakeProvider(['{"problem_id":"seed-problem-001","solution":"4x3=12","final_answer":"12"}'])
    result = execute_targeted_generation(setup, output_path=output, provider=provider)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert result["request_count"] == 2
    assert provider.calls == 2
    assert result["network_used"] is False
    assert result["ledger"]["usage_unknown_attempts"] == 2
    assert result["ledger"]["cost_status"] == "unknown"
    assert all(row["record_id"].startswith("seed-") for row in rows)
    assert "reference" not in rows[0]
    assert "diagnostic" not in output.read_text(encoding="utf-8")
    ledger_text = (output.parent.parent / "requests.jsonl").read_text(encoding="utf-8")
    assert "Write a short" not in ledger_text
    assert '"response"' not in ledger_text


def test_dry_run_cli_emits_non_execution_status(tmp_path, capsys):
    paths, _ = _prepare(tmp_path)
    args = build_parser().parse_args([
        "dry-run-targeted-generation", "--plan", str(paths["plan"]), "--profile", str(paths["profile"]),
        "--template", str(paths["template"]), "--output-protocol", str(paths["protocol"]),
        "--definition", str(paths["definition"]), "--report", str(paths["report"]),
        "--role-registry", str(paths["registry"]), "--train-seeds", str(paths["train"]),
        "--protected-jsonl", str(paths["diagnostic"]), "--protected-jsonl", str(paths["holdout"]),
    ])
    assert args.func(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["provider_calls"] == 0
    assert result["execution_status"] == "not_generated_not_trained_not_reevaluated"


def test_dry_run_does_not_read_local_credential_file(tmp_path, monkeypatch, capsys):
    paths, _ = _prepare(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ONEGPU_SYNTHETIC_API_KEY", raising=False)
    (tmp_path / ".env.local").write_text("invalid local secret file", encoding="utf-8")
    args = build_parser().parse_args([
        "dry-run-targeted-generation", "--plan", str(paths["plan"]), "--profile", str(paths["profile"]),
        "--template", str(paths["template"]), "--output-protocol", str(paths["protocol"]),
        "--definition", str(paths["definition"]), "--report", str(paths["report"]),
        "--role-registry", str(paths["registry"]), "--train-seeds", str(paths["train"]),
        "--protected-jsonl", str(paths["diagnostic"]), "--protected-jsonl", str(paths["holdout"]),
    ])

    assert args.func(args) == 0
    assert "ONEGPU_SYNTHETIC_API_KEY" not in os.environ
    assert "invalid local secret file" not in capsys.readouterr().out


def test_explicit_openai_execution_loads_dotenv_secret_without_printing_it(tmp_path, monkeypatch, capsys):
    paths = _prepare_files(tmp_path)
    openai_profile_path = tmp_path / "profile-openai-compatible.yaml"
    _write_yaml(openai_profile_path, {
        "profile_id": "fake-local", "adapter": "openai_compatible",
        "endpoint": "https://example.invalid/v1/chat/completions", "model": "fixture-model",
        "api_key_env": "ONEGPU_SYNTHETIC_API_KEY",
    })
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ONEGPU_SYNTHETIC_API_KEY", raising=False)
    (tmp_path / ".env.local").write_text("ONEGPU_SYNTHETIC_API_KEY=do-not-print-this\n", encoding="utf-8")
    output = tmp_path / "plan" / "private" / "responses.jsonl"
    args = build_parser().parse_args([
        "generate-targeted-data", "--plan", str(paths["plan"]), "--profile", str(openai_profile_path),
        "--template", str(paths["template"]), "--output-protocol", str(paths["protocol"]),
        "--definition", str(paths["definition"]), "--report", str(paths["report"]),
        "--role-registry", str(paths["registry"]), "--train-seeds", str(paths["train"]),
        "--protected-jsonl", str(paths["diagnostic"]), "--protected-jsonl", str(paths["holdout"]),
        "--output", str(output), "--execute",
    ])
    observed = {}

    def fake_execute(setup, *, output_path, retry_uncertain):
        observed["key"] = os.environ["ONEGPU_SYNTHETIC_API_KEY"]
        return {"status": "fixture-complete"}

    monkeypatch.setattr(cli_module, "execute_targeted_generation", fake_execute)
    assert args.func(args) == 0
    assert observed["key"] == "do-not-print-this"
    assert "do-not-print-this" not in capsys.readouterr().out


def test_generation_cli_calls_fake_only_with_explicit_execute(tmp_path, capsys):
    paths, setup = _prepare(tmp_path)
    output = _private_output(tmp_path, setup, "cli-responses.jsonl")
    args = build_parser().parse_args([
        "generate-targeted-data", "--plan", str(paths["plan"]), "--profile", str(paths["profile"]),
        "--template", str(paths["template"]), "--output-protocol", str(paths["protocol"]),
        "--definition", str(paths["definition"]), "--report", str(paths["report"]),
        "--role-registry", str(paths["registry"]), "--train-seeds", str(paths["train"]),
        "--protected-jsonl", str(paths["diagnostic"]), "--protected-jsonl", str(paths["holdout"]),
        "--output", str(output), "--execute",
    ])
    assert args.func(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "partial"
    assert result["request_count"] == 2
    assert result["execution_status"] == "generated_not_trained_not_reevaluated"
    assert output.exists()


def test_invalid_secret_endpoint_config_is_redacted_by_loader(tmp_path):
    profile_path = tmp_path / "bad-profile.yaml"
    profile_path.write_text(
        "profile_id: bad\nadapter: openai_compatible\nmodel: m\n"
        "endpoint: https://user:super-secret@example.invalid/v1/chat/completions\napi_key_env: KEY\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc:
        load_provider_profile(profile_path)
    assert "super-secret" not in str(exc.value)


def test_missing_openai_credential_does_not_leave_empty_response_file(tmp_path, monkeypatch):
    paths = _prepare_files(tmp_path)
    openai_profile_path = tmp_path / "profile-openai-compatible.yaml"
    _write_yaml(openai_profile_path, {
        "profile_id": "fake-local", "adapter": "openai_compatible",
        "endpoint": "https://example.invalid/v1/chat/completions", "model": "fixture-model",
        "api_key_env": "MISSING_FIXTURE_KEY",
    })
    paths["profile"] = openai_profile_path
    # Keep the profile id bound to the existing plan while validating the setup.
    setup = prepare_generation(
        plan_path=paths["plan"], profile_path=paths["profile"], template_path=paths["template"],
        protocol_path=paths["protocol"], definition_path=paths["definition"], report_path=paths["report"],
        registry_path=paths["registry"], train_seed_path=paths["train"],
        protected_dataset_paths=[paths["diagnostic"], paths["holdout"]],
    )
    monkeypatch.delenv("MISSING_FIXTURE_KEY", raising=False)
    output = _private_output(tmp_path, setup, "responses.jsonl")
    with pytest.raises(RuntimeError, match="MISSING_FIXTURE_KEY"):
        execute_targeted_generation(setup, output_path=output)
    assert not output.exists()


def test_retry_attempts_consume_request_budget_and_cost_stays_unknown_without_rates(tmp_path):
    _, setup = _prepare(tmp_path)
    setup = replace(
        setup,
        plan=setup.plan.model_copy(update={
            "budget": setup.plan.budget.model_copy(update={"max_requests": 2, "max_retries": 4}),
        }),
    )

    class AlwaysRateLimited:
        calls = 0

        def generate(self, profile, request):
            self.calls += 1
            raise ProviderError("rate_limited", "provider returned HTTP 429", retryable=True, uncertain=False,
                                status_code=429)

    provider = AlwaysRateLimited()
    output = _private_output(tmp_path, setup, "rate-limited.jsonl")
    result = execute_targeted_generation(setup, output_path=output, provider=provider)
    assert provider.calls == 2
    assert result["request_count"] == 2
    assert result["successful_candidates"] == 0
    assert result["ledger"]["request_attempts"] == 2
    assert result["ledger"]["cost_status"] == "unknown"
    assert result["ledger"]["cost_unknown_attempts"] == 2


def test_usage_and_user_declared_rates_are_reported_as_estimates(tmp_path):
    _, setup = _prepare(tmp_path)
    setup = replace(
        setup,
        profile=setup.profile.model_copy(update={
            "input_cost_per_million_tokens": 1.0,
            "output_cost_per_million_tokens": 2.0,
        }),
    )

    class MeteredProvider:
        def generate(self, profile, request):
            return GenerationResponse(text="{}", prompt_tokens=10, completion_tokens=5)

    output = _private_output(tmp_path, setup, "metered.jsonl")
    result = execute_targeted_generation(setup, output_path=output, provider=MeteredProvider())
    assert result["ledger"]["reported_prompt_tokens"] == 20
    assert result["ledger"]["reported_completion_tokens"] == 10
    assert result["ledger"]["cost_status"] == "estimate"
    assert result["ledger"]["estimated_cost_usd"] == pytest.approx(0.00004)


def test_uncertain_timeout_is_not_retried_automatically(tmp_path):
    _, setup = _prepare(tmp_path)
    setup = replace(
        setup,
        plan=setup.plan.model_copy(update={
            "budget": setup.plan.budget.model_copy(update={"max_requests": 5, "max_retries": 4}),
        }),
    )

    class TimedOutProvider:
        calls = 0

        def generate(self, profile, request):
            self.calls += 1
            raise ProviderError("timeout", "provider request timed out", retryable=True, uncertain=True)

    provider = TimedOutProvider()
    result = execute_targeted_generation(
        setup, output_path=_private_output(tmp_path, setup, "uncertain.jsonl"), provider=provider,
    )
    assert provider.calls == 1
    assert result["stop_reason"] == "uncertain_request_requires_explicit_retry"
    assert result["ledger"]["request_states"] == {"uncertain": 1}


def test_request_budget_persists_across_runs_for_same_plan(tmp_path):
    _, setup = _prepare(tmp_path)
    setup = replace(
        setup,
        plan=setup.plan.model_copy(update={
            "budget": GenerationBudget(max_requests=2, max_output_tokens=128, target_candidates=3),
        }),
    )
    first = execute_targeted_generation(
        setup, output_path=_private_output(tmp_path, setup, "first.jsonl"), provider=FakeProvider(),
    )
    second = execute_targeted_generation(
        setup, output_path=_private_output(tmp_path, setup, "second.jsonl"), provider=FakeProvider(),
    )
    assert first["request_count"] == 2
    assert second["status"] == "partial"
    assert second["provider_calls"] == 0
    assert second["cache_reused_count"] == 2
    assert second["ledger"]["request_attempts"] == 2
    assert _private_output(tmp_path, setup, "second.jsonl").exists()


def test_malformed_request_ledger_cannot_reset_budget(tmp_path):
    ledger_path = tmp_path / "requests.jsonl"
    ledger_path.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to reset"):
        RequestLedger(ledger_path)


def test_completed_response_cache_is_reused_without_spending_requests(tmp_path):
    _, setup = _prepare(tmp_path)
    first_provider = FakeProvider()
    first = execute_targeted_generation(
        setup, output_path=_private_output(tmp_path, setup, "first-cache.jsonl"), provider=first_provider,
    )
    second_provider = FakeProvider()
    second = execute_targeted_generation(
        setup, output_path=_private_output(tmp_path, setup, "second-cache.jsonl"), provider=second_provider,
    )
    second_rows = [
        json.loads(line)
        for line in _private_output(tmp_path, setup, "second-cache.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert first["request_count"] == 2
    assert second["request_count"] == 0
    assert second["cache_reused_count"] == 2
    assert second_provider.calls == 0
    assert all(row["response_source"] == "cache" for row in second_rows)


def test_tampered_completed_cache_is_rejected_instead_of_reused(tmp_path):
    _, setup = _prepare(tmp_path)
    execute_targeted_generation(
        setup, output_path=_private_output(tmp_path, setup, "first-cache.jsonl"), provider=FakeProvider(),
    )
    ledger_rows = [
        json.loads(line)
        for line in (tmp_path / setup.plan.plan_id / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    first_completed = next(
        row for row in ledger_rows
        if row.get("event") == "request_attempt_result" and row.get("state") == "completed"
        and row.get("record_id") == "seed-algebra-001"
    )
    cache_file = tmp_path / setup.plan.plan_id / first_completed["response_cache_path"]
    cache_file.write_text('{"response":"tampered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash-mismatched"):
        execute_targeted_generation(
            setup, output_path=_private_output(tmp_path, setup, "tampered-cache.jsonl"), provider=FakeProvider(),
        )
    assert not _private_output(tmp_path, setup, "tampered-cache.jsonl").exists()


def test_atomic_cache_written_before_ledger_completion_is_recovered(tmp_path, monkeypatch):
    _, setup = _prepare(tmp_path)
    original_write = generation_module.write_response_cache
    provider = FakeProvider()

    def write_then_crash(plan_dir, request_sha256, response):
        original_write(plan_dir, request_sha256, response)
        raise RuntimeError("simulated interruption after atomic response publication")

    monkeypatch.setattr(generation_module, "write_response_cache", write_then_crash)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        execute_targeted_generation(
            setup,
            output_path=_private_output(tmp_path, setup, "interrupted.jsonl"),
            provider=provider,
        )
    assert provider.calls == 1

    monkeypatch.setattr(generation_module, "write_response_cache", original_write)
    resumed_provider = FakeProvider()
    resumed = execute_targeted_generation(
        setup,
        output_path=_private_output(tmp_path, setup, "resumed.jsonl"),
        provider=resumed_provider,
    )
    assert resumed["cache_recovered_count"] == 1
    assert resumed["request_count"] == 1
    assert resumed_provider.calls == 1


def test_profile_revision_change_causes_cache_miss(tmp_path):
    _, setup = _prepare(tmp_path)
    execute_targeted_generation(
        setup, output_path=_private_output(tmp_path, setup, "first-profile.jsonl"), provider=FakeProvider(),
    )
    changed_setup = replace(setup, profile=setup.profile.model_copy(update={"revision": "revision-2"}))
    provider = FakeProvider()
    result = execute_targeted_generation(
        changed_setup,
        output_path=_private_output(tmp_path, setup, "changed-profile.jsonl"),
        provider=provider,
    )
    assert result["cache_reused_count"] == 0
    assert result["request_count"] == 2
    assert provider.calls == 2


def test_uncertain_attempt_requires_explicit_retry_and_remains_budgeted(tmp_path):
    _, setup = _prepare(tmp_path)

    class TimedOutProvider:
        def generate(self, profile, request):
            raise ProviderError("timeout", "provider request timed out", retryable=True, uncertain=True)

    first = execute_targeted_generation(
        setup,
        output_path=_private_output(tmp_path, setup, "first-uncertain.jsonl"),
        provider=TimedOutProvider(),
    )
    assert first["ledger"]["request_states"] == {"uncertain": 1}
    blocked_provider = FakeProvider()
    blocked = execute_targeted_generation(
        setup,
        output_path=_private_output(tmp_path, setup, "blocked-uncertain.jsonl"),
        provider=blocked_provider,
    )
    assert blocked["stop_reason"] == "uncertain_request_requires_explicit_retry"
    assert blocked_provider.calls == 0
    retry_provider = FakeProvider()
    retried = execute_targeted_generation(
        setup,
        output_path=_private_output(tmp_path, setup, "explicit-retry.jsonl"),
        provider=retry_provider,
        retry_uncertain=True,
    )
    assert retry_provider.calls == 2
    assert retried["stop_reason"] == "retry_uncertain_may_duplicate_billing"
    assert retried["ledger"]["request_attempts"] == 3


def test_same_plan_execution_lock_rejects_concurrent_execution(tmp_path):
    _, setup = _prepare(tmp_path)
    plan_dir = tmp_path / setup.plan.plan_id
    with PlanExecutionLock(plan_dir), pytest.raises(RuntimeError, match="another execution owns"):
        execute_targeted_generation(
            setup,
            output_path=_private_output(tmp_path, setup, "concurrent.jsonl"),
            provider=FakeProvider(),
        )
