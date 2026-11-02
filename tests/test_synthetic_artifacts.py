from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from onegpu_posttrain.evaluation import verification

from onegpu_posttrain.data.decontaminate import BenchmarkIndex
from onegpu_posttrain.data.manifest import load_manifest, validate_manifest_evidence
from onegpu_posttrain.data.policy import DataPolicy, PolicyTokenizer
from onegpu_posttrain.data.targeted import (
    DataRoleRegistry,
    GenerationParameters,
    RoleRecord,
)
from onegpu_posttrain.data.tokenization import WhitespaceTokenizer
from onegpu_posttrain.lineage_validation import validate_lineage_root
from onegpu_posttrain.synthetic.artifacts import write_targeted_artifacts
from onegpu_posttrain.synthetic.candidates import validate_candidate_response
from onegpu_posttrain.synthetic.policy_export import (
    CandidatePolicyInput,
    export_candidates_with_policy,
)
from onegpu_posttrain.synthetic.providers import OutputProtocol, ProviderProfile
from onegpu_posttrain.data.policy import PolicyRules
from onegpu_posttrain.data.targeted import GenerationParameters
from onegpu_posttrain.synthetic.policy_export import CandidatePolicyInput, export_candidates_with_policy
from onegpu_posttrain.ui.catalog import ArtifactCatalog, ArtifactError

SHA = "a" * 64
PROTOCOL = OutputProtocol(
    protocol_id="candidate", version="v1", required_fields=["problem_id", "solution", "final_answer"],
)


def test_synthetic_artifacts_use_standard_lineage_and_hide_private_content_from_ui(tmp_path):
    question = "A unique private seed question for artifact lineage testing."
    response = json.dumps({
        "problem_id": "seed-problem", "solution": "2+2=4. Final answer: 4", "final_answer": "4",
    })
    candidate = validate_candidate_response(
        response, source_record_id="seed-record", expected_problem_id="seed-problem", candidate_slot=0,
        request_sha256=hashlib.sha256(b"request").hexdigest(), finish_reason="stop", question=question,
        protocol=PROTOCOL, tokenizer=WhitespaceTokenizer(), max_training_tokens=128,
    )
    registry = DataRoleRegistry(records=[RoleRecord(
        dataset_id="seed-dataset", dataset_sha256=SHA, record_id="seed-record",
        problem_id="seed-problem", role="train_seed",
    )])
    policy = DataPolicy(stage="sft", tokenizer=PolicyTokenizer(test_only=True))
    exported = export_candidates_with_policy(
        [CandidatePolicyInput(candidate=candidate, question=question, reference_answer="4")],
        policy=policy, tokenizer=WhitespaceTokenizer(), registry=registry,
        protected_index=BenchmarkIndex([("holdout", "unrelated held out benchmark phrase")]),
        protected_index_id="protected-v1", protected_index_sha256=SHA,
    )
    plan = SimpleNamespace(
        plan_id="plan-targeted-fixture",
        training_seed=SimpleNamespace(dataset_id="seed-dataset", content_sha256=SHA),
        prompt_template_sha256=SHA,
        generation_parameters=GenerationParameters(temperature=0.2),
        budget=SimpleNamespace(target_candidates=1),
        training_budget=1,
    )
    profile = ProviderProfile(profile_id="fake-local", adapter="fake", model="fixture-v1", revision=None)
    build_dir = tmp_path / "artifacts" / "synthetic" / "plan-targeted-fixture" / "build-001"
    result = write_targeted_artifacts(
        output_dir=build_dir,
        plan=plan,
        profile=profile,
        tokenizer=WhitespaceTokenizer(),
        candidates=[candidate],
        policy_export=exported,
        generation_summary={
            "request_count": 1, "successful_candidates": 1, "target_candidate_count": 1,
            "ledger": {"cost_status": "unknown", "request_attempts": 1},
            "output_path": "PRIVATE_SENTINEL_PATH",
        },
        validation_summary={"candidate_count": 1, "raw_response": "PRIVATE_SENTINEL_RESPONSE"},
    )
    assert result["public_response_bodies_included"] is False
    assert result["model_revision_status"] == "unavailable"
    candidate_manifest_path = Path(result["candidate_manifest"])
    training_manifest_path = Path(result["training_manifest"])
    candidate_manifest = load_manifest(candidate_manifest_path)
    training_manifest = load_manifest(training_manifest_path)
    assert candidate_manifest.stage == "synthetic_candidates"
    assert training_manifest.stage == "sft"
    assert not validate_manifest_evidence(candidate_manifest_path)
    assert not validate_manifest_evidence(training_manifest_path)
    lineage = validate_lineage_root(build_dir, require_complete=True)
    assert not lineage["errors"], lineage

    report_text = Path(result["generation_validation_cost_report"]).read_text(encoding="utf-8")
    assert "PRIVATE_SENTINEL" not in report_text
    private_text = (build_dir / "private" / "candidates.jsonl").read_text(encoding="utf-8")
    assert "2+2=4" in private_text
    catalog = ArtifactCatalog(tmp_path)
    with pytest.raises(ArtifactError, match="not exposed"):
        catalog._resolve(build_dir / "private" / "candidates.jsonl")
    index = catalog.scan()
    generated = [build for build in index["builds"] for dataset in build["datasets"]
                 if dataset["dataset_id"] == result["training_dataset_id"]]
    assert len(generated) == 1
    summary_dataset = next(dataset for build in index["builds"] for dataset in build["datasets"]
                           if dataset["dataset_id"] == result["training_dataset_id"])
    assert summary_dataset["source"]["kind"] == "model_generated"
    assert summary_dataset["quality_status"] == "NOT_EVALUATED"
    assert summary_dataset["content_sha256"]
    detail = catalog.detail(generated[0]["id"])
    registered = next(dataset for dataset in detail["datasets"]
                      if dataset["dataset_id"] == result["training_dataset_id"])
    assert registered["source"]["kind"] == "model_generated"
    assert registered["source"]["source_id"] == "llm-generation"
    assert "2+2=4" not in json.dumps(detail)
    processing_report = yaml.safe_load(
        (build_dir / "reports" / "processing_report.json").read_text(encoding="utf-8")
    )
    assert processing_report["source_acquisition"]["revision_status"] == "unavailable"


def test_preference_generation_artifact_preserves_both_candidate_ancestors(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    verifier = ModuleType("math_verify")
    verifier.parse = lambda value: value
    verifier.verify = lambda gold, pred: gold == pred
    monkeypatch.setitem(sys.modules, "math_verify", verifier)
    monkeypatch.setattr(verification, "_verification_identity", lambda: ("math-verify", "test"))
    monkeypatch.setattr(
        verification, "_run_bounded_batch",
        lambda pairs, _timeout: ["correct" if p.prediction == p.reference else "incorrect" for p in pairs],
    )
    question = "Private preference fixture: what is two plus two?"
    registry = DataRoleRegistry(records=[RoleRecord(
        dataset_id="seed-dataset", dataset_sha256=SHA, record_id="seed-record",
        problem_id="seed-problem", role="train_seed",
    )])
    candidates = [
        validate_candidate_response(
            json.dumps({"problem_id": "seed-problem", "solution": solution, "final_answer": answer}),
            source_record_id="seed-record", expected_problem_id="seed-problem", candidate_slot=slot,
            request_sha256=hashlib.sha256(f"request-{slot}".encode()).hexdigest(), finish_reason="stop",
            question=question, protocol=PROTOCOL, tokenizer=WhitespaceTokenizer(), max_training_tokens=128,
        )
        for slot, (solution, answer) in enumerate((("2+2=4. Final answer: 4", "4"), ("2+2=5. Final answer: 5", "5")))
    ]
    policy = DataPolicy(
        stage="preference", tokenizer=PolicyTokenizer(test_only=True),
        rules=PolicyRules(min_length_ratio=0.5, max_length_ratio=2.0),
    )
    exported = export_candidates_with_policy(
        [CandidatePolicyInput(candidate=item, question=question, reference_answer="4") for item in candidates],
        policy=policy, tokenizer=WhitespaceTokenizer(), registry=registry,
        protected_index=BenchmarkIndex([("holdout", "unrelated protected question")]),
        protected_index_id="index-v1", protected_index_sha256=SHA,
    )
    assert exported["selected_count"] == 1
    plan = SimpleNamespace(
        plan_id="plan-preference-fixture",
        training_seed=SimpleNamespace(dataset_id="seed-dataset", content_sha256=SHA),
        prompt_template_sha256=SHA,
        generation_parameters=GenerationParameters(temperature=0.2),
        budget=SimpleNamespace(target_candidates=2), training_budget=1,
    )
    result = write_targeted_artifacts(
        output_dir=tmp_path / "artifacts" / "synthetic" / "plan-preference-fixture" / "build-001",
        plan=plan, profile=ProviderProfile(profile_id="fake-pref", adapter="fake", model="fixture"),
        tokenizer=WhitespaceTokenizer(), candidates=candidates, policy_export=exported,
        generation_summary={"successful_candidates": 2, "provider_calls": 0},
    )
    manifest = load_manifest(result["training_manifest"])
    pair = json.loads((Path(result["training_manifest"]).parent.parent / "private/canonical/preference.jsonl").read_text())
    assert manifest.stage == "preference"
    assert manifest.lineage.parents == [result["candidate_dataset_id"]]
    assert len(pair["metadata"]["synthetic_candidate_ids"]) == 2
    assert pair["metadata"]["parent_record_ids"] == ["seed-record"]
    assert validate_lineage_root(Path(result["training_manifest"]).parent.parent, require_complete=True)["valid"]
    catalog = ArtifactCatalog(tmp_path)
    index = catalog.scan()
    assert any(dataset["dataset_id"] == result["training_dataset_id"]
               for build in index["builds"] for dataset in build["datasets"])
