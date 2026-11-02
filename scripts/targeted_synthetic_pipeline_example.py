"""Offline end-to-end example for candidate validation and policy export."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from onegpu_posttrain.data.decontaminate import BenchmarkIndex
from onegpu_posttrain.data.io import write_jsonl
from onegpu_posttrain.data.policy import DataPolicy, PolicyTokenizer
from onegpu_posttrain.data.roles import register_dataset, save_role_registry
from onegpu_posttrain.data.targeted import GenerationParameters
from onegpu_posttrain.data.tokenization import WhitespaceTokenizer
from onegpu_posttrain.lineage_validation import validate_lineage_root
from onegpu_posttrain.synthetic.artifacts import write_targeted_artifacts
from onegpu_posttrain.synthetic.candidates import validate_candidate_response
from onegpu_posttrain.synthetic.policy_export import (
    CandidatePolicyInput,
    export_candidates_with_policy,
)
from onegpu_posttrain.synthetic.providers import (
    FakeProvider,
    GenerationRequest,
    OutputProtocol,
    ProviderProfile,
)


def main() -> int:
    question = "A rectangle has perimeter 26 and length 8. What is its width?"
    reference_answer = str(26 // 2 - 8)  # independently computed from 2(L+W)=P
    response = json.dumps({
        "problem_id": "fixture-rectangle-width",
        "solution": "For a rectangle, 2(L+W)=26, so L+W=13. With L=8, W=13-8=5. Final answer: 5",
        "final_answer": "5",
    })
    profile = ProviderProfile(profile_id="fake-local", adapter="fake", model="fixture-responder-v1",
                              revision="fixture-v1")
    protocol = OutputProtocol(
        protocol_id="sft-math-candidate", version="v1",
        required_fields=["problem_id", "solution", "final_answer"],
    )
    provider = FakeProvider([response])
    provider_response = provider.generate(profile, GenerationRequest(
        prompt=f"Solve the training-seed question: {question}",
        max_output_tokens=128,
        output_protocol=protocol,
    ))

    with tempfile.TemporaryDirectory(prefix=".tmp-targeted-example-", dir=Path.cwd()) as temp:
        root = Path(temp)
        seed_path = root / "train_seed.jsonl"
        write_jsonl(seed_path, [{
            "id": "seed-rectangle-001",
            "problem_id": "fixture-rectangle-width",
            "question": question,
            "topic": "algebra",
        }])
        registry = register_dataset(
            None, seed_path, dataset_id="fixture-targeted-train-seeds-v1", role="train_seed",
        )
        registry_path = root / "role_registry.json"
        save_role_registry(registry, registry_path)
        candidate = validate_candidate_response(
            provider_response.text,
            source_record_id="seed-rectangle-001",
            expected_problem_id="fixture-rectangle-width",
            candidate_slot=0,
            request_sha256=hashlib.sha256(b"fixture-request-v1").hexdigest(),
            finish_reason=provider_response.finish_reason,
            question=question,
            protocol=protocol,
            tokenizer=WhitespaceTokenizer(),
            max_training_tokens=256,
        )
        policy = DataPolicy(stage="sft", tokenizer=PolicyTokenizer(test_only=True))
        policy_export = export_candidates_with_policy(
            [CandidatePolicyInput(candidate=candidate, question=question,
                                  reference_answer=reference_answer)],
            policy=policy,
            tokenizer=WhitespaceTokenizer(),
            registry=registry,
            protected_index=BenchmarkIndex([("unrelated-holdout", "An unrelated private test item.")]),
            protected_index_id="fixture-protected-index-v1",
            protected_index_sha256="a" * 64,
        )
        plan = SimpleNamespace(
            plan_id="plan-targeted-offline-example",
            training_seed=SimpleNamespace(
                dataset_id="fixture-targeted-train-seeds-v1",
                content_sha256=hashlib.sha256(seed_path.read_bytes()).hexdigest(),
            ),
            prompt_template_sha256=hashlib.sha256(b"offline-example-template-v1").hexdigest(),
            generation_parameters=GenerationParameters(temperature=0.0, provider_seed=42),
            budget=SimpleNamespace(target_candidates=1),
            training_budget=1,
        )
        build_dir = root / "build-001"
        artifact_summary = write_targeted_artifacts(
            output_dir=build_dir,
            plan=plan,
            profile=profile,
            tokenizer=WhitespaceTokenizer(),
            candidates=[candidate],
            policy_export=policy_export,
            generation_summary={
                "request_count": 1,
                "successful_candidates": 1,
                "target_candidate_count": 1,
                "ledger": {"request_attempts": 1, "cost_status": "unknown"},
                "network_used": False,
                "execution_status": "generated_not_trained_not_reevaluated",
            },
            validation_summary={"candidate_count": 1, "valid_format": 1},
        )
        lineage = validate_lineage_root(build_dir, require_complete=True)
        print(json.dumps({
            "evidence_kind": "fixture",
            "provider_adapter": "fake",
            "provider_calls": provider.calls,
            "network_used": False,
            "trusted_reference_source": "programmatic rectangle perimeter calculation",
            "candidate_validation": candidate.validation_status,
            "answer_correctness": policy_export["assessment_counts"],
            "before_after_data_counts": {
                "generated_candidates": len([candidate]),
                "policy_selected_training_records": policy_export["selected_count"],
            },
            "quality_gate_status": json.loads(Path(artifact_summary["quality_report"]).read_text(
                encoding="utf-8"))["gate"]["summary_status"],
            "lineage_errors": lineage["errors"],
            "model_effect": "not_evaluated",
            "execution_status": "generated_not_trained_not_reevaluated",
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
