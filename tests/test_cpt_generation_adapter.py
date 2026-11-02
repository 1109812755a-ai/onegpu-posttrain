import json

import pytest

from onegpu_posttrain.data.decontaminate import BenchmarkIndex
from onegpu_posttrain.synthetic.cpt_generation import generate_cpt_candidates
from onegpu_posttrain.synthetic.providers import FakeProvider, ProviderProfile


def _candidate(text="Fractions describe equal parts of a whole and support clear mathematical reasoning."):
    return json.dumps({"text": text, "subtopic": "fractions"})


def _run(responses, **overrides):
    profile = ProviderProfile(profile_id="fixture", adapter="fake", model="fixture-model")
    provider = FakeProvider(responses)
    args = {
        "provider": provider,
        "profile": profile,
        "domain": "mathematics",
        "topic": "fractions",
        "target_token_budget": 10,
        "request_budget": len(responses),
        "candidate_budget": 10,
        "max_output_tokens": 32,
        "timeout_seconds": 5,
        "tokenizer_id": "fixture-tokenizer-v1",
        "count_tokens": lambda text: 10,
        "benchmark_index": BenchmarkIndex([]),
        "benchmark_index_id": "fixture-index-v1",
        "benchmark_index_sha256": "a" * 64,
    }
    args.update(overrides)
    return generate_cpt_candidates(**args), provider


def test_cpt_adapter_returns_reviewable_unverified_candidates_with_token_and_provider_evidence():
    result, provider = _run([_candidate(), _candidate("Decimals express parts using powers of ten and place value notation.")],
                            target_token_budget=20)

    assert result["status"] == "completed"
    assert result["registered"] is False
    assert result["usable_tokens"] == 20
    assert result["tokenizer_id"] == "fixture-tokenizer-v1"
    assert result["factuality"] == "not_verified"
    assert result["candidates"][0]["metadata"]["factuality"] == "not_verified"
    assert result["candidates"][0]["metadata"]["provider_profile_id"] == "fixture"
    assert provider.calls == 2


def test_cpt_adapter_deduplicates_exact_candidate_and_keeps_generating_within_request_budget():
    first = "Fractions describe equal parts of a whole and support clear mathematical reasoning."
    result, provider = _run([_candidate(first), _candidate("  " + first.upper() + "  "), _candidate(
        "Decimals express parts using powers of ten and place value notation.")],
        request_budget=3, target_token_budget=20)

    assert result["status"] == "completed"
    assert result["candidate_count"] == 2
    assert result["request_count"] == 3
    assert result["rejections"]["exact_duplicate"] == 1
    assert provider.calls == 3


def test_cpt_adapter_rejects_benchmark_contamination_against_identified_index():
    contaminated = "Fractions describe equal parts of a whole and support clear mathematical reasoning."
    index = BenchmarkIndex([("gsm8k-fixture", contaminated)])
    result, provider = _run([_candidate(contaminated)], benchmark_index=index, target_token_budget=10)

    assert result["status"] == "failed"
    assert result["registered"] is False
    assert result["rejections"]["benchmark_contamination"] == 1
    assert result["contamination_status"] == "screened_against_identified_index"
    assert provider.calls == 1


def test_cpt_adapter_reports_empty_and_partial_without_fabricating_candidates():
    empty, empty_provider = _run(["not json"], target_token_budget=10)
    partial, partial_provider = _run([_candidate()], target_token_budget=20)

    assert empty["status"] == "failed"
    assert empty["candidate_count"] == 0
    assert empty["rejections"]["invalid_candidate"] == 1
    assert partial["status"] == "partial"
    assert partial["usable_tokens"] == 10
    assert partial["shortfall_tokens"] == 10
    assert empty_provider.calls == partial_provider.calls == 1


def test_cpt_adapter_rejects_non_training_seed_and_budget_overflow():
    with pytest.raises(ValueError, match="train_seed"):
        _run([_candidate()], training_seeds=[{
            "id": "holdout-1", "text": "secret benchmark answer",
            "metadata": {"data_role": "final_holdout"},
        }])

    with pytest.raises(ValueError, match="safety bound"):
        _run([_candidate()], target_token_budget=100_000_001)
