from __future__ import annotations

import json

import pytest

from onegpu_posttrain.data.manifest import load_manifest
from onegpu_posttrain.data.ordinary_preference import (
    build_human_reviewed_preference_pairs,
    generate_ordinary_preference_candidates,
    register_human_reviewed_preference_pairs,
)
from onegpu_posttrain.synthetic.providers import (
    FakeProvider,
    GenerationResponse,
    ProviderProfile,
)


def test_ordinary_candidates_are_unranked_until_explicit_same_prompt_review(tmp_path):
    provider = FakeProvider([
        '{"completion":"A concise response."}',
        '{"completion":"A detailed response."}',
        '{"completion":"Another answer."}',
        '{"completion":"A different answer."}',
    ])
    profile = ProviderProfile(profile_id="offline", adapter="fake", model="fixture")
    generated = generate_ordinary_preference_candidates(
        ["What is a preference pair?", "Explain a second concept."], profile=profile,
        provider=provider, request_budget=3, candidate_budget=4, max_output_tokens=100,
    )
    assert generated["status"] == "partial"
    assert generated["request_count"] == provider.calls == 3
    assert generated["trainable"] is False
    assert all(row["ordering_status"] == "unreviewed" for row in generated["candidates"])

    first, second = generated["candidates"][:2]
    reviewed = build_human_reviewed_preference_pairs(
        generated["candidates"], [{
            "problem_id": first["problem_id"],
            "chosen_candidate_id": first["candidate_id"],
            "rejected_candidate_id": second["candidate_id"],
            "rationale": "The chosen response directly answers and explains the concept.",
        }], reviewer_id="local-user",
    )
    assert reviewed["usable_pairs"] == 1
    row = reviewed["records"][0]
    evidence = row["metadata"]["ordering_evidence"]
    assert row["metadata"]["ordering_evidence_status"] == "human_verified"
    assert evidence["scope"] == "user_declared_ranking; reviewer_identity_not_independently_authenticated"
    assert row["chosen"] == first["completion"]
    assert row["rejected"] == second["completion"]

    registered = register_human_reviewed_preference_pairs(
        reviewed, workspace=tmp_path, output_root=tmp_path / "data/processed/generated",
        provider_profile_id="offline", provider_model="fixture", provider_revision=None,
    )
    manifest_path = tmp_path / registered["registration_path"] / "manifests/preference-model-generated-v1.yaml"
    manifest = load_manifest(manifest_path)
    assert manifest.dataset_id == registered["dataset_id"]
    assert manifest.lineage.parents == []
    assert manifest.hashes["canonical_records_sha256"] == registered["canonical_sha256"]
    assert manifest.metadata["ordering_evidence_status"] == "human_verified"
    assert manifest.metadata["reviewer_identity_status"] == "not_independently_authenticated"
    canonical = tmp_path / registered["registration_path"] / "canonical/preference.jsonl"
    saved = json.loads(canonical.read_text(encoding="utf-8"))
    assert saved["metadata"]["ordering_evidence"]["chosen_candidate_id"] == first["candidate_id"]
    assert saved["metadata"]["data_role"] == "train_pairs"


def test_manual_review_cannot_mix_problems_or_forge_provider_labels():
    provider = FakeProvider([
        '{"completion":"answer a", "chosen":true}',
        '{"completion":"answer b"}',
    ])
    profile = ProviderProfile(profile_id="offline", adapter="fake", model="fixture")
    generated = generate_ordinary_preference_candidates(
        ["Question A", "Question B"], profile=profile, provider=provider,
        request_budget=2, candidate_budget=2, max_output_tokens=40,
    )
    assert generated["candidate_count"] == 1
    assert generated["request_errors"] == {"invalid_or_rank_bearing_response": 1}
    candidate = generated["candidates"][0]
    other_problem = {**candidate, "candidate_id": "other", "problem_id": "other-p"}
    result = build_human_reviewed_preference_pairs(
        [candidate, other_problem], [{
            "problem_id": candidate["problem_id"], "chosen_candidate_id": candidate["candidate_id"],
            "rejected_candidate_id": "other", "rationale": "looks somewhat better",
        }], reviewer_id="operator",
    )
    assert result["records"] == []
    assert result["rejections"]["review_candidates_do_not_match_same_problem"] == 1
    with pytest.raises(ValueError, match="审阅者标识"):
        build_human_reviewed_preference_pairs([], [], reviewer_id=" ")


def test_ordinary_preference_uses_configured_provider_timeout():
    class CapturingProvider:
        timeout = None

        def generate(self, profile, request):
            self.timeout = request.timeout_seconds
            return GenerationResponse(text='{"completion":"A bounded response."}')

    provider = CapturingProvider()
    report = generate_ordinary_preference_candidates(
        ["Question"], profile=ProviderProfile(profile_id="offline", adapter="fake", model="fixture"),
        provider=provider, request_budget=1, candidate_budget=1, max_output_tokens=80, timeout_seconds=211,
    )
    assert provider.timeout == report["timeout_seconds"] == 211


def test_empty_or_unauditable_preference_report_cannot_be_registered(tmp_path):
    for report in ({"stage": "preference", "records": [], "usable_pairs": 0},
                   {"stage": "preference", "records": [{"metadata": {}}], "usable_pairs": 1}):
        with pytest.raises(ValueError):
            register_human_reviewed_preference_pairs(
                report, workspace=tmp_path, output_root=tmp_path / "data/processed/generated",
                provider_profile_id="offline", provider_model="fixture", provider_revision=None,
            )
