from onegpu_posttrain.synthetic.provider_identity import provider_identity_evidence


def test_profile_name_matching_registered_model_does_not_claim_identity_without_attestation():
    evidence = provider_identity_evidence(
        profile_id="ollama-qwen", model="model-registered-123", revision="r42",
        adapter="openai_compatible",
    )
    assert evidence["provider_model"] == "model-registered-123"
    assert evidence["provider_revision"] == "r42"
    assert evidence["provider_revision_status"] == "available"
    assert evidence["provider_identity_status"] == "profile_declared_not_independently_verified"
    assert evidence["registered_model_association_status"] == "not_proven"
    assert evidence["registered_model_id"] is None
    assert evidence["registered_model_sha256"] is None


def test_missing_provider_revision_is_explicitly_unavailable_not_inferred():
    evidence = provider_identity_evidence(
        profile_id="remote", model="Qwen3-0.6B", revision=None, adapter="openai_compatible",
    )
    assert evidence["provider_revision"] is None
    assert evidence["provider_revision_status"] == "unavailable"
    assert evidence["registered_model_id"] is None
    assert evidence["registered_model_sha256"] is None
