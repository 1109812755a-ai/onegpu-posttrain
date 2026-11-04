"""Pinned registry lookups reject changed model or dataset evidence."""

from types import SimpleNamespace

import pytest

from onegpu_posttrain.weight_space import registry


def test_resolver_rejects_changed_registered_model(tmp_path, monkeypatch):
    class ChangedModelAdapter:
        def __init__(self, workspace):
            assert workspace == tmp_path

        def _registered_base_model(self, model_id):
            assert model_id == registry.MODEL_ID
            return SimpleNamespace(digest=SimpleNamespace(value="0" * 64)), None

    monkeypatch.setattr(registry, "StageEvaluationAdapter", ChangedModelAdapter)
    with pytest.raises(ValueError, match="checkpoint digest changed"):
        registry.resolve_inputs(tmp_path, "exp-weight-space-fixture01")


def test_dataset_lookup_rejects_changed_canonical_digest(tmp_path, monkeypatch):
    dataset_id = registry.PRACTICE_DATASET_ID
    manifest = SimpleNamespace(
        dataset_id=dataset_id, stage="eval", sample_count=300,
        hashes={"canonical_records_sha256": "0" * 64},
        metadata={"dataset_role": "evaluation_only"},
    )
    monkeypatch.setattr(registry, "_find_manifest", lambda *_: tmp_path / "manifest.yaml")
    monkeypatch.setattr(registry, "load_manifest", lambda *_: manifest)
    monkeypatch.setattr(registry, "validate_manifest_evidence", lambda *_: [])
    monkeypatch.setattr(registry, "resolve_canonical_path", lambda *_: tmp_path / "eval.jsonl")
    with pytest.raises(ValueError, match="registered dataset identity changed"):
        registry._dataset(tmp_path, dataset_id, registry.PRACTICE_DATASET_SHA256, 300, "practice")
