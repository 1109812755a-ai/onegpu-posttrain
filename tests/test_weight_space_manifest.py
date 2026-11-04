"""Reject incomplete or conflated experiment identities before edits can start."""

import pytest
from pydantic import ValidationError

from onegpu_posttrain.weight_space.manifest import DOCKER_IMAGE, ExperimentManifest


def _manifest():
    model_id = "model-89d12ba9eb8fb14d6eb6"
    final_id = "ds-eval-upload-54ca42ca021d295d8390"
    final_digest = "19f02eab2e2d3894269eda8f41115c22c2f203cfe3c3cbb3604ab19488004ac2"
    return {
        "schema_version": "1.0", "experiment_id": "exp-weight-space-fixture01",
        "target": {"model_id": model_id, "repository": "Qwen/Qwen3.5-0.8B",
                   "revision": "05f056712f8c582c8708395156518d963b46fb29",
                   "checkpoint_sha256": "89d12ba9eb8fb14d6eb6b8b2ea85688cd254d98f64f2bfeee85db0f3d741a4be",
                   "processor_sha256": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"},
        "runtime": {"image": DOCKER_IMAGE, "torch_version": "2.6.0+cu124",
                    "transformers_version": "5.8.0",
                    "model_loader": "AutoProcessor+AutoModelForMultimodalLM"},
        "practice": {"dataset_id": "ds-eval-upload-14aee30e5c7ec6ec0959",
                     "role": "practice", "canonical_sha256": "361c56f60059392c82e092762121c6ef50d53496dc0f6902cdd330f304b2c0be", "record_count": 300},
        "final_holdout": {"dataset_id": final_id, "role": "final_holdout",
                          "canonical_sha256": final_digest, "record_count": 81},
        "inference": {"prompt_template": "qwen3.5-processor-chat-template",
                      "scoring_protocol": "choice-accuracy-v1", "max_new_tokens": 16,
                      "do_sample": False, "skip_special_tokens": True},
        "baseline": {"model_id": model_id, "dataset_id": final_id,
                     "input_sha256": final_digest, "correct": 33, "denominator": 81,
                     "parse_failures": 0, "evidence": "observed_container_run"},
        "editor_model": "qwen3.8-max",
    }


def test_manifest_accepts_pinned_distinct_identities():
    assert ExperimentManifest.model_validate(_manifest()).final_holdout.record_count == 81


@pytest.mark.parametrize("mutation", [
    lambda data: data.pop("runtime"),
    lambda data: data["target"].pop("revision"),
    lambda data: data["practice"].update(canonical_sha256=data["final_holdout"]["canonical_sha256"]),
    lambda data: data["baseline"].update(dataset_id=data["practice"]["dataset_id"]),
    lambda data: data["inference"].update(max_new_tokens=32),
])
def test_manifest_rejects_missing_or_conflicting_identity(mutation):
    data = _manifest()
    mutation(data)
    with pytest.raises(ValidationError):
        ExperimentManifest.model_validate(data)
