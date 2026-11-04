"""Final comparisons require the same pinned input and inference identity."""

import pytest

from onegpu_posttrain.weight_space.evaluation import compare_final_results


def _report(correct):
    return {"role": "final_holdout", "comparison_identity": "a" * 64,
            "dataset_id": "final-suite", "dataset_sha256": "b" * 64,
            "sample_count": 81, "protocol": "choice-accuracy-v1",
            "model_revision": "pinned-revision", "processor_sha256": "c" * 64,
            "runtime_image": "image@sha256:pinned", "inference": {"max_new_tokens": 16},
            "metrics": {"overall": {"correct": correct, "denominator": 81,
                                    "accuracy": correct / 81}}}


def test_comparison_accepts_same_identity_and_coverage():
    result = compare_final_results(_report(33), _report(40))
    assert result["denominator"] == 81
    assert result["delta_accuracy"] == pytest.approx(7 / 81)


@pytest.mark.parametrize("change", [
    lambda report: report.update(role="practice"),
    lambda report: report.update(processor_sha256="d" * 64),
    lambda report: report.update(runtime_image="another-image"),
    lambda report: report["metrics"]["overall"].update(denominator=80),
])
def test_comparison_rejects_incompatible_result(change):
    candidate = _report(40)
    change(candidate)
    with pytest.raises(ValueError, match="not comparable|coverage"):
        compare_final_results(_report(33), candidate)
