from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import yaml

from onegpu_posttrain import cli
from onegpu_posttrain.cli import build_parser
from onegpu_posttrain.data.manifest import load_manifest
from onegpu_posttrain.data.mini_readiness import evaluate_mini_readiness
from onegpu_posttrain.data.source_cache import SourceIdentity, create_source_cache
from onegpu_posttrain.lineage_validation import validate_lineage_root


def _openr1_rows(count: int = 80):
    for index in range(count):
        yield {
            "problem": f"Compute x in x + {index} = {index + 2}.",
            "solution": "2",
            "answer": "2",
            "problem_type": "algebra",
            "question_type": "math-word-problem",
            "source": "fixture",
            "uuid": f"row-{index}",
            "is_reasoning_complete": [True, True],
            "generations": [f"Subtract {index}; final answer: 2", f"Incorrect; final answer: {index + 3}"],
            "correctness_math_verify": [True, False],
            "correctness_llama": [True, False],
            "finish_reasons": ["stop", "stop"],
            "correctness_count": 1,
        }


def _manifest_evidence(output, stage: str):
    processing = json.loads((output / "reports/processing_report.json").read_text(encoding="utf-8"))
    manifest = load_manifest(output / f"manifests/{stage}-smoke-v1.yaml")
    return processing, manifest


def test_openr1_build_uses_verified_cache_offline_and_records_provenance(tmp_path, monkeypatch, capsys):
    config = tmp_path / "openr1.yaml"
    config.write_text(
        "source:\n  repo: fixture/openr1\n  config: default\n  revision: pinned-openr1\nsplit_seed: 42\n",
        encoding="utf-8",
    )
    source = yaml.safe_load(config.read_text(encoding="utf-8"))["source"]
    identity = SourceIdentity(**source, split="train")
    cache_manifest, _, _ = create_source_cache(
        _openr1_rows(), cache_root=tmp_path / "cache", identity=identity
    )
    monkeypatch.setattr(cli, "iter_hf_dataset", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network source called")))
    output = tmp_path / "openr1-build"
    args = build_parser().parse_args([
        "build-openr1", "--config", str(config), "--output", str(output),
        "--source-cache", str(cache_manifest), "--test-tokenizer",
        "--sft-target", "1", "--alignment-target", "1", "--holdout-target", "1",
        "--preference-target", "1", "--preference-eval-target", "1", "--grpo-target", "1",
    ])

    assert args.func(args) == 0
    capsys.readouterr()
    processing, manifest = _manifest_evidence(output, "sft")
    acquisition = processing["source_acquisition"]
    assert acquisition["mode"] == "verified_cache"
    assert acquisition["cache_key"] == identity.cache_key
    assert acquisition["row_count"] == 80
    assert len(acquisition["manifest_sha256"]) == 64
    assert manifest.metadata["source_acquisition"] == acquisition
    assert processing["source"]["revision"] == "pinned-openr1"

    lineage = validate_lineage_root(output, require_complete=True)
    assert lineage["valid"] is True
    profile = tmp_path / "mini-profile.yaml"
    profile.write_text(
        "schema_version: '1.0'\nprofile_id: cache-integration\nprofile_version: '1'\n"
        "description: cache integration fixture\ndatasets:\n  - dataset_key: openr1.sft_train\n"
        "    stage: sft\n    required: true\n    target: 1\n    unit: examples\n"
        "    count_source: processing_report.outputs.sft.budget_count\n"
        "    quality_report_required: true\n    processing_report_required: true\n"
        "    complete_lineage_required: true\n",
        encoding="utf-8",
    )
    readiness = evaluate_mini_readiness(output, profile)
    sft_readiness = readiness.datasets[0]
    assert sft_readiness.processing_evidence.status == "pass"
    assert sft_readiness.lineage.status == "pass"
    assert sft_readiness.quality_evidence.status == "not_evaluated"
    assert str(tmp_path) not in json.dumps(acquisition)


def test_finemath_build_uses_verified_cache_offline_and_records_provenance(tmp_path, monkeypatch, capsys):
    config = tmp_path / "finemath.yaml"
    config.write_text(
        "source:\n  repo: fixture/finemath\n  config: finemath-4plus\n  revision: pinned-finemath\nsplit_seed: 42\n",
        encoding="utf-8",
    )
    source = yaml.safe_load(config.read_text(encoding="utf-8"))["source"]
    identity = SourceIdentity(**source, split="train")
    rows = ({"url": f"fixture:{index}", "text": f"A distinct math document {index}. " * 12} for index in range(30))
    cache_manifest, _, _ = create_source_cache(
        rows, cache_root=tmp_path / "cache", identity=identity
    )
    monkeypatch.setattr(cli, "iter_hf_dataset", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network source called")))
    output = tmp_path / "finemath-build"
    args = build_parser().parse_args([
        "build-finemath", "--config", str(config), "--output", str(output),
        "--source-cache", str(cache_manifest), "--test-tokenizer",
        "--train-token-budget", "30", "--holdout-token-budget", "10",
        "--max-seq-tokens", "20", "--holdout-permille", "500",
    ])

    assert args.func(args) == 0
    capsys.readouterr()
    processing, manifest = _manifest_evidence(output, "cpt")
    acquisition = processing["source_acquisition"]
    assert acquisition["mode"] == "verified_cache"
    assert acquisition["cache_key"] == identity.cache_key
    assert acquisition["row_count"] == 30
    assert manifest.metadata["source_acquisition"] == acquisition
    assert processing["source"]["revision"] == "pinned-finemath"


def test_source_iterator_preserves_remote_stream_and_local_jsonl_modes(tmp_path, monkeypatch):
    remote_calls = []
    monkeypatch.setattr(cli, "iter_hf_dataset", lambda *args, **kwargs: remote_calls.append((args, kwargs)) or iter([{"id": 1}]))
    config = {"source": {"repo": "fixture/data", "config": "default", "revision": "rev-1"}}
    remote_args = SimpleNamespace(input_jsonl=None, source_cache=None)
    remote_rows, remote_acquisition = cli._source_iter(remote_args, config)
    assert list(remote_rows) == [{"id": 1}]
    assert remote_acquisition == {"mode": "remote_stream"}
    assert remote_calls[0][1]["streaming"] is True

    local_path = tmp_path / "rows.jsonl"
    local_path.write_text('{"id":2}\n', encoding="utf-8")
    local_args = SimpleNamespace(input_jsonl=str(local_path), source_cache=None)
    local_rows, local_acquisition = cli._source_iter(local_args, config)
    assert list(local_rows) == [{"id": 2}]
    assert local_acquisition["mode"] == "local_jsonl"
    assert len(local_acquisition["content_sha256"]) == 64


def test_build_rejects_conflicting_source_inputs():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([
            "build-openr1", "--config", "config.yaml", "--output", "out",
            "--input-jsonl", "rows.jsonl", "--source-cache", "cache/manifest.json",
        ])
    assert exc.value.code == 2


@pytest.mark.parametrize("problem", ["revision", "tampered"])
def test_build_rejects_bad_cache_without_remote_fallback(tmp_path, monkeypatch, capsys, problem):
    config = tmp_path / "openr1.yaml"
    config.write_text(
        "source:\n  repo: fixture/openr1\n  config: default\n  revision: pinned-openr1\nsplit_seed: 42\n",
        encoding="utf-8",
    )
    identity = SourceIdentity(repo="fixture/openr1", config="default", revision="pinned-openr1", split="train")
    cache_manifest, manifest, _ = create_source_cache(
        _openr1_rows(5), cache_root=tmp_path / "cache", identity=identity
    )
    if problem == "revision":
        config.write_text(
            "source:\n  repo: fixture/openr1\n  config: default\n  revision: other-revision\nsplit_seed: 42\n",
            encoding="utf-8",
        )
    else:
        (cache_manifest.parent / manifest.data_file).write_bytes(b'{"id":99}\n')

    monkeypatch.setattr(cli, "iter_hf_dataset", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not fallback to network")))
    args = build_parser().parse_args([
        "build-openr1", "--config", str(config), "--output", str(tmp_path / "out"),
        "--source-cache", str(cache_manifest), "--test-tokenizer",
    ])
    assert args.func(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "invalid_source"
    assert "mismatch" in payload["error"].lower() or "sha256" in payload["error"].lower()
