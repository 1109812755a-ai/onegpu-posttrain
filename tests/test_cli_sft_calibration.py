from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from onegpu_posttrain import cli
from onegpu_posttrain.data.sft_calibration import (
    CalibrationError,
    calibrate_sft_lengths,
)
from onegpu_posttrain.data.tokenization import WhitespaceTokenizer


def test_cli_missing_local_source_fails_without_network_or_output(tmp_path, capsys):
    output = tmp_path / "calibration"
    args = cli.build_parser().parse_args([
        "calibrate-sft-length", "--source", str(tmp_path / "missing"), "--output", str(output),
    ])
    assert args.func(args) == 2
    assert "不会访问网络" in capsys.readouterr().out
    assert not output.exists()


def test_official_mini_output_is_blocked(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(CalibrationError, match="官方 Mini 路径"):
        calibrate_sft_lengths(
            source=tmp_path / "missing",
            output=tmp_path / "data" / "processed" / "mini-v1" / "calibration",
            tokenizer=WhitespaceTokenizer(),
        )


def test_gpu_probe_is_opt_in_and_preserves_each_run_record(tmp_path, monkeypatch):
    output = tmp_path / "calibration"
    model = tmp_path / "local-model"
    model.mkdir()
    native_files = []
    for cap in (1024, 2048, 4096):
        native = tmp_path / f"native-{cap}.jsonl"
        native.write_text('{"messages": []}\n', encoding="utf-8")
        native_files.append(native)

    def fake_calibrate(**kwargs):
        kwargs["output"]
        output.mkdir()
        candidates = [{
            "max_length": cap,
            "training_examples": 1,
            "native_path": str(native),
            "content_sha256": f"{cap:064x}",
        } for cap, native in zip((1024, 2048, 4096), native_files)]
        report = {"calibration_id": "sft-length-test", "candidates": candidates,
                  "interpretation": [], "probes": []}
        (output / "calibration_report.json").write_text(json.dumps(report), encoding="utf-8")
        return report

    statuses = iter(("PASS", "OOM", "FAILED"))
    probe_training_configs = []

    def fake_run_sft(args):
        probe_training_configs.append(yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["training"])
        status = next(statuses)
        probe = Path(args.output)
        probe.mkdir(exist_ok=True)
        (probe / "run.json").write_text(json.dumps({
            "run_id": f"run-{status.lower()}-{args.config[-9:-5]}",
            "train": {"status": status, "wall_seconds": 1.5,
                      "peak_gpu_memory_mib_sampled": 4096 if status == "PASS" else None},
        }), encoding="utf-8")
        return 0 if status == "PASS" else 4

    monkeypatch.setattr(cli, "calibrate_sft_lengths", fake_calibrate)
    monkeypatch.setattr(cli, "cmd_run_sft", fake_run_sft)
    args = cli.build_parser().parse_args([
        "calibrate-sft-length", "--source", str(tmp_path / "source"), "--output", str(output),
        "--run-probes", "--model", str(model), "--probe-steps", "2",
    ])
    assert args.func(args) == 0
    report = json.loads((output / "calibration_report.json").read_text(encoding="utf-8"))
    assert [probe["status"] for probe in report["probes"]] == ["PASS", "OOM", "FAILED"]
    assert report["probes"][1]["failure_reason"] == "out_of_memory"
    assert "hardware-feasibility" in report["interpretation"][-1]
    assert all((output / f"length-{cap}" / "probe" / "run.json").is_file() for cap in (1024, 2048, 4096))
    fixed = [item["max_length"] for item in probe_training_configs]
    fixed_profiles = [{key: value for key, value in item.items() if key != "max_length"}
                      for item in probe_training_configs]
    assert fixed == [1024, 2048, 4096]
    assert all(config["max_steps"] == 2 and config["report_to"] == "none" for config in fixed_profiles)
    assert all(config == fixed_profiles[0] for config in fixed_profiles)


def test_cpu_only_cli_never_runs_sft_probe(tmp_path, monkeypatch):
    output = tmp_path / "calibration"
    received = {}

    def fake_calibrate(**kwargs):
        received.update(kwargs)
        output.mkdir()
        report = {"calibration_id": "sft-length-cpu", "candidates": [], "interpretation": [], "probes": []}
        (output / "calibration_report.json").write_text(json.dumps(report), encoding="utf-8")
        return report

    def unexpected_probe(_args):
        raise AssertionError("CPU-only invocation must not run training")

    monkeypatch.setattr(cli, "calibrate_sft_lengths", fake_calibrate)
    monkeypatch.setattr(cli, "cmd_run_sft", unexpected_probe)
    args = cli.build_parser().parse_args([
        "calibrate-sft-length", "--source", str(tmp_path / "source"), "--output", str(output),
        "--answer-verification", "semantic",
    ])
    assert args.func(args) == 0
    assert received["answer_verification"] == "semantic"


def test_gpu_probe_refuses_nonlocal_model_without_writing_output(tmp_path):
    args = cli.build_parser().parse_args([
        "calibrate-sft-length", "--source", str(tmp_path / "source"), "--output", str(tmp_path / "out"),
        "--run-probes", "--model", "hub/model-id",
    ])
    with pytest.raises(SystemExit, match="locally available"):
        args.func(args)
    assert not (tmp_path / "out").exists()


def test_gpu_probe_requires_model_before_building_artifacts(tmp_path, capsys):
    args = cli.build_parser().parse_args([
        "calibrate-sft-length", "--source", str(tmp_path / "source"), "--output", str(tmp_path / "out"),
        "--run-probes",
    ])
    with pytest.raises(SystemExit, match="--model is required"):
        args.func(args)
    assert not (tmp_path / "out").exists()
