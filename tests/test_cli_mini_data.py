import json

import pytest

from onegpu_posttrain import cli
from onegpu_posttrain.cli import build_parser


def test_validate_mini_data_help_exposes_explicit_inputs(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["validate-mini-data", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--root" in help_text
    assert "--profile" in help_text
    assert "--quality-assessment" in help_text


def test_assess_mini_data_quality_help_exposes_local_inputs(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["assess-mini-data-quality", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--root" in help_text
    assert "--profile" in help_text


def test_validate_mini_data_writes_reports_and_blocks_empty_artifact_root(tmp_path, capsys):
    root = tmp_path / "empty-artifacts"
    root.mkdir()
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "schema_version: '1.0'\nprofile_id: test\nprofile_version: '1'\n"
        "description: fixture\ndatasets:\n  - dataset_key: sample.sft\n"
        "    stage: sft\n    target: 1\n    unit: examples\n"
        "    count_source: processing_report.outputs.sft.budget_count\n",
        encoding="utf-8",
    )
    args = build_parser().parse_args([
        "validate-mini-data", "--root", str(root), "--profile", str(profile),
    ])

    assert args.func(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "not_ready"
    assert (root / "reports/mini-readiness" / payload["assessment_id"] / "mini_readiness.json").is_file()
    assert (root / "reports/mini-readiness" / payload["assessment_id"] / "mini_readiness.md").is_file()


def test_validate_mini_data_malformed_profile_returns_nonzero(tmp_path, capsys):
    profile = tmp_path / "bad.yaml"
    profile.write_text("datasets: [", encoding="utf-8")
    args = build_parser().parse_args([
        "validate-mini-data", "--root", str(tmp_path), "--profile", str(profile),
    ])

    assert args.func(args) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "invalid_input"


def test_validate_mini_data_returns_zero_for_ready_assessment(tmp_path, monkeypatch, capsys):
    class ReadyReport:
        def __init__(self):
            self.overall_status = "ready"
            self.assessment_id = "mini-check-ready"
            self.blockers = []
            self.warnings = []

    monkeypatch.setattr(
        cli, "evaluate_mini_readiness",
        lambda root, profile, quality_assessment=None, quality_review=None: ReadyReport(),
    )
    monkeypatch.setattr(
        cli,
        "write_mini_readiness_report",
        lambda report, output: (tmp_path / "ready.json", tmp_path / "ready.md"),
    )
    args = build_parser().parse_args([
        "validate-mini-data", "--root", str(tmp_path), "--profile", "profile.yaml",
    ])

    assert args.func(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ready"
