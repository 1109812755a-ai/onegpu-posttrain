from __future__ import annotations

import json

import pytest

from onegpu_posttrain import cli
from onegpu_posttrain.cli import build_parser


def _config(path):
    path.write_text(
        "source:\n  repo: fixture/dataset\n  config: default\n  revision: pinned-rev\n",
        encoding="utf-8",
    )
    return path


def test_cache_source_help_explains_full_split_and_refresh(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["cache-source", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--config" in help_text
    assert "--cache-dir" in help_text
    assert "--refresh" in help_text


def test_cache_source_writes_verified_manifest_from_fake_source(tmp_path, monkeypatch, capsys):
    config = _config(tmp_path / "source.yaml")
    monkeypatch.setattr(cli, "iter_hf_dataset", lambda *args, **kwargs: iter([{"id": 1}, {"id": 2}]))
    args = build_parser().parse_args([
        "cache-source", "--config", str(config), "--cache-dir", str(tmp_path / "cache"),
    ])

    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "cached"
    assert payload["row_count"] == 2
    assert payload["byte_count"] > 0
    assert payload["source"]["revision"] == "pinned-rev"
    assert (tmp_path / "cache" / payload["cache_key"][:24] / "manifest.json").is_file()


def test_cache_source_reports_failed_incomplete_fetch(tmp_path, monkeypatch, capsys):
    config = _config(tmp_path / "source.yaml")

    def interrupted(*args, **kwargs):
        yield {"id": 1}
        raise OSError("source connection dropped")

    monkeypatch.setattr(cli, "iter_hf_dataset", interrupted)
    args = build_parser().parse_args([
        "cache-source", "--config", str(config), "--cache-dir", str(tmp_path / "cache"),
    ])

    assert args.func(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert "source connection dropped" in payload["error"]
    entries = list((tmp_path / "cache").iterdir())
    assert len(entries) == 1
    entry = entries[0]
    assert not (entry / "manifest.json").exists()
    assert not list(entry.glob("*.partial"))
