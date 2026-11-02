from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from onegpu_posttrain import cli
from onegpu_posttrain.ui import server


def test_serve_ui_cli_defaults_to_loopback_and_forwards_remote_options(monkeypatch, tmp_path):
    captured = {}

    def fake_serve(workspace: Path, **kwargs):
        captured["workspace"] = workspace
        captured.update(kwargs)

    monkeypatch.setattr(server, "serve", fake_serve)
    args = cli.build_parser().parse_args(["serve-ui", "--workspace", str(tmp_path)])
    assert args.host == "127.0.0.1"
    assert args.allow_remote is False
    assert cli.cmd_serve_ui(args) == 0
    assert captured == {
        "workspace": tmp_path,
        "port": 8765,
        "host": "127.0.0.1",
        "allow_remote": False,
        "enable_policy_editor": False,
        "enable_experiment_launcher": False,
        "enable_synthetic_generation": False,
    }

    args = cli.build_parser().parse_args([
        "serve-ui", "--host", "0.0.0.0", "--port", "8766", "--allow-remote",
    ])
    assert args.host == "0.0.0.0"
    assert args.allow_remote is True

    launcher_args = cli.build_parser().parse_args([
        "serve-ui", "--enable-experiment-launcher",
    ])
    assert launcher_args.enable_experiment_launcher is True


def test_non_loopback_bind_is_rejected_before_opening_server(monkeypatch, tmp_path):
    uvicorn_stub = SimpleNamespace(run=lambda *args, **kwargs: pytest.fail("server must not start"))
    monkeypatch.setitem(__import__("sys").modules, "uvicorn", uvicorn_stub)

    with pytest.raises(SystemExit, match="--allow-remote"):
        server.serve(tmp_path, host="0.0.0.0")


def test_launcher_refuses_remote_mode_before_opening_server(monkeypatch, tmp_path):
    uvicorn_stub = SimpleNamespace(run=lambda *args, **kwargs: pytest.fail("server must not start"))
    monkeypatch.setitem(__import__("sys").modules, "uvicorn", uvicorn_stub)
    with pytest.raises(SystemExit, match="training/generation"):
        server.serve(tmp_path, host="0.0.0.0", allow_remote=True, enable_experiment_launcher=True)


def test_synthetic_generation_refuses_remote_mode_before_opening_server(monkeypatch, tmp_path):
    uvicorn_stub = SimpleNamespace(run=lambda *args, **kwargs: pytest.fail("server must not start"))
    monkeypatch.setitem(__import__("sys").modules, "uvicorn", uvicorn_stub)
    with pytest.raises(SystemExit, match="training/generation"):
        server.serve(tmp_path, host="0.0.0.0", allow_remote=True, enable_synthetic_generation=True)


def test_remote_opt_in_warns_and_binds_requested_address(monkeypatch, tmp_path, capsys):
    calls = {}
    monkeypatch.setattr(server, "create_app", lambda *args, **kwargs: (args, kwargs))
    monkeypatch.setitem(
        __import__("sys").modules,
        "uvicorn",
        SimpleNamespace(run=lambda app, **kwargs: calls.update(app=app, **kwargs)),
    )

    server.serve(tmp_path, port=8766, host="0.0.0.0", allow_remote=True)

    output = capsys.readouterr().out
    assert "WARNING" in output
    assert "No firewall rules are changed" in output
    assert calls["host"] == "0.0.0.0"
    assert calls["port"] == 8766
    assert calls["app"][1]["allow_remote"] is True
