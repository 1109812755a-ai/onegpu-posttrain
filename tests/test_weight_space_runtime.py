"""The Docker probe accepts only the pinned GPU-capable runtime."""

from types import SimpleNamespace

import pytest

from onegpu_posttrain.weight_space import runtime


def test_probe_rejects_changed_transformers(monkeypatch):
    monkeypatch.setattr(runtime, "docker_prefix", lambda: ["docker"])
    monkeypatch.setattr(runtime.subprocess, "run", lambda *_, **__: SimpleNamespace(
        returncode=0, stdout=b'{"torch":"2.6.0+cu124","transformers":"4.57.6","cuda":true,"gpu":"GPU"}\n'))
    with pytest.raises(RuntimeError, match="versions or GPU"):
        runtime.probe_runtime()


def test_probe_returns_pinned_runtime(monkeypatch):
    monkeypatch.setattr(runtime, "docker_prefix", lambda: ["docker"])
    monkeypatch.setattr(runtime.subprocess, "run", lambda *_, **__: SimpleNamespace(
        returncode=0, stdout=b'{"torch":"2.6.0+cu124","transformers":"5.8.0","cuda":true,"gpu":"RTX 4060"}\n'))
    assert runtime.probe_runtime().gpu == "RTX 4060"
