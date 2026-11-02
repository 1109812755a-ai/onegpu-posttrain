from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace

import pytest

from onegpu_posttrain.evaluation import evalscope_runner


def test_run_evalscope_uses_utf8_replacement_for_windows_child_output(monkeypatch):
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="中文 output", stderr="")

    monkeypatch.setattr(evalscope_runner, "resolve_console_command", lambda command: command)
    monkeypatch.setattr(evalscope_runner.subprocess, "run", fake_run)

    result = evalscope_runner.run_evalscope(["evalscope", "eval"])

    assert seen["encoding"] == "utf-8"
    assert seen["errors"] == "replace"
    assert result["stdout"] == "中文 output"


def test_run_evalscope_failure_is_reported_without_decode_failure(monkeypatch):
    monkeypatch.setattr(evalscope_runner, "resolve_console_command", lambda command: command)
    monkeypatch.setattr(
        evalscope_runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=2, stdout="", stderr="错误: failed"),
    )

    with pytest.raises(RuntimeError, match=r"EvalScope failed \(2\): 错误: failed"):
        evalscope_runner.run_evalscope(["evalscope", "eval"])


def test_run_evalscope_streams_child_output_and_preserves_exit_code(monkeypatch):
    class FakeProcess:
        stdout = io.StringIO("step one\nstep two\n")
        stderr = io.StringIO("warning line\n")
        returncode = 0

        def wait(self):
            return self.returncode

    monkeypatch.setattr(evalscope_runner, "resolve_console_command", lambda command: command)
    monkeypatch.setattr(evalscope_runner.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        result = evalscope_runner.run_evalscope(["evalscope", "eval"], stream_output=True)
    assert out.getvalue() == result["stdout"] == "step one\nstep two\n"
    assert err.getvalue() == result["stderr"] == "warning line\n"
    assert result["returncode"] == 0


def test_run_evalscope_streaming_raises_with_child_failure_code(monkeypatch):
    class FakeProcess:
        stdout = io.StringIO("partial result\n")
        stderr = io.StringIO("CUDA out of memory\n")
        returncode = 9

        def wait(self):
            return self.returncode

    monkeypatch.setattr(evalscope_runner, "resolve_console_command", lambda command: command)
    monkeypatch.setattr(evalscope_runner.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    with pytest.raises(RuntimeError, match=r"EvalScope failed \(9\): CUDA out of memory"):
        evalscope_runner.run_evalscope(["evalscope", "eval"], stream_output=True)
