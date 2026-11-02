from __future__ import annotations

import os
import sys
from pathlib import Path

from onegpu_posttrain.evaluation.answers import answers_equal, extract_answer
from onegpu_posttrain.evaluation.evalscope_runner import build_evalscope_command
from onegpu_posttrain.evaluation.internal_math import evaluate_internal_math
from onegpu_posttrain.subprocess_utils import child_process_env, resolve_console_command


class FakeGenerator:
    def generate(self, prompt: str) -> str:
        return "Reasoning... Therefore, \\boxed{42}."


def test_extract_answer():
    assert extract_answer("x then \\boxed{4}") == "4"
    assert extract_answer("Answer: 12") == "12"
    assert answers_equal("1,000", "1000")


def test_internal_eval(tmp_path: Path):
    result = evaluate_internal_math(
        "data/examples/eval.jsonl", FakeGenerator(), tmp_path
    )
    assert result.samples == 1
    assert result.accuracy == 1.0
    assert result.parse_rate == 1.0
    assert (tmp_path / "internal_math_predictions.jsonl").exists()


def test_evalscope_command():
    cmd = build_evalscope_command(
        model="Qwen/Qwen3-0.6B-Base",
        datasets=["gsm8k", "math_500"],
        output_dir="runs/base/evalscope",
        limit=5,
        extra_args=["--no-timestamp", "--seed", "42"],
    )
    assert cmd[:2] == ["evalscope", "eval"]
    assert "gsm8k" in cmd and "math_500" in cmd
    assert "--work-dir" in cmd


def test_child_process_path_starts_with_current_python_scripts():
    env = child_process_env()
    assert Path(env["PATH"].split(os.pathsep)[0]) == Path(sys.executable).resolve().parent


def test_resolve_console_command_from_scripts_dir(tmp_path: Path, monkeypatch):
    executable = tmp_path / ("example-tool.exe" if os.name == "nt" else "example-tool")
    executable.touch()
    monkeypatch.setattr("onegpu_posttrain.subprocess_utils.shutil.which", lambda _: None)
    monkeypatch.setattr("onegpu_posttrain.subprocess_utils.sysconfig.get_path", lambda _: str(tmp_path))
    resolved = resolve_console_command(["example-tool", "run"])
    assert resolved == [str(executable), "run"]
