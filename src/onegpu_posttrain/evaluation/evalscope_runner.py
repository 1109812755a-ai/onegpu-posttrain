from __future__ import annotations

import json
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from ..subprocess_utils import child_process_env, resolve_console_command


def build_evalscope_command(
    *,
    model: str,
    datasets: list[str],
    output_dir: str | Path,
    limit: int | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    cmd = ["evalscope", "eval", "--model", model, "--datasets", *datasets]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    # EvalScope owns its internal result layout; put all official benchmark
    # artifacts under the run-specific directory.
    cmd += ["--work-dir", str(output_dir)]
    if extra_args:
        cmd += list(extra_args)
    return cmd


def command_string(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def run_evalscope(cmd: list[str], *, dry_run: bool = False, stream_output: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"command": cmd, "command_string": command_string(cmd), "dry_run": dry_run}
    if dry_run:
        return result
    if stream_output:
        proc = subprocess.Popen(
            resolve_console_command(cmd), text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, env=child_process_env(),
        )
        captured: dict[str, list[str]] = {"stdout": [], "stderr": []}

        def forward(name: str, stream, destination) -> None:
            assert stream is not None
            for line in iter(stream.readline, ""):
                captured[name].append(line)
                destination.write(line)
                destination.flush()

        threads = [
            threading.Thread(target=forward, args=("stdout", proc.stdout, sys.stdout), daemon=True),
            threading.Thread(target=forward, args=("stderr", proc.stderr, sys.stderr), daemon=True),
        ]
        for thread in threads:
            thread.start()
        returncode = proc.wait()
        for thread in threads:
            thread.join()
        stdout, stderr = "".join(captured["stdout"]), "".join(captured["stderr"])
    else:
        proc = subprocess.run(
            resolve_console_command(cmd), check=False, text=True, encoding="utf-8", errors="replace",
            capture_output=True, env=child_process_env(),
        )
        returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    result.update({"returncode": returncode, "stdout": stdout, "stderr": stderr})
    if returncode != 0:
        raise RuntimeError(f"EvalScope failed ({returncode}): {stderr[-2000:]}")
    return result


def write_plan(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
