from __future__ import annotations

import os
import shutil
import sys
import sysconfig
from pathlib import Path


def child_process_env() -> dict[str, str]:
    """Return an environment that can resolve this Python's console scripts.

    Calling a virtual-environment executable by absolute path does not activate
    the environment, so its Scripts/bin directory may be absent from PATH.
    Commands such as ``swift`` and ``evalscope`` must still resolve to the
    entrypoints installed beside the running Python interpreter.
    """
    env = os.environ.copy()
    scripts_dir = str(Path(sys.executable).resolve().parent)
    current_path = env.get("PATH", "")
    env["PATH"] = scripts_dir if not current_path else scripts_dir + os.pathsep + current_path
    if os.name == "nt":
        env.setdefault("PYTHONUTF8", "1")
    return env


def console_command_available(name: str) -> bool:
    """Check a console script using the same PATH passed to child processes."""
    try:
        return shutil.which(name, path=child_process_env().get("PATH")) is not None
    except TypeError:
        # Preserve compatibility with lightweight monkeypatched/test resolvers
        # that implement the historical one-argument shutil.which signature.
        return shutil.which(name) is not None


def resolve_console_command(cmd: list[str]) -> list[str]:
    """Resolve a bare console-script name to this environment's executable."""
    if not cmd or Path(cmd[0]).parent != Path("."):
        return cmd
    resolved = shutil.which(cmd[0])
    if resolved:
        return [resolved, *cmd[1:]]
    suffix = ".exe" if os.name == "nt" else ""
    candidate = Path(sysconfig.get_path("scripts")) / f"{cmd[0]}{suffix}"
    if candidate.is_file():
        return [str(candidate), *cmd[1:]]
    return cmd
