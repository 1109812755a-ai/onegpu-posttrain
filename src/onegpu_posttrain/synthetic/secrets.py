"""Load a single provider credential from an explicitly requested local file."""

from __future__ import annotations

import os
import re
from pathlib import Path


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class LocalCredentialFileError(ValueError):
    """A sanitized local credential-file validation error."""


def load_local_provider_credential(env_name: str, path: str | Path) -> bool:
    """Fill a missing process environment variable from a strict local file.

    The file is intentionally not a shell script and may contain only the one
    credential named by the non-secret provider profile. Existing environment
    values take precedence. Returns whether a value was loaded from the file.
    """
    if not _ENV_NAME.fullmatch(env_name):
        raise LocalCredentialFileError("provider credential environment variable name is invalid")
    if os.environ.get(env_name):
        return False

    file_path = Path(path)
    if not file_path.exists():
        return False

    values: dict[str, str] = {}
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        raise LocalCredentialFileError("无法读取本地凭据文件 .env.local") from None

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise LocalCredentialFileError(f"本地凭据文件第 {line_number} 行格式无效")

        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = raw_value.strip()
        if not _ENV_NAME.fullmatch(key):
            raise LocalCredentialFileError(f"本地凭据文件第 {line_number} 行变量名无效")
        if key != env_name:
            raise LocalCredentialFileError(f"本地凭据文件第 {line_number} 行包含不允许的变量名: {key}")
        if key in values:
            raise LocalCredentialFileError(f"本地凭据文件第 {line_number} 行变量重复: {key}")

        if value.startswith(("'", '"')):
            quote = value[0]
            if len(value) < 2 or value[-1] != quote:
                raise LocalCredentialFileError(f"本地凭据文件第 {line_number} 行引号不匹配")
            value = value[1:-1]
        elif "\n" in value or "\r" in value:
            raise LocalCredentialFileError(f"本地凭据文件第 {line_number} 行格式无效")

        if "\x00" in value:
            raise LocalCredentialFileError(f"本地凭据文件第 {line_number} 行凭据格式无效")
        if not value:
            continue
        values[key] = value

    if env_name not in values:
        return False
    os.environ[env_name] = values[env_name]
    return True
