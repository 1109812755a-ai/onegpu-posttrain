from __future__ import annotations

import hashlib
import re
import unicodedata

_WS = re.compile(r"\s+")


def normalize_problem(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _WS.sub(" ", text.strip())
    return text


def problem_id(text: str) -> str:
    normalized = normalize_problem(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
