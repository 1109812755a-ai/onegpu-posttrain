from __future__ import annotations

import re
from typing import Any

_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_FINAL_RE = re.compile(r"(?:final\s+answer|answer)\s*[:=]\s*([^\n]+)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?")


def extract_last_boxed_content(text: str) -> str | None:
    """Extract the final balanced ``\\boxed{...}`` payload, including nested LaTeX."""
    text = text or ""
    needle = r"\boxed"
    start = 0
    found: list[str] = []
    while True:
        idx = text.find(needle, start)
        if idx < 0:
            break
        brace = idx + len(needle)
        while brace < len(text) and text[brace].isspace():
            brace += 1
        if brace >= len(text) or text[brace] != "{":
            start = idx + len(needle)
            continue
        depth = 0
        content_start = brace + 1
        end = None
        for pos in range(brace, len(text)):
            ch = text[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = pos
                    break
        if end is None:
            break
        payload = text[content_start:end].strip()
        if payload:
            found.append(payload)
        start = end + 1
    return found[-1] if found else None


def extract_answer(text: str) -> str | None:
    text = text or ""
    boxed = extract_last_boxed_content(text)
    if boxed is not None:
        return boxed
    tags = _ANSWER_TAG_RE.findall(text)
    if tags:
        return tags[-1].strip()
    finals = _FINAL_RE.findall(text)
    if finals:
        return finals[-1].strip().rstrip(". ")
    nums = _NUMBER_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def normalize_answer(value: Any) -> str:
    s = str(value).strip()
    s = s.replace("$", "").replace(",", "")
    s = re.sub(r"^\\boxed\s*\{(.*)\}$", r"\1", s)
    s = re.sub(r"\s+", "", s)
    return s.lower()


def answers_equal(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    # Local import avoids a module cycle: the bounded verifier uses normalization.
    from .verification import verify_answer_pairs

    try:
        return verify_answer_pairs([(pred, gold)])[0].status == "correct"
    except Exception:  # noqa: BLE001 - preserve the historical bool API on worker failures.
        # Preserve the historical bool API. Rich callers use typed outcomes.
        return False
