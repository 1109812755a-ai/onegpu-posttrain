from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .decontaminate import BenchmarkIndex
from .io import read_jsonl

QUESTION_FIELDS = ("question", "problem", "prompt", "text")


def _pick_question(row: dict[str, Any], field: str | None = None) -> str:
    if field:
        return str(row.get(field) or "")
    for candidate in QUESTION_FIELDS:
        if row.get(candidate):
            return str(row[candidate])
    return ""


def benchmark_index_from_jsonl(paths: Iterable[str | Path], *, shingle_n: int = 5) -> BenchmarkIndex:
    rows: list[tuple[str, str]] = []
    for path in paths:
        path = Path(path)
        for i, row in enumerate(read_jsonl(path), 1):
            text = _pick_question(row)
            if text:
                rows.append((f"{path.stem}:{i}", text))
    return BenchmarkIndex(rows, shingle_n=shingle_n)


def benchmark_index_from_hf(specs: list[dict[str, Any]], *, shingle_n: int = 5) -> BenchmarkIndex:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install onegpu-posttrain[data] to fetch benchmark datasets") from exc
    rows: list[tuple[str, str]] = []
    for spec in specs:
        ds = load_dataset(
            spec["repo"],
            spec.get("config"),
            split=spec.get("split", "test"),
            revision=spec.get("revision"),
        )
        field = spec.get("field")
        prefix = spec.get("id", spec["repo"])
        for i, row in enumerate(ds):
            text = _pick_question(dict(row), field)
            if text:
                rows.append((f"{prefix}:{i}", text))
    return BenchmarkIndex(rows, shingle_n=shingle_n)
