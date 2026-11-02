from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .io import read_jsonl


def iter_local_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    yield from read_jsonl(path)


def iter_hf_dataset(
    repo: str,
    *,
    config: str | None = None,
    split: str = "train",
    revision: str | None = None,
    streaming: bool = True,
) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install onegpu-posttrain[data] to read Hugging Face datasets") from exc
    ds = load_dataset(repo, config, split=split, revision=revision, streaming=streaming)
    for row in ds:
        yield dict(row)
