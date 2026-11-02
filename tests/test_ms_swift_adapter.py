from __future__ import annotations

import json
from pathlib import Path

import pytest

from onegpu_posttrain.adapters.ms_swift import adapt_file


@pytest.mark.parametrize(
    ("stage", "filename"),
    [
        ("cpt", "cpt.jsonl"),
        ("sft", "sft.jsonl"),
        ("dpo", "preference.jsonl"),
        ("simpo", "preference.jsonl"),
        ("grpo", "grpo.jsonl"),
    ],
)
def test_adapt_examples(tmp_path: Path, stage: str, filename: str):
    src = Path("data/examples") / filename
    out = tmp_path / f"{stage}.jsonl"
    report = adapt_file(stage, src, out)
    assert report["records"] == 1
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["messages"]
    if stage in {"dpo", "simpo"}:
        assert row["messages"][-1]["role"] == "assistant"
        assert row["rejected_response"]
    if stage == "grpo":
        assert row["solution"] == "4"
        assert row["messages"][-1]["role"] == "user"
