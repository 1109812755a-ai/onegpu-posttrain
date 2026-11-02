from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..data.io import read_jsonl
from ..schemas import CPTRecord


@dataclass
class PerplexityResult:
    samples: int
    predicted_tokens: int
    total_nll: float
    mean_nll: float | None
    perplexity: float | None
    truncated_samples: int
    wall_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def aggregate_nll(items: Iterable[tuple[float, int]]) -> tuple[float, int, float | None, float | None]:
    total_nll = 0.0
    total_tokens = 0
    for nll, tokens in items:
        if tokens <= 0:
            continue
        total_nll += float(nll)
        total_tokens += int(tokens)
    if total_tokens == 0:
        return total_nll, total_tokens, None, None
    mean_nll = total_nll / total_tokens
    # Prevent an accidental overflow from turning a valid evaluation artifact
    # into an exception; huge PPL is still informative.
    ppl = math.exp(mean_nll) if mean_nll < 700 else float("inf")
    return total_nll, total_tokens, mean_nll, ppl


def evaluate_cpt_perplexity(
    *,
    model: str,
    input_path: str | Path,
    output_path: str | Path,
    revision: str | None = None,
    max_length: int = 1024,
    limit: int | None = None,
    dtype: str = "float16",
    device_map: str = "auto",
) -> PerplexityResult:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install torch and transformers to evaluate perplexity") from exc

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
    kwargs: dict[str, Any] = {"revision": revision, "trust_remote_code": False, "device_map": device_map}
    if dtype == "auto":
        kwargs["torch_dtype"] = "auto"
    else:
        kwargs["torch_dtype"] = getattr(torch, dtype)
    lm = AutoModelForCausalLM.from_pretrained(model, **kwargs)
    lm.eval()
    model_device = next(lm.parameters()).device

    weighted: list[tuple[float, int]] = []
    truncated = samples = 0
    started = time.perf_counter()
    with torch.no_grad():
        for idx, row in enumerate(read_jsonl(input_path)):
            if limit is not None and idx >= limit:
                break
            rec = CPTRecord.model_validate(row)
            full = tokenizer(rec.text, add_special_tokens=True, return_tensors="pt")
            original_len = int(full["input_ids"].shape[1])
            enc = tokenizer(rec.text, add_special_tokens=True, truncation=True, max_length=max_length, return_tensors="pt")
            seq_len = int(enc["input_ids"].shape[1])
            if original_len > seq_len:
                truncated += 1
            if seq_len < 2:
                continue
            enc = {k: v.to(model_device) for k, v in enc.items()}
            out = lm(**enc, labels=enc["input_ids"])
            predicted = seq_len - 1
            weighted.append((float(out.loss.item()) * predicted, predicted))
            samples += 1

    total_nll, tokens, mean_nll, ppl = aggregate_nll(weighted)
    result = PerplexityResult(
        samples=samples,
        predicted_tokens=tokens,
        total_nll=total_nll,
        mean_nll=mean_nll,
        perplexity=ppl,
        truncated_samples=truncated,
        wall_seconds=time.perf_counter() - started,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
