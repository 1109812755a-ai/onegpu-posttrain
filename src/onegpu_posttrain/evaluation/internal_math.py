from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..data.io import read_jsonl, write_jsonl
from ..schemas import EvalRecord
from .answers import answers_equal, extract_answer


class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass
class InternalEvalResult:
    samples: int
    correct: int
    accuracy: float
    parse_success: int
    parse_rate: float
    empty_outputs: int
    wall_seconds: float
    predictions_path: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def evaluate_internal_math(
    input_path: str | Path,
    generator: Generator,
    output_dir: str | Path,
    limit: int | None = None,
) -> InternalEvalResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = []
    correct = parse_success = empty_outputs = 0
    start = time.perf_counter()
    for idx, row in enumerate(read_jsonl(input_path)):
        if limit is not None and idx >= limit:
            break
        rec = EvalRecord.model_validate(row)
        raw_output = generator.generate(rec.prompt)
        if not raw_output.strip():
            empty_outputs += 1
        extracted = extract_answer(raw_output)
        if extracted is not None:
            parse_success += 1
        is_correct = answers_equal(extracted, rec.answer)
        correct += int(is_correct)
        predictions.append({
            "id": rec.id,
            "problem_id": rec.problem_id,
            "prompt": rec.prompt,
            "gold": rec.answer,
            "raw_output": raw_output,
            "extracted_answer": extracted,
            "parse_success": extracted is not None,
            "correct": is_correct,
        })
    elapsed = time.perf_counter() - start
    n = len(predictions)
    pred_path = output_dir / "internal_math_predictions.jsonl"
    write_jsonl(pred_path, predictions)
    result = InternalEvalResult(
        samples=n,
        correct=correct,
        accuracy=(correct / n if n else 0.0),
        parse_success=parse_success,
        parse_rate=(parse_success / n if n else 0.0),
        empty_outputs=empty_outputs,
        wall_seconds=elapsed,
        predictions_path=str(pred_path),
    )
    (output_dir / "internal_math_metrics.json").write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


class TransformersGenerator:
    def __init__(
        self,
        model: str,
        *,
        revision: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        dtype: str = "auto",
        device_map: str = "auto",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install onegpu-posttrain[data] plus torch to run local evaluation") from exc
        kwargs: dict[str, Any] = {"revision": revision, "trust_remote_code": False}
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
        if dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, dtype)
        else:
            kwargs["torch_dtype"] = "auto"
        kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        do_sample = self.temperature > 0
        generation = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        continuation = generation[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(continuation, skip_special_tokens=True)
