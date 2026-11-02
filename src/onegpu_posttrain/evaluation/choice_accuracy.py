"""Deterministic exact-match evaluator for registered objective choice datasets."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol

from ..data.io import read_jsonl, write_jsonl
from ..lineage import sha256_json
from ..schemas import EvalRecord

CHOICE_ACCURACY_PROTOCOL = "choice-accuracy-v1"
CHOICE_SCORING_VERSION = "exact-normalized-answer-v1"
_LABELS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def choice_comparability_key(dataset_sha256: str, sample_count: int) -> str:
    """Stable suite/protocol identity shared by preflight and persisted result."""
    return sha256_json({"protocol_version": CHOICE_ACCURACY_PROTOCOL,
                        "dataset_sha256": dataset_sha256, "sample_count": sample_count,
                        "scoring": CHOICE_SCORING_VERSION})


class ChoiceGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


def _prompt(record: EvalRecord) -> str:
    if record.question_type == "true_false":
        return ("请判断下列陈述是否正确。只输出 true 或 false，不要解释。\n\n"
                f"陈述：{record.prompt}\n答案：")
    choices = "\n".join(f"{chr(65 + i)}. {item}" for i, item in enumerate(record.options))
    answer_hint = "可选择多个字母，用逗号分隔" if record.question_type == "multiple_choice" else "只选择一个字母"
    return (f"请回答以下选择题。{answer_hint}，只输出选项字母，不要解释。\n\n"
            f"题目：{record.prompt}\n{choices}\n答案：")


def _parse_prediction(record: EvalRecord, output: str) -> str | None:
    text = output.strip()
    if record.question_type == "true_false":
        first = text.splitlines()[0].strip().lower() if text else ""
        match = re.match(r"^(true|false|✅|❌|对|错|正确|错误)(?:\b|[。.!！\s]|$)", first)
        if not match:
            return None
        return {"true": "true", "✅": "true", "对": "true", "正确": "true",
                "false": "false", "❌": "false", "错": "false", "错误": "false"}[match.group(1)]
    if not text:
        return None
    first_line = text.splitlines()[0].strip().upper()
    allowed = _LABELS.intersection(chr(65 + i) for i in range(len(record.options)))
    if record.question_type == "single_choice":
        match = re.match(r"^(?:答案\s*[:：]?\s*|ANSWER\s*[:：]?\s*)?([A-Z])(?:\b|[.、)）:：]|$)", first_line)
        if match and match.group(1) in allowed:
            return match.group(1)
        return None
    marker = re.search(r"(?:答案|ANSWER)\s*[:：]?\s*([A-Z](?:\s*[,，、和及]\s*[A-Z])*)", text, re.IGNORECASE)
    candidate = marker.group(1) if marker else first_line
    labels = re.findall(r"[A-Z]", candidate.upper())
    residue = re.sub(r"[A-Z\s,，、和及。.!！:：]", "", candidate.upper())
    if not labels or residue or any(label not in allowed for label in labels) or len(labels) != len(set(labels)):
        return None
    return ",".join(sorted(labels))


def evaluate_choice_records(records: list[dict[str, Any]], generator: ChoiceGenerator) -> dict[str, Any]:
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "denominator": 0, "parse_failures": 0})
    predictions: list[dict[str, Any]] = []
    total_correct = total = total_parse_failures = 0
    for payload in records:
        record = EvalRecord.model_validate(payload)
        if record.question_type is None:
            raise ValueError("choice-accuracy evaluator only accepts objective choice records")
        raw_output = generator.generate(_prompt(record))
        predicted = _parse_prediction(record, raw_output)
        gold = record.answer
        correct = predicted == gold
        stats = by_type[record.question_type]
        stats["denominator"] += 1
        stats["correct"] += int(correct)
        stats["parse_failures"] += int(predicted is None)
        total += 1
        total_correct += int(correct)
        total_parse_failures += int(predicted is None)
        predictions.append({"id": record.id, "question_type": record.question_type,
                            "gold": gold, "prediction": predicted, "raw_output": raw_output,
                            "parse_success": predicted is not None, "correct": correct})

    metrics: dict[str, dict[str, Any]] = {}
    for kind, stats in sorted(by_type.items()):
        metrics[kind] = {**stats, "accuracy": stats["correct"] / stats["denominator"] if stats["denominator"] else None,
                         "status": "measured" if stats["denominator"] else "not_measured"}
    metrics["overall"] = {"correct": total_correct, "denominator": total,
                          "parse_failures": total_parse_failures,
                          "accuracy": total_correct / total if total else None,
                          "status": "measured" if total else "not_measured"}
    return {"protocol": CHOICE_ACCURACY_PROTOCOL, "sample_count": total,
            "metrics": metrics, "predictions": predictions}


def evaluate_registered_choice_dataset(
    *, model_path: str | Path, canonical_path: str | Path, output_dir: str | Path,
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    """Load the selected local model once and evaluate a bounded registered dataset."""
    from .internal_math import TransformersGenerator

    generator = TransformersGenerator(str(model_path), max_new_tokens=max_new_tokens, temperature=0.0)
    records = list(read_jsonl(canonical_path))
    if not records:
        raise ValueError("registered evaluation dataset is empty")
    result = evaluate_choice_records(records, generator)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "choice_predictions.jsonl"
    write_jsonl(prediction_path, result.pop("predictions"))
    result["predictions_path"] = prediction_path.name
    (output / "choice_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
