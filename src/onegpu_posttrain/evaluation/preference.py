from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..data.io import read_jsonl, write_jsonl
from ..schemas import PreferenceRecord


@dataclass
class CandidateScore:
    mean_logprob: float
    sum_logprob: float
    token_count: int


@dataclass
class PreferenceEvalResult:
    pairs: int
    policy_pair_accuracy: float
    policy_pair_margin: float
    chosen_mean_logprob: float
    rejected_mean_logprob: float
    ties: int
    max_response_tokens_observed: int
    scoring_protocol: str = "response-mean-logprob-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate_pair_scores(rows: Iterable[dict[str, float]]) -> PreferenceEvalResult:
    rows = list(rows)
    if not rows:
        return PreferenceEvalResult(
            pairs=0,
            policy_pair_accuracy=0.0,
            policy_pair_margin=0.0,
            chosen_mean_logprob=0.0,
            rejected_mean_logprob=0.0,
            ties=0,
            max_response_tokens_observed=0,
        )
    margins = [float(r["chosen_mean_logprob"]) - float(r["rejected_mean_logprob"]) for r in rows]
    wins = sum(m > 0 for m in margins)
    ties = sum(math.isclose(m, 0.0, abs_tol=1e-12) for m in margins)
    return PreferenceEvalResult(
        pairs=len(rows),
        policy_pair_accuracy=wins / len(rows),
        policy_pair_margin=sum(margins) / len(margins),
        chosen_mean_logprob=sum(float(r["chosen_mean_logprob"]) for r in rows) / len(rows),
        rejected_mean_logprob=sum(float(r["rejected_mean_logprob"]) for r in rows) / len(rows),
        ties=ties,
        max_response_tokens_observed=max(int(r.get("response_tokens", 0)) for r in rows),
    )


def _torch_dtype(torch: Any, name: str, device: str):
    if device == "cpu":
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def _render_generation_prompt(tokenizer: Any, prompt: list[dict[str, str]]) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if template:
        return tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in prompt) + "\nASSISTANT:"


def score_candidate(
    *,
    model: Any,
    tokenizer: Any,
    prompt: list[dict[str, str]],
    candidate: str,
    device: str,
    max_length: int,
) -> CandidateScore:
    """Length-normalized conditional log-probability of response tokens only.

    The prompt is rendered with the model's chat template and generation marker,
    then the raw candidate text is appended. Template-closing tokens are excluded
    from the score so the public metric compares response content rather than
    chat-format terminators.
    """
    import torch

    prompt_text = _render_generation_prompt(tokenizer, prompt)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt_text + candidate, add_special_tokens=False)
    if len(full_ids) > max_length:
        raise ValueError(
            f"Preference eval record exceeds max_length={max_length}: {len(full_ids)} tokens. "
            "Refuse silent truncation."
        )
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("Candidate produced no scoreable response tokens")
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenizer boundary changed between rendered prompt and prompt+candidate")

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = input_ids[:, 1:]
        token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    # token_log_probs[:, j] scores input token j+1. Candidate begins at prompt_len.
    start = max(0, len(prompt_ids) - 1)
    selected = token_log_probs[0, start:]
    if selected.numel() == 0:
        raise ValueError("Candidate produced no scoreable response tokens")
    total = float(selected.sum().item())
    count = int(selected.numel())
    return CandidateScore(mean_logprob=total / count, sum_logprob=total, token_count=count)


def evaluate_preference_pairs(
    *,
    model_name_or_path: str | Path,
    input_path: str | Path,
    output_dir: str | Path,
    revision: str | None = None,
    dtype: str = "float16",
    max_length: int = 1024,
    limit: int | None = None,
) -> PreferenceEvalResult:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("Install onegpu-posttrain[all] to run preference evaluation") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_name_or_path), revision=revision, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_name_or_path),
        revision=revision,
        trust_remote_code=False,
        torch_dtype=_torch_dtype(torch, dtype, device),
    ).to(device)
    model.eval()

    predictions: list[dict[str, Any]] = []
    for idx, row in enumerate(read_jsonl(input_path)):
        if limit is not None and idx >= limit:
            break
        rec = PreferenceRecord.model_validate(row)
        prompt = [m.model_dump() for m in rec.prompt]
        chosen = score_candidate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            candidate=rec.chosen,
            device=device,
            max_length=max_length,
        )
        rejected = score_candidate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            candidate=rec.rejected,
            device=device,
            max_length=max_length,
        )
        margin = chosen.mean_logprob - rejected.mean_logprob
        predictions.append({
            "id": rec.id,
            "problem_id": rec.metadata.get("problem_id"),
            "chosen_mean_logprob": chosen.mean_logprob,
            "rejected_mean_logprob": rejected.mean_logprob,
            "margin": margin,
            "preferred": margin > 0,
            "chosen_tokens": chosen.token_count,
            "rejected_tokens": rejected.token_count,
            "response_tokens": max(chosen.token_count, rejected.token_count),
        })

    result = aggregate_pair_scores(predictions)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "preference_predictions.jsonl", predictions)
    payload = {
        **result.to_dict(),
        "model": str(model_name_or_path),
        "revision": revision,
        "input": str(input_path),
        "dtype": dtype,
        "device": device,
        "max_length_limit": max_length,
        "metric_definition": {
            "policy_pair_accuracy": "fraction of held-out pairs where mean response logprob(chosen) > mean response logprob(rejected)",
            "policy_pair_margin": "mean over pairs of chosen_mean_logprob - rejected_mean_logprob",
            "normalization": "response-token mean log probability",
            "silent_truncation": False,
        },
    }
    (out / "preference_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result
