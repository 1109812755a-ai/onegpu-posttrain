from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

from ..schemas import EvalRecord, GRPORecord, Message, PreferenceRecord, SFTRecord
from .decontaminate import BenchmarkIndex
from .evidence import ratio, step
from .normalize import normalize_problem, problem_id
from .tokenization import TokenizerBackend


def _stable_hash_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def _boxed_answer(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start >= 0:
        i = start + len(marker)
        depth = 1
        out: list[str] = []
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    ans = "".join(out).strip()
                    return ans or None
            out.append(ch)
            i += 1
    lower = text.lower()
    for marker2 in ("final answer:", "answer:"):
        idx = lower.rfind(marker2)
        if idx >= 0:
            trailing = text[idx + len(marker2) :].strip()
            if not trailing:
                continue
            value = trailing.splitlines()[0].strip()
            if value:
                return value
    return None


def _is_complete(row: dict[str, Any], idx: int) -> bool:
    complete = row.get("is_reasoning_complete") or []
    if idx >= len(complete) or not bool(complete[idx]):
        return False
    reasons = row.get("finish_reasons") or []
    if idx < len(reasons):
        reason = str(reasons[idx] or "").lower()
        if "length" in reason or "max" in reason and "token" in reason:
            return False
    return True


def _math_correctness(row: dict[str, Any], idx: int) -> bool | None:
    flags = row.get("correctness_math_verify") or []
    if idx >= len(flags):
        return None
    value = flags[idx]
    return bool(value) if value is not None else None


def _safe_ground_truth(row: dict[str, Any]) -> bool:
    answer = str(row.get("answer") or "").strip()
    question_type = str(row.get("question_type") or "").strip().lower()
    if not answer:
        return False
    if answer.lower() in {"proof", "notfound", "not found"}:
        return False
    if question_type == "proof" or "proof" in question_type:
        return False
    flags = row.get("correctness_math_verify") or []
    return any(v is True for v in flags)


@dataclass
class OpenR1BuildConfig:
    prompt_template_id: str = "qwen3-math-v1"
    eval_prompt_version: str = "math-v1"
    seed: int = 42
    sft_target: int = 200
    alignment_target: int = 300
    holdout_target: int = 100
    preference_target: int = 200
    preference_eval_target: int = 50
    grpo_target: int = 200
    sft_max_length: int = 1024
    preference_max_length: int = 1024
    grpo_max_prompt_length: int = 512
    preference_min_length_ratio: float = 0.5
    preference_max_length_ratio: float = 2.0
    fuzzy_threshold: float = 0.85
    max_raw_rows: int | None = 50_000


@dataclass
class OpenR1BuildResult:
    sft: list[dict[str, Any]] = dataclass_field(default_factory=list)
    preference: list[dict[str, Any]] = dataclass_field(default_factory=list)
    preference_eval: list[dict[str, Any]] = dataclass_field(default_factory=list)
    grpo: list[dict[str, Any]] = dataclass_field(default_factory=list)
    holdout: list[dict[str, Any]] = dataclass_field(default_factory=list)
    report: dict[str, Any] = dataclass_field(default_factory=dict)
    contaminated: list[dict[str, Any]] = dataclass_field(default_factory=list)


class OpenR1Builder:
    def __init__(self, tokenizer: TokenizerBackend, config: OpenR1BuildConfig, benchmark_index: BenchmarkIndex | None = None):
        self.tokenizer = tokenizer
        self.config = config
        self.benchmark_index = benchmark_index
        self.counters: Counter[str] = Counter()
        self.lengths: dict[str, list[float]] = defaultdict(list)
        self.distributions_before: dict[str, Counter[str]] = defaultdict(Counter)
        self.distributions_after: dict[str, Counter[str]] = defaultdict(Counter)
        self.contaminated: list[dict[str, Any]] = []
        self.seen_problem_ids: set[str] = set()

    def _pool(self, pid: str) -> str:
        # Weighted deterministic problem-level split based on requested eligible pool sizes.
        weights = {
            "sft": max(1, self.config.sft_target),
            "alignment": max(1, self.config.alignment_target),
            "holdout": max(1, self.config.holdout_target),
        }
        total = sum(weights.values())
        x = _stable_hash_int(f"{self.config.seed}:{pid}") % total
        if x < weights["sft"]:
            return "sft"
        if x < weights["sft"] + weights["alignment"]:
            return "alignment"
        return "holdout"

    def _base_metadata(self, row: dict[str, Any], pid: str) -> dict[str, Any]:
        source_uuid = row.get("uuid")
        return {
            "problem_id": pid,
            "source": row.get("source"),
            "source_uuid": source_uuid,
            "problem_type": row.get("problem_type"),
            "question_type": row.get("question_type"),
            "record_provenance": "record" if source_uuid else "dataset_only",
            "source_record_id": f"openr1:{source_uuid}" if source_uuid else None,
        }

    @staticmethod
    def _generation_id(row: dict[str, Any], index: int) -> str | None:
        source_uuid = row.get("uuid")
        return f"openr1:{source_uuid}:generation:{index}" if source_uuid else None

    def _trace_candidates(self, row: dict[str, Any], problem: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        generations: Sequence[str] = row.get("generations") or []
        prompt = [Message(role="user", content=problem).model_dump()]
        for idx, generation in enumerate(generations):
            self.counters["trace_total"] += 1
            generation = str(generation or "")
            if not generation.strip():
                self.counters["trace_empty"] += 1
                continue
            self.counters["trace_nonempty"] += 1
            correct = _math_correctness(row, idx)
            complete = _is_complete(row, idx)
            answer = _boxed_answer(generation)
            self.counters["trace_complete_observed"] += int(complete)
            self.counters["trace_parseable_observed"] += int(answer is not None)
            if not complete:
                self.counters["trace_incomplete"] += 1
                continue
            if answer is None:
                self.counters["trace_unparseable"] += 1
                continue
            messages = prompt + [Message(role="assistant", content=generation).model_dump()]
            rendered_tokens = self.tokenizer.count_messages(messages)
            self.lengths["trace_rendered_tokens_complete_parseable"].append(rendered_tokens)
            self.counters["trace_eligible"] += 1
            out.append({
                "idx": idx,
                "text": generation,
                "correct": correct,
                "parsed_answer": answer,
                "tokens": rendered_tokens,
            })
        return out

    def _make_sft(self, row: dict[str, Any], pid: str, problem: str, traces: list[dict[str, Any]]) -> dict[str, Any] | None:
        self.counters["sft_attempted"] += 1
        correct_candidates = [t for t in traces if t["correct"] is True]
        self.lengths["sft_candidate_correct_tokens"].extend(t["tokens"] for t in correct_candidates)
        eligible = [t for t in correct_candidates if t["tokens"] <= self.config.sft_max_length]
        if not eligible:
            self.counters["sft_no_eligible_correct_trace"] += 1
            return None
        chosen = min(eligible, key=lambda t: (t["tokens"], t["idx"]))
        messages = [Message(role="user", content=problem), Message(role="assistant", content=chosen["text"])]
        rendered = self.tokenizer.render_messages([m.model_dump() for m in messages])
        metadata = self._base_metadata(row, pid)
        metadata.update({
            "generation_index": chosen["idx"],
            "parent_record_ids": [self._generation_id(row, chosen["idx"])] if row.get("uuid") else [],
            "correctness_math_verify": True,
            "is_reasoning_complete": True,
            "rendered_tokens": chosen["tokens"],
            "parsed_answer": chosen["parsed_answer"],
        })
        record = SFTRecord(
            id=f"sft_{pid[:16]}",
            raw_problem=problem,
            messages=messages,
            prompt_template_id=self.config.prompt_template_id,
            rendered_template_hash=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            metadata=metadata,
        )
        self.lengths["sft_rendered_tokens"].append(chosen["tokens"])
        return record.model_dump(mode="json")

    def _make_preference(
        self,
        row: dict[str, Any],
        pid: str,
        problem: str,
        traces: list[dict[str, Any]],
        *,
        namespace: str = "preference",
        id_prefix: str = "pref",
        evaluation_only: bool = False,
    ) -> dict[str, Any] | None:
        self.counters[f"{namespace}_attempted"] += 1
        correct_all = [t for t in traces if t["correct"] is True]
        wrong_all = [t for t in traces if t["correct"] is False]
        self.lengths[f"{namespace}_candidate_correct_tokens"].extend(t["tokens"] for t in correct_all)
        self.lengths[f"{namespace}_candidate_wrong_tokens"].extend(t["tokens"] for t in wrong_all)
        correct = [t for t in correct_all if t["tokens"] <= self.config.preference_max_length]
        wrong = [t for t in wrong_all if t["tokens"] <= self.config.preference_max_length]
        if not correct or not wrong:
            self.counters[f"{namespace}_missing_valid_pair_side"] += 1
            return None
        # Chosen follows the SFT rule: shortest complete correct trace.
        chosen = min(correct, key=lambda t: (t["tokens"], t["idx"]))
        pair_candidates: list[tuple[float, int, dict[str, Any]]] = []
        for rejected in wrong:
            if chosen["tokens"] <= 0:
                continue
            ratio = rejected["tokens"] / chosen["tokens"]
            if self.config.preference_min_length_ratio <= ratio <= self.config.preference_max_length_ratio:
                pair_candidates.append((abs(math.log(max(ratio, 1e-9))), rejected["idx"], rejected))
        if not pair_candidates:
            self.counters[f"{namespace}_length_ratio_gate"] += 1
            return None
        _, _, rejected = min(pair_candidates)
        ratio = rejected["tokens"] / chosen["tokens"]
        prompt = [Message(role="user", content=problem)]
        metadata = self._base_metadata(row, pid)
        metadata.update({
            "chosen_generation_index": chosen["idx"],
            "rejected_generation_index": rejected["idx"],
            "parent_record_ids": [self._generation_id(row, chosen["idx"]), self._generation_id(row, rejected["idx"])]
            if row.get("uuid") else [],
            "chosen_correct": True,
            "rejected_correct": False,
            "chosen_tokens": chosen["tokens"],
            "rejected_tokens": rejected["tokens"],
            "length_ratio": ratio,
            "chosen_parsed_answer": chosen["parsed_answer"],
            "rejected_parsed_answer": rejected["parsed_answer"],
            "evaluation_only": evaluation_only,
        })
        record = PreferenceRecord(
            id=f"{id_prefix}_{pid[:16]}",
            raw_problem=problem,
            prompt=prompt,
            chosen=chosen["text"],
            rejected=rejected["text"],
            prompt_template_id=self.config.prompt_template_id,
            metadata=metadata,
        )
        self.lengths[f"{namespace}_chosen_tokens"].append(chosen["tokens"])
        self.lengths[f"{namespace}_rejected_tokens"].append(rejected["tokens"])
        self.lengths[f"{namespace}_length_ratio"].append(ratio)
        return record.model_dump(mode="json")

    def _make_grpo(self, row: dict[str, Any], pid: str, problem: str) -> dict[str, Any] | None:
        self.counters["grpo_attempted"] += 1
        if not _safe_ground_truth(row):
            self.counters["grpo_unverifiable_ground_truth"] += 1
            return None
        prompt = [Message(role="user", content=problem)]
        prompt_tokens = self.tokenizer.count_messages([m.model_dump() for m in prompt], add_generation_prompt=True)
        self.lengths["grpo_candidate_prompt_tokens"].append(prompt_tokens)
        if prompt_tokens > self.config.grpo_max_prompt_length:
            self.counters["grpo_prompt_length_gate"] += 1
            return None
        metadata = self._base_metadata(row, pid)
        metadata.update({"prompt_tokens": prompt_tokens, "verifier": "math_verify"})
        record = GRPORecord(
            id=f"grpo_{pid[:16]}",
            raw_problem=problem,
            prompt=prompt,
            answer=str(row.get("answer") or "").strip(),
            prompt_template_id=self.config.prompt_template_id,
            metadata=metadata,
        )
        self.lengths["grpo_prompt_tokens"].append(prompt_tokens)
        return record.model_dump(mode="json")

    def _make_eval(self, row: dict[str, Any], pid: str, problem: str) -> dict[str, Any] | None:
        self.counters["holdout_attempted"] += 1
        if not _safe_ground_truth(row):
            self.counters["holdout_unverifiable_ground_truth"] += 1
            return None
        metadata = self._base_metadata(row, pid)
        record = EvalRecord(
            id=f"eval_{pid[:16]}",
            problem_id=pid,
            raw_problem=problem,
            prompt=problem,
            answer=str(row.get("answer") or "").strip(),
            eval_prompt_version=self.config.eval_prompt_version,
            metadata=metadata,
        )
        return record.model_dump(mode="json")

    def _done(self, out: OpenR1BuildResult) -> bool:
        return (
            len(out.sft) >= self.config.sft_target
            and len(out.preference) >= self.config.preference_target
            and len(out.preference_eval) >= self.config.preference_eval_target
            and len(out.grpo) >= self.config.grpo_target
            and len(out.holdout) >= self.config.holdout_target
            and self.counters.get("alignment_core_seen", 0) >= self.config.alignment_target
        )

    def build(self, rows: Iterable[dict[str, Any]]) -> OpenR1BuildResult:
        out = OpenR1BuildResult()
        alignment_seen = 0
        stop_reason = "source_exhausted"
        for raw_index, row in enumerate(rows):
            if self.config.max_raw_rows is not None and raw_index >= self.config.max_raw_rows:
                self.counters["max_raw_rows_reached"] += 1
                stop_reason = "max_raw_rows"
                break
            self.counters["raw_rows"] += 1
            problem = normalize_problem(str(row.get("problem") or ""))
            if not problem:
                self.counters["empty_problem"] += 1
                continue
            pid = problem_id(problem)
            if pid in self.seen_problem_ids:
                self.counters["exact_duplicate_problem"] += 1
                continue
            self.seen_problem_ids.add(pid)
            self.counters["normalized_unique"] += 1
            for field in ("source", "problem_type", "question_type"):
                self.distributions_before[field][str(row.get(field) or "<missing>")] += 1

            # Benchmark decontamination occurs before verifiable core and split.
            if self.benchmark_index is not None:
                hits = self.benchmark_index.check(problem, fuzzy_threshold=self.config.fuzzy_threshold)
                if hits:
                    self.counters["benchmark_contamination_removed"] += 1
                    self.contaminated.append({"problem_id": pid, "problem": problem, "hits": [h.to_dict() for h in hits]})
                    continue

            traces = self._trace_candidates(row, problem)
            if not any(t["correct"] is True for t in traces):
                self.counters["no_verifier_correct_trace"] += 1
                continue
            self.counters["verifiable_core"] += 1
            pool = self._pool(pid)
            self.counters[f"pool_{pool}"] += 1

            if pool == "sft" and len(out.sft) < self.config.sft_target:
                record = self._make_sft(row, pid, problem, traces)
                if record is not None:
                    out.sft.append(record)
                    self.counters["sft_selected"] += 1
                    for field in ("source", "problem_type", "question_type"):
                        self.distributions_after[f"sft.{field}"][str(row.get(field) or "<missing>")] += 1
            elif pool == "alignment":
                alignment_seen += 1
                self.counters["alignment_core_seen"] += 1
                if len(out.preference) < self.config.preference_target:
                    pref = self._make_preference(row, pid, problem, traces)
                    if pref is not None:
                        out.preference.append(pref)
                        self.counters["preference_selected"] += 1
                if len(out.grpo) < self.config.grpo_target:
                    grpo = self._make_grpo(row, pid, problem)
                    if grpo is not None:
                        out.grpo.append(grpo)
                        self.counters["grpo_selected"] += 1
                if (len(out.preference) <= self.config.preference_target or len(out.grpo) <= self.config.grpo_target):
                    for field in ("source", "problem_type", "question_type"):
                        self.distributions_after[f"alignment.{field}"][str(row.get(field) or "<missing>")] += 1
            elif pool == "holdout" and (
                len(out.holdout) < self.config.holdout_target
                or len(out.preference_eval) < self.config.preference_eval_target
            ):
                selected_any = False
                if len(out.holdout) < self.config.holdout_target:
                    ev = self._make_eval(row, pid, problem)
                    if ev is not None:
                        out.holdout.append(ev)
                        self.counters["holdout_selected"] += 1
                        selected_any = True
                if len(out.preference_eval) < self.config.preference_eval_target:
                    pref_eval = self._make_preference(
                        row,
                        pid,
                        problem,
                        traces,
                        namespace="preference_eval",
                        id_prefix="prefeval",
                        evaluation_only=True,
                    )
                    if pref_eval is not None:
                        out.preference_eval.append(pref_eval)
                        self.counters["preference_eval_selected"] += 1
                        selected_any = True
                if selected_any:
                    for field in ("source", "problem_type", "question_type"):
                        self.distributions_after[f"holdout.{field}"][str(row.get(field) or "<missing>")] += 1

            if self._done(out):
                stop_reason = "targets_met"
                break

        out.contaminated = self.contaminated
        selected_pids = {
            *(r["metadata"]["problem_id"] for r in out.sft),
            *(r["metadata"]["problem_id"] for r in out.preference),
            *(r["metadata"]["problem_id"] for r in out.preference_eval),
            *(r["metadata"]["problem_id"] for r in out.grpo),
            *(r["problem_id"] for r in out.holdout),
        }
        overlap = {
            "sft_preference": len({r["metadata"]["problem_id"] for r in out.sft} & {r["metadata"]["problem_id"] for r in out.preference}),
            "sft_grpo": len({r["metadata"]["problem_id"] for r in out.sft} & {r["metadata"]["problem_id"] for r in out.grpo}),
            "train_holdout": len(
                ({r["metadata"]["problem_id"] for r in out.sft}
                 | {r["metadata"]["problem_id"] for r in out.preference}
                 | {r["metadata"]["problem_id"] for r in out.grpo})
                & (
                    {r["problem_id"] for r in out.holdout}
                    | {r["metadata"]["problem_id"] for r in out.preference_eval}
                )
            ),
            "preference_grpo": len({r["metadata"]["problem_id"] for r in out.preference} & {r["metadata"]["problem_id"] for r in out.grpo}),
            "preference_train_eval": len(
                {r["metadata"]["problem_id"] for r in out.preference}
                & {r["metadata"]["problem_id"] for r in out.preference_eval}
            ),
            "preference_eval_internal_holdout": len(
                {r["metadata"]["problem_id"] for r in out.preference_eval}
                & {r["problem_id"] for r in out.holdout}
            ),
        }
        out.report = {
            "protocol": "openr1-build-v1",
            "stop_reason": stop_reason,
            "processing_steps": self._processing_steps(),
            "tokenizer": self.tokenizer.name,
            "targets": {
                "sft": self.config.sft_target,
                "alignment_core": self.config.alignment_target,
                "preference": self.config.preference_target,
                "preference_eval": self.config.preference_eval_target,
                "grpo": self.config.grpo_target,
                "holdout": self.config.holdout_target,
            },
            "selected": {
                "sft": len(out.sft),
                "preference": len(out.preference),
                "preference_eval": len(out.preference_eval),
                "grpo": len(out.grpo),
                "holdout": len(out.holdout),
                "unique_problem_ids": len(selected_pids),
                "alignment_rows_seen": alignment_seen,
            },
            "counters": dict(sorted(self.counters.items())),
            "lengths": {k: _summary(v) for k, v in self.lengths.items()},
            "distribution_before": {k: dict(v) for k, v in self.distributions_before.items()},
            "distribution_after": {k: dict(v) for k, v in self.distributions_after.items()},
            "overlap": overlap,
        }
        return out

    def _processing_steps(self) -> list[dict[str, Any]]:
        c = self.counters
        raw = c["raw_rows"]
        normalized = raw - c["empty_problem"]
        unique = c["normalized_unique"]
        screened = unique - c["benchmark_contamination_removed"]
        core = c["verifiable_core"]
        trace_out = c["trace_eligible"]
        steps = [
            step(1, "normalize", "transform", "source:processed_rows", "pool:normalized", raw, normalized,
                 "raw_rows", rejected={"empty_problem": c["empty_problem"]}),
            step(2, "exact-dedup", "filter", "pool:normalized", "pool:unique", normalized, unique,
                 "raw_rows", output_unit="unique_problems", rejected={"exact_duplicate_problem": c["exact_duplicate_problem"]}),
            step(3, "decontaminate", "filter", "pool:unique", "pool:screened", unique, screened,
                 "unique_problems", rejected={"benchmark_contamination": c["benchmark_contamination_removed"]},
                 skipped=self.benchmark_index is None or not self.benchmark_index.normalized_to_ids,
                 limitations=["No nonempty benchmark index available; pass-through is not evidence of no contamination"]
                 if self.benchmark_index is None or not self.benchmark_index.normalized_to_ids else [],
                 details={"fuzzy_threshold": self.config.fuzzy_threshold, "metric": "jaccard"}),
            step(4, "trace-eligibility", "filter", "pool:screened/generations", "pool:eligible_traces",
                 c["trace_total"], trace_out, "generations",
                 rejected={key: c[key] for key in ("trace_empty", "trace_incomplete", "trace_unparseable")},
                 before={"complete_rate": ratio(c["trace_complete_observed"], c["trace_nonempty"], "upstream_complete_flag_and_finish_reason:v1"),
                         "answer_parse_rate": ratio(c["trace_parseable_observed"], c["trace_nonempty"], "boxed_or_final_answer_parser:v1")},
                 after={"complete_rate": ratio(trace_out, trace_out, "upstream_complete_flag_and_finish_reason:v1"),
                        "answer_parse_rate": ratio(trace_out, trace_out, "boxed_or_final_answer_parser:v1")},
                 limitations=["Before metrics cover nonempty generations, not raw problems",
                              "After rates are conditioned on passing this filter; correctness is not reverified"]),
            step(5, "verifiable-core", "filter", "pool:screened", "pool:core", screened, core,
                 "unique_problems", rejected={"no_verifier_correct_trace": c["no_verifier_correct_trace"]},
                 limitations=["Uses upstream correctness_math_verify flags on complete, parseable traces; no fresh verifier execution"]),
            step(6, "problem-split", "split", "pool:core", "pool:assigned", core, core, "unique_problems",
                 details={"seed": self.config.seed, "pool_counts": {p: c[f"pool_{p}"] for p in ("sft", "alignment", "holdout")}}),
        ]
        stages = [
            ("sft", "sft", "examples", ["sft_no_eligible_correct_trace"]),
            ("preference", "alignment", "preference_pairs", ["preference_missing_valid_pair_side", "preference_length_ratio_gate"]),
            ("grpo", "alignment", "prompts", ["grpo_unverifiable_ground_truth", "grpo_prompt_length_gate"]),
            ("holdout", "holdout", "examples", ["holdout_unverifiable_ground_truth"]),
            ("preference_eval", "holdout", "preference_pairs", ["preference_eval_missing_valid_pair_side", "preference_eval_length_ratio_gate"]),
        ]
        for name, pool, unit, reasons in stages:
            attempts = c[f"{name}_attempted"]
            steps.append(step(
                len(steps) + 1, f"build-{name}", "transform", f"pool:{pool}/attempted", f"canonical:{name}",
                attempts, c[f"{name}_selected"], "unique_problems", output_unit=unit,
                rejected={key: c[key] for key in reasons},
                details={"assigned_pool_problems": c[f"pool_{pool}"], "not_attempted_after_quota": c[f"pool_{pool}"] - attempts,
                         "lengths": {key: _summary(values) for key, values in self.lengths.items() if key.startswith(f"{name}_")}},
                limitations=["Branch summaries describe actual builder attempts; quota-full problems are not quality rejections",
                             "Branch order in this aggregate DAG is not a serial pipeline through all training stages"],
            ))
        return steps


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p90": None, "p95": None, "p99": None}
    xs = sorted(values)
    def percentile(q: float) -> float:
        pos = (len(xs) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(xs) - 1)
        frac = pos - lo
        return xs[lo] * (1 - frac) + xs[hi] * frac
    return {
        "count": len(xs),
        "min": xs[0],
        "max": xs[-1],
        "mean": sum(xs) / len(xs),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }
