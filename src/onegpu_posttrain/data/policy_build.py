"""Offline policy execution. Preview and full builds select candidates identically."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from ..reporting.data_report import length_summary, write_report
from ..schemas import Message, PreferenceRecord, SFTRecord
from .benchmarks import benchmark_index_from_jsonl
from .build import _write_evidence
from .evidence import step
from .io import read_jsonl, write_jsonl
from .manifest import (
    DatasetManifest,
    SourceSpec,
    TokenizerSpec,
    save_manifest,
    sha256_file,
)
from .normalize import normalize_problem, problem_id
from .openr1 import _boxed_answer, _is_complete
from .policy import DataPolicy, SourceDefinition, digest_json
from .source_cache import SourceIdentity, validate_source_cache
from .tokenization import HuggingFaceTokenizer, WhitespaceTokenizer


class InputUnavailable(ValueError):
    pass


def load_source(path: str | Path) -> SourceDefinition:
    path = Path(path).resolve(strict=True)
    source = SourceDefinition.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    for name in ("input_jsonl", "source_cache", "benchmark_jsonl"):
        value = getattr(source, name)
        if value:
            setattr(source, name, str((path.parent / value).resolve()))
    if source.tokenizer.local_path:
        source.tokenizer.local_path = str((path.parent / source.tokenizer.local_path).resolve())
    return source


def prepare_source(source: SourceDefinition) -> tuple[Path, Any, dict]:
    if source.license_status != "declared":
        raise InputUnavailable("来源许可/使用范围尚未声明；请先完善本地来源配置")
    if source.source_cache:
        manifest_path = Path(source.source_cache)
        if manifest_path.is_dir():
            manifest_path /= "manifest.json"
        if not manifest_path.is_file():
            raise InputUnavailable("本地源缓存不存在；不会下载或切回远程流")
        manifest = validate_source_cache(manifest_path, expected=SourceIdentity(
            repo=source.source_repo, config=source.source_config, revision=source.source_revision, split="train"))
        data_path = manifest_path.parent / manifest.data_file
        content_hash = manifest.content_sha256
    else:
        data_path = Path(source.input_jsonl)
        if not data_path.is_file():
            raise InputUnavailable("本地原始输入不存在；已筛选产物不能还原被删除候选")
        content_hash = sha256_file(data_path)
    if source.expected_sha256 and content_hash != source.expected_sha256:
        raise ValueError("本地来源内容与声明摘要不一致")
    benchmark = Path(source.benchmark_jsonl)
    if not benchmark.is_file():
        raise InputUnavailable("缺少本地评测污染索引；不会自动下载 benchmark")
    index = benchmark_index_from_jsonl([benchmark])
    if not index.shingles_by_id:
        raise InputUnavailable("本地评测污染索引为空，不能宣称已完成污染防护")
    pool = {"source_repo": source.source_repo, "source_revision": source.source_revision,
            "source_config": source.source_config, "content_sha256": content_hash,
            "benchmark_sha256": sha256_file(benchmark), "input_scope": source.input_scope,
            "existing_stage": source.existing_stage, "license_status": source.license_status,
            "license_note": source.license_note, "normalization": "v1", "decontamination": "v1"}
    pool["pool_id"] = "pool-" + digest_json(pool)
    return data_path, index, pool


def local_tokenizer(policy: DataPolicy):
    ref = policy.tokenizer
    if ref.test_only:
        return WhitespaceTokenizer()
    try:
        return HuggingFaceTokenizer(ref.local_path or ref.repo,
                                   None if ref.local_path else ref.revision, local_files_only=True)
    except (OSError, RuntimeError) as exc:
        raise InputUnavailable("缺少可用本地 tokenizer；请先准备指定版本，不会自动下载") from exc


def observe_candidates(row: dict, problem: str, tokenizer) -> list[dict]:
    flags = row.get("correctness_math_verify") or []
    candidates = []
    for index, value in enumerate(row.get("generations") or []):
        text = str(value or "")
        flag = flags[index] if index < len(flags) else None
        correctness = flag if type(flag) is bool else None
        candidates.append({"idx": index, "text": text, "correct": correctness,
                           "complete": _is_complete(row, index), "parsed_answer": _boxed_answer(text),
                           "tokens": tokenizer.count_messages([
                               {"role": "user", "content": problem}, {"role": "assistant", "content": text}])})
    return candidates


def _existing_row(row: dict, stage: str) -> dict:
    model = SFTRecord if stage == "sft" else PreferenceRecord
    record = model.model_validate(row)
    md = record.metadata
    if md.get("evaluation_only"):
        raise ValueError("评测记录不能作为训练来源")
    if stage == "sft":
        text = next(m.content for m in reversed(record.messages) if m.role == "assistant")
        generations, flags = [text], [md.get("correctness_math_verify")]
        complete = [md.get("is_reasoning_complete") is True]
    else:
        generations, flags = [record.chosen, record.rejected], [md.get("chosen_correct"), md.get("rejected_correct")]
        complete = [md.get("is_reasoning_complete") is True] * 2
    return {"problem": record.raw_problem, "answer": md.get("reference_answer") or md.get("parsed_answer"),
            "generations": generations, "correctness_math_verify": flags, "is_reasoning_complete": complete,
            "finish_reasons": ["stop"] * len(generations), "source": md.get("source"),
            "uuid": md.get("source_uuid"), "parent_record_id": record.id,
            **{key: md.get(key) for key in ("problem_type", "question_type", "difficulty")}}


def select_record(policy: DataPolicy, row: dict, pid: str, problem: str,
                  observations: list[dict], diagnostics: Counter, lengths: dict[str, list]) -> tuple[dict | None, str | None]:
    diagnostics["problems_observed"] += 1
    for candidate in observations:
        diagnostics["traces_observed"] += 1
        diagnostics["unknown_correctness_traces"] += int(candidate["correct"] is None)
        diagnostics["incomplete_traces"] += int(not candidate["complete"])
        diagnostics["unparseable_traces"] += int(candidate["parsed_answer"] is None)
        diagnostics["over_length_traces"] += int(candidate["tokens"] > policy.resources.max_length)
        label = "correct" if candidate["correct"] is True else "wrong" if candidate["correct"] is False else "unknown"
        lengths.setdefault(f"candidate_{label}_tokens", []).append(candidate["tokens"])
    candidates = [item for item in observations if item["text"].strip()]
    if policy.rules.completeness == "filter":
        candidates = [item for item in candidates if item["complete"]]
    if policy.rules.answer_parse == "filter":
        candidates = [item for item in candidates if item["parsed_answer"] is not None]
    if not candidates:
        return None, "no_candidate_after_completeness_parse"
    known_correct = [item for item in candidates if item["correct"] is True]
    known_wrong = [item for item in candidates if item["correct"] is False]
    if not known_correct:
        diagnostics["no_known_correct_candidate"] += 1
    if not known_wrong:
        diagnostics["no_known_wrong_candidate"] += 1
    within = lambda item: item["tokens"] <= policy.resources.max_length
    correct = [item for item in known_correct if within(item)]
    wrong = [item for item in known_wrong if within(item)]
    if known_correct and not correct:
        diagnostics["correct_candidates_all_over_length"] += 1
    if known_wrong and not wrong:
        diagnostics["wrong_candidates_all_over_length"] += 1
    if policy.stage == "sft":
        if policy.rules.correctness == "filter":
            if not known_correct:
                return None, "no_known_correct_candidate"
            if not correct:
                return None, "correct_candidates_all_over_length"
            eligible = correct
        else:
            eligible = [item for item in candidates if within(item)]
            if not eligible:
                return None, "all_candidates_over_length"
        chosen = min(eligible, key=lambda item: (item["tokens"], item["idx"]))
        selected_candidate_ids = [
            row["candidate_ids"][chosen["idx"]]
        ] if isinstance(row.get("candidate_ids"), list) else []
    else:
        for items, reason in ((known_correct, "no_known_correct_candidate"), (known_wrong, "no_known_wrong_candidate"),
                              (correct, "correct_candidates_all_over_length"), (wrong, "wrong_candidates_all_over_length")):
            if not items:
                return None, reason
        chosen = min(correct, key=lambda item: (item["tokens"], item["idx"]))
        pairs = []
        for item in wrong:
            ratio = item["tokens"] / max(chosen["tokens"], 1)
            if item["text"].strip() == chosen["text"].strip():
                continue
            if policy.rules.min_length_ratio is None or policy.rules.min_length_ratio <= ratio <= policy.rules.max_length_ratio:
                pairs.append((abs(1 - ratio), item["idx"], item))
        if not pairs:
            return None, "pair_length_ratio_or_identical_sides"
        rejected = min(pairs, key=lambda item: item[:2])[2]
        selected_candidate_ids = [
            row["candidate_ids"][index]
            for index in (chosen["idx"], rejected["idx"])
            if isinstance(row.get("candidate_ids"), list) and index < len(row["candidate_ids"])
        ]
    def ancestor(index: int) -> str:
        return row.get("parent_record_id") or f"openr1:{row.get('uuid') or pid}:generation:{index}"
    md = {"problem_id": pid, "source": row.get("source"), "source_uuid": row.get("uuid"),
          **{key: row.get(key) for key in ("problem_type", "question_type", "difficulty")},
          "usage": policy.usage, "policy_id": policy.policy_id, "policy_config_sha256": policy.config_sha256,
          "is_reasoning_complete": chosen["complete"], "parsed_answer": chosen["parsed_answer"],
          "reference_answer": row.get("answer"), "parent_record_ids": [ancestor(chosen["idx"])],
          "source_label_status": "known" if chosen["correct"] is not None else "unknown"}
    if selected_candidate_ids:
        md["synthetic_candidate_ids"] = selected_candidate_ids
    identity = digest_json({"pid": pid, "policy": policy.policy_id, "chosen": chosen["text"]})
    prompt = [Message(role="user", content=problem)]
    if policy.stage == "sft":
        md.update(correctness_math_verify=chosen["correct"], rendered_tokens=chosen["tokens"])
        messages = prompt + [Message(role="assistant", content=chosen["text"])]
        record = SFTRecord(id=f"sft_{identity}", raw_problem=problem, messages=messages,
                           prompt_template_id=policy.prompt_template_id, metadata=md)
    else:
        md.update(chosen_correct=True, rejected_correct=False, chosen_tokens=chosen["tokens"],
                  rejected_tokens=rejected["tokens"], length_ratio=rejected["tokens"] / max(chosen["tokens"], 1),
                  rejected_is_reasoning_complete=rejected["complete"])
        md["parent_record_ids"].append(ancestor(rejected["idx"]))
        identity = digest_json({"base": identity, "rejected": rejected["text"]})
        record = PreferenceRecord(id=f"pref_{identity}", raw_problem=problem, prompt=prompt,
                                  chosen=chosen["text"], rejected=rejected["text"],
                                  prompt_template_id=policy.prompt_template_id, metadata=md)
    return record.model_dump(mode="json"), None


def execute_policy(policy: DataPolicy, source: SourceDefinition, *, preview_rows: int | None = None,
                   seconds: float = 30, tokenizer=None) -> tuple[list[dict], dict]:
    if source.input_scope == "existing_dataset" and source.existing_stage != policy.stage:
        raise ValueError("已筛选数据的阶段与策略不一致")
    started = time.perf_counter()
    path, index, pool = prepare_source(source)
    preparation_seconds = time.perf_counter() - started
    if tokenizer is None and policy.tokenizer.local_path is None and source.tokenizer.local_path:
        policy = policy.model_copy(deep=True)
        policy.tokenizer.local_path = source.tokenizer.local_path
    tokenizer = tokenizer or local_tokenizer(policy)
    if isinstance(tokenizer, WhitespaceTokenizer) and not policy.tokenizer.test_only:
        raise ValueError("必须显式标注 test_only，不能将空格计数冒充真实 tokenizer")
    backend = getattr(tokenizer, "tokenizer", None)
    token_engine = getattr(backend, "backend_tokenizer", None)
    tokenizer_data = {"name": tokenizer.name} if backend is None else {
        "engine": token_engine.to_str() if token_engine is not None else backend.get_vocab(),
        "chat_template": getattr(backend, "chat_template", None),
        "special_tokens": backend.special_tokens_map}
    pool["tokenizer_sha256"] = digest_json(tokenizer_data)
    pool.pop("pool_id")
    pool["pool_id"] = "pool-" + digest_json(pool)
    counters, rejected, lengths = Counter(), Counter(), {}
    distributions_before = {key: Counter() for key in ("source", "problem_type", "question_type", "difficulty")}
    records, seen, assignments = [], set(), {}
    stop = "source_exhausted"
    start = time.perf_counter()
    limit = policy.resources.max_scan_rows
    if preview_rows is not None:
        if not 1 <= preview_rows <= 1000:
            raise ValueError("预览范围必须为 1..1000 raw rows")
        limit = min(limit, preview_rows) if limit else preview_rows
    for row in read_jsonl(path):
        if limit is not None and counters["raw_rows"] >= limit:
            stop = "max_raw_rows"
            break
        if preview_rows is not None and time.perf_counter() - start >= seconds:
            stop = "time_budget"
            break
        counters["raw_rows"] += 1
        if source.input_scope == "existing_dataset":
            row = _existing_row(row, policy.stage)
        problem = str(row.get("problem") or "").strip()
        if not normalize_problem(problem):
            counters["empty_problem"] += 1
            continue
        pid = problem_id(problem)
        if pid in seen:
            counters["duplicate_problem"] += 1
            continue
        seen.add(pid)
        counters["normalized_unique"] += 1
        if index.check(problem):
            counters["benchmark_contamination"] += 1
            continue
        assigned = ("sft" if policy.stage == "sft" else "alignment") if source.input_scope == "existing_dataset" else source.split.pool(pid)
        assignments[pid] = assigned
        if assigned != ("sft" if policy.stage == "sft" else "alignment"):
            counters["other_pool"] += 1
            continue
        counters["stage_attempted"] += 1
        for key, values in distributions_before.items():
            values[str(row.get(key) or "<unknown>")] += 1
        answer = str(row.get("answer") or "").strip().lower()
        if source.input_scope == "raw_source" and policy.rules.correctness == "filter" and (
                not answer or answer in {"proof", "notfound", "not found"} or "proof" in str(row.get("question_type") or "").lower()):
            rejected["reference_unverifiable"] += 1
            continue
        observation = observe_candidates(row, problem, tokenizer)
        record, reason = select_record(policy, row, pid, problem, observation, counters, lengths)
        if reason:
            rejected[reason] += 1
        else:
            counters["eligible"] += 1
            if len(records) < policy.resources.target_count:
                if policy.stage == "sft":
                    record["rendered_template_hash"] = digest_json(tokenizer.render_messages(record["messages"]))
                records.append(record)
        if preview_rows is None and len(records) >= policy.resources.target_count:
            stop = "targets_met"
            break
    unit = "examples" if policy.stage == "sft" else "preference_pairs"
    report = {"schema_version": "1.0", "protocol": "openr1-policy-build-v1", "policy": policy.artifact(),
              "source_pool": pool, "split_snapshot": {"snapshot_id": source.split.snapshot_id,
                                                         "config": source.split.model_dump(), "assignments": assignments},
              "counters": dict(counters), "first_rejection_reasons": dict(rejected),
              "diagnostics_semantics": "Additional trace/problem diagnostics may overlap; do not sum them as losses",
              "lengths": {name: length_summary(values) for name, values in lengths.items()},
              "coverage": {"before": {key: dict(values) for key, values in distributions_before.items()},
                           "after": {key: dict(Counter(str(r["metadata"].get(key) or "<unknown>") for r in records))
                                     for key in distributions_before}},
              "selected": {policy.stage: len(records)}, "targets": {policy.stage: policy.resources.target_count},
              "count_unit": unit, "stop_reason": stop, "preview": preview_rows is not None,
              "partial": preview_rows is not None or stop != "targets_met", "sampling_method": "deterministic_source_prefix",
              "input_preparation_seconds": preparation_seconds, "execution_seconds": time.perf_counter() - start,
              "warnings": ["来源标签不等于独立语义验证；小样本结果不能代表稳定模型效果",
                           "前缀预览不代表全源结果；不能恢复已筛选来源丢弃的候选"]}
    return records, report


def build_policy_artifacts(policy: DataPolicy, source: SourceDefinition, *, output_dir: str | Path,
                           tokenizer=None) -> dict:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("构建需要新的空输出目录；不覆盖历史产物")
    records, report = execute_policy(policy, source, tokenizer=tokenizer)
    stage = policy.stage
    canonical = output / "canonical" / f"{stage}.jsonl"
    write_jsonl(canonical, records)
    c = report["counters"]
    normalized = c.get("normalized_unique", 0)
    clean = normalized - c.get("benchmark_contamination", 0)
    attempted = c.get("stage_attempted", 0)
    eligible = c.get("eligible", 0)
    report["processing_steps"] = [
        step(1, "normalize-exact-dedup", "transform", "pool:raw", "pool:normalized", c.get("raw_rows", 0),
             normalized, "raw_rows", output_unit="unique_problems",
             rejected={key: c.get(key, 0) for key in ("empty_problem", "duplicate_problem")}),
        step(2, "decontaminate", "filter", "pool:normalized", "pool:clean", normalized, clean,
             "unique_problems", rejected={"benchmark_contamination": c.get("benchmark_contamination", 0)}),
        step(3, "problem-split", "split", "pool:clean", "pool:stage", clean, attempted, "unique_problems",
             unselected={"other_pool": c.get("other_pool", 0)}),
        step(4, "policy-eligibility", "filter", "pool:stage", "pool:eligible", attempted, eligible,
             "unique_problems", rejected=report["first_rejection_reasons"],
             details={"policy_id": policy.policy_id, "diagnostics": c, "coverage": report["coverage"]}),
        step(5, "policy-selection", "sample", "pool:eligible", f"dataset:{stage}", eligible, len(records),
             "unique_problems", output_unit=report["count_unit"], unselected={"target_count": eligible - len(records)}),
    ]
    source_spec = SourceSpec(repo=source.source_repo, config=source.source_config, revision=source.source_revision)
    tokenizer_spec = TokenizerSpec(repo=policy.tokenizer.repo, revision=policy.tokenizer.revision)
    config = {"policy": policy.artifact(), "pool_id": report["source_pool"]["pool_id"],
              "split_snapshot_id": source.split.snapshot_id}
    evidence = _write_evidence(
        output_dir=output, name="policy", report=report, config=config,
        datasets={stage: (canonical, records)}, source=source_spec, tokenizer=tokenizer_spec,
        tokenizer_name="whitespace-test-only" if policy.tokenizer.test_only else f"{policy.tokenizer.repo}@{policy.tokenizer.revision}",
        benchmark_index=benchmark_index_from_jsonl([source.benchmark_jsonl]),
        benchmark_metadata=[{"kind": "local_jsonl", "sha256": report["source_pool"]["benchmark_sha256"]}],
        overlap=0, source_acquisition={"mode": "verified_cache" if source.source_cache else "local_jsonl",
                                     **report["source_pool"]})
    manifest = DatasetManifest(
        **evidence[stage], stage=stage, source=source_spec, tokenizer=tokenizer_spec,
        prompt_template_id=policy.prompt_template_id, split_seed=source.split.seed, sample_count=len(records),
        protocol_versions={"policy_build": "v1", "split": source.split.protocol},
        hashes={"canonical_records_sha256": sha256_file(canonical)},
        metadata={"canonical_path": f"canonical/{stage}.jsonl",
                  "usage": policy.usage, "data_policy": policy.artifact(),
                  "source_pool_id": report["source_pool"]["pool_id"], "split_snapshot_id": source.split.snapshot_id,
                  "policy_integrity": "passed" if not report["quality_gate_failures"] else "failed",
                  "max_rendered_tokens": max((max(r["metadata"].get("rendered_tokens", 0),
                                                     r["metadata"].get("chosen_tokens", 0),
                                                     r["metadata"].get("rejected_tokens", 0)) for r in records), default=0),
                  "tokenizer_backend": "whitespace-test-only" if policy.tokenizer.test_only else "huggingface-local"})
    save_manifest(output / "manifests" / f"{stage}-policy-v1.yaml", manifest)
    write_report(output / "source_pool.json", report["source_pool"])
    write_report(output / "split_snapshot.json", report["split_snapshot"])
    write_report(output / "policy.json", policy.artifact())
    write_report(output / "reports" / "policy_report.json", report)
    return {"dataset_id": manifest.dataset_id, "manifest": str(output / "manifests" / f"{stage}-policy-v1.yaml"),
            "report": report, "trainable": bool(records) and not report["quality_gate_failures"],
            "official_mini_acceptance": "not_assessed"}
