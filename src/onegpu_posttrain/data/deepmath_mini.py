"""Offline, deterministic DeepMath-103K source builder for the Mini SFT cohort."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..adapters.ms_swift import adapt_sft
from ..evaluation.answers import extract_answer, normalize_answer
from ..lineage import sha256_file
from ..reporting.data_report import write_markdown_report, write_report
from ..schemas import Message, SFTRecord
from .build import _write_evidence
from .decontaminate import BenchmarkIndex
from .evidence import ReportReference, step
from .io import write_jsonl
from .manifest import DatasetManifest, SourceSpec, TokenizerSpec, save_manifest
from .manifest import sha256_file as file_sha
from .normalize import normalize_problem, problem_id
from .policy import DataPolicy, PolicyRules, PolicyTokenizer, ResourceLimits
from .tokenization import HuggingFaceTokenizer, TokenizerBackend

PINNED_DEEPMATH_REVISION = "5cf055d1fe3d7a2eb19719ac020211469736ae44"
PINNED_TOKENIZER_REPO = "Qwen/Qwen3-0.6B-Base"
PINNED_TOKENIZER_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
BUILD_PROTOCOL = "deepmath-mini-sft-v1"


class DeepMathMiniBuildError(ValueError):
    """Raised when a pinned local source cannot be safely built."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_inventory(source: str | Path) -> tuple[Path, list[dict[str, Any]], str, str, dict[str, str]]:
    root = Path(source).expanduser().resolve()
    if not root.is_dir():
        raise DeepMathMiniBuildError(f"本地 DeepMath snapshot 不存在：{root}；不会联网下载。")
    match = re.search(r"snapshots[\\/]([^\\/]+)$", str(root))
    revision = match.group(1) if match else None
    if revision != PINNED_DEEPMATH_REVISION:
        raise DeepMathMiniBuildError(
            f"DeepMath revision 不匹配：期望 {PINNED_DEEPMATH_REVISION}，实际 {revision or 'unknown'}；不会替换来源。"
        )
    readme = root / "README.md"
    if not readme.is_file():
        raise DeepMathMiniBuildError("Pinned snapshot 缺少 README/license 证据；不能构建。")
    text = readme.read_text(encoding="utf-8")
    license_match = re.search(r"(?m)^license:\s*['\"]?([a-zA-Z0-9.-]+)", text.split("---", 2)[1] if text.startswith("---") and text.count("---") >= 2 else text)
    if not license_match:
        raise DeepMathMiniBuildError("Pinned snapshot README 未提供可审阅的 license metadata；不能构建。")
    license_id = license_match.group(1).lower()
    license_evidence = {"path": "README.md", "sha256": sha256_file(readme)}
    parquet_files = sorted(root.rglob("*.parquet"))
    if not parquet_files:
        raise DeepMathMiniBuildError("Pinned snapshot 中没有 Parquet 文件；不会联网下载。")
    inventory = []
    for path in parquet_files:
        if not path.is_file():
            raise DeepMathMiniBuildError(f"Parquet shard 不可用：{path}")
        inventory.append({
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "absolute_path": path,
        })
    public_inventory = [{k: v for k, v in item.items() if k != "absolute_path"} for item in inventory]
    snapshot_digest = _sha256_bytes(json.dumps(public_inventory, sort_keys=True, separators=(",", ":")).encode())
    return root, inventory, license_id, snapshot_digest, license_evidence


def _load_rows(inventory: list[dict[str, Any]]):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional environment dependency
        raise DeepMathMiniBuildError("离线 DeepMath builder 需要 pyarrow。") from exc
    required = {"question", "final_answer", "r1_solution_1", "r1_solution_2", "r1_solution_3"}
    for item in inventory:
        parquet = pq.ParquetFile(item["absolute_path"])
        missing = sorted(required - set(parquet.schema_arrow.names))
        if missing:
            raise DeepMathMiniBuildError(f"Parquet shard {item['path']} 缺少列：{missing}")
        offset = 0
        for batch in parquet.iter_batches(batch_size=256, columns=sorted(required)):
            for row in batch.to_pylist():
                yield item, offset, row
                offset += 1


def _split(problem: str, seed: int) -> str:
    rank = int(_sha256_bytes(f"{seed}:{problem}".encode())[:16], 16)
    return "eval" if rank % 20 == 0 else "train"


def build_deepmath_mini_sft(
    *,
    source: str | Path,
    output_dir: str | Path,
    tokenizer: TokenizerBackend | None = None,
    benchmark_index: BenchmarkIndex | None = None,
    contamination_sources: list[dict[str, str]] | None = None,
    seed: int = 42,
    max_length: int = 2048,
    target_examples: int = 5000,
    tokenizer_repo: str = PINNED_TOKENIZER_REPO,
    tokenizer_revision: str = PINNED_TOKENIZER_REVISION,
) -> dict[str, Any]:
    """Build a fresh immutable SFT cohort from the exact locally pinned snapshot."""
    output = Path(output_dir).expanduser()
    if output.exists():
        raise FileExistsError(f"Mini SFT 输出必须为全新目录：{output}")
    _, inventory, license_id, snapshot_digest, license_evidence = _source_inventory(source)
    if license_id != "mit":
        raise DeepMathMiniBuildError(f"Pinned revision license metadata 为 {license_id!r}，本变更要求人工复核 MIT 后再接入。")
    tokenizer_backend = tokenizer or HuggingFaceTokenizer(tokenizer_repo, tokenizer_revision, local_files_only=True)
    if tokenizer_repo != PINNED_TOKENIZER_REPO or tokenizer_revision != PINNED_TOKENIZER_REVISION:
        raise DeepMathMiniBuildError("Mini SFT 必须使用已固定的 Qwen3-0.6B-Base tokenizer revision。")
    template = getattr(getattr(tokenizer_backend, "tokenizer", None), "chat_template", None)
    template_sha256 = _sha256_bytes(str(template or "").encode("utf-8"))
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    contamination_rows: list[dict[str, Any]] = []
    topic_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    length_counts: list[int] = []
    raw_count = 0
    unique_count = 0
    train_assigned = 0
    eval_assigned = 0
    eligible_count = 0
    for item, row_index, row in _load_rows(inventory):
        raw_count += 1
        question = normalize_problem(str(row.get("question") or ""))
        if not question:
            counts["empty_question"] += 1
            continue
        pid = problem_id(question)
        if pid in seen:
            counts["duplicate_problem"] += 1
            continue
        seen.add(pid)
        unique_count += 1
        split = _split(pid, seed)
        if split == "eval":
            eval_assigned += 1
            continue
        train_assigned += 1
        if benchmark_index is not None:
            hits = benchmark_index.check(question, fuzzy_threshold=0.85)
            if hits:
                counts["contamination"] += 1
                contamination_rows.append({"problem_id": pid, "source_row_id": f"{item['sha256']}:{row_index}",
                                           "hits": [hit.to_dict() for hit in hits]})
                continue
        gold = str(row.get("final_answer") or "").strip()
        trace_candidates = []
        has_nonempty = False
        has_exact = False
        for trace_index in range(1, 4):
            trace = str(row.get(f"r1_solution_{trace_index}") or "").strip()
            if not trace:
                continue
            has_nonempty = True
            answer = extract_answer(trace)
            if answer is None or not gold or normalize_answer(answer) != normalize_answer(gold):
                continue
            has_exact = True
            messages = [Message(role="user", content=question), Message(role="assistant", content=trace)]
            tokens = tokenizer_backend.count_messages([message.model_dump() for message in messages])
            length_counts.append(tokens)
            if tokens <= max_length:
                trace_candidates.append((tokens, trace_index, trace, messages))
        if not has_nonempty:
            counts["no_nonempty_trace"] += 1
            continue
        if not has_exact:
            counts["no_exact_answer_match"] += 1
            continue
        if not trace_candidates:
            counts["rendered_length_over_limit"] += 1
            continue
        tokens, trace_index, trace, messages = min(trace_candidates, key=lambda x: (x[0], x[1]))
        eligible_count += 1
        source_row_id = _sha256_bytes(f"{item['sha256']}:{row_index}".encode())
        content_fingerprint = _sha256_bytes(trace.encode("utf-8"))[:12]
        record_id = f"dm_{source_row_id[:20]}_r{trace_index}_{content_fingerprint}"
        stable_rank = _sha256_bytes(f"{seed}:{pid}:{record_id}".encode())
        record = SFTRecord(
            id=record_id,
            raw_problem=question,
            messages=messages,
            prompt_template_id="qwen3-math-v1",
            metadata={
                "problem_id": pid,
                "source_record_id": f"deepmath:{source_row_id}",
                "source_row_id": source_row_id,
                "source_trace_id": f"deepmath:{source_row_id}:r{trace_index}",
                "rendered_tokens": tokens,
                "reference_answer": gold,
                "answer_agreement": "exact_normalized_match",
                "answer_verification_method": "extract_answer+normalize_answer:v1",
                "answer_verification_status": "exact_string_agreement_only",
                "reasoning_review": "not_measured",
                "semantic_verification": "not_measured",
                "topic": str(row.get("topic") or "<missing>"),
                "difficulty": row.get("difficulty"),
                "source_revision": PINNED_DEEPMATH_REVISION,
                "source_file_sha256": item["sha256"],
                "source_row_index": row_index,
            },
        ).model_dump(mode="json")
        topic_counts[str(row.get("topic") or "<missing>")] += 1
        difficulty_counts[str(row.get("difficulty") or "<missing>")] += 1
        candidates.append((stable_rank, pid, record))

    selected = [record for _, _, record in sorted(candidates, key=lambda item: (item[0], item[1]))[:target_examples]]
    if not selected:
        raise DeepMathMiniBuildError("筛选后没有合格 SFT 记录；不会放宽规则或生成替代样本。")
    output.mkdir(parents=True, exist_ok=False)
    canonical = output / "canonical" / "sft.jsonl"
    native = output / "native" / "sft.jsonl"
    write_jsonl(canonical, selected)
    write_jsonl(native, (adapt_sft(record) for record in selected))
    write_jsonl(output / "reports" / "contamination.jsonl", contamination_rows)
    content_sha = sha256_file(canonical)
    native_sha = sha256_file(native)
    write_report(native.with_suffix(native.suffix + ".adapter.json"), {
        "adapter": "ms-swift", "stage": "sft", "input": str(canonical), "output": str(native),
        "records": len(selected), "input_sha256": content_sha, "output_sha256": native_sha,
    })
    config = {"seed": seed, "max_length": max_length, "target_examples": target_examples,
              "split": "sha256(seed:problem_id) % 20 == 0 => eval", "selection": "lowest sha256 rank per problem",
              "answer_gate": "extract final answer and exact normalized match", "overlength": "reject; never truncate"}
    processing_steps = [
        step(1, "normalize-and-deduplicate", "filter", "source:deepmath_rows", "pool:unique_problems",
             raw_count, unique_count, "source_rows", output_unit="unique_problems",
             rejected={"empty_question": counts["empty_question"], "duplicate_problem": counts["duplicate_problem"]},
             limitations=["The first source row for each normalized problem identity is retained"]),
        step(2, "problem-level-split", "split", "pool:unique_problems", "pool:train+eval",
             unique_count, unique_count, "unique_problems",
             details={"seed": seed, "train": train_assigned, "eval": eval_assigned, "eval_fraction": 0.05,
                      "split_key": "problem_id", "split_precedes_selection": True}),
        step(3, "decontaminate-training", "filter", "pool:train", "pool:train_screened",
             train_assigned, train_assigned - counts["contamination"], "unique_problems",
             rejected={"benchmark_or_holdout_overlap": counts["contamination"]},
             skipped=benchmark_index is None,
             limitations=["Missing/partial local contamination indexes do not prove global non-contamination"]),
        step(4, "answer-and-rendered-length-gates", "filter", "pool:train_screened", "pool:sft_eligible",
             train_assigned - counts["contamination"], eligible_count, "unique_problems",
             rejected={"no_nonempty_trace": counts["no_nonempty_trace"],
                       "no_exact_answer_match": counts["no_exact_answer_match"],
                       "rendered_length_over_limit": counts["rendered_length_over_limit"]}),
        step(5, "deterministic-budget-selection", "sample", "pool:sft_eligible", "canonical:sft_train",
             eligible_count, len(selected), "unique_problems", output_unit="examples",
             unselected={"budget_excess": eligible_count - len(selected)},
             details={"target": target_examples, "stable_order": "sha256(seed:problem_id:record_id)"}),
    ]
    build_report = {
        "protocol": BUILD_PROTOCOL,
        "stop_reason": "targets_met" if len(selected) >= target_examples else "source_exhausted",
        "processing_steps": processing_steps,
        "selected": {"sft": len(selected)},
        "targets": {"sft": target_examples},
        "counters": dict(sorted(counts.items())),
        "split": {"train_problems": train_assigned, "eval_problems": eval_assigned},
        "length_tokens": {"count": len(length_counts), "min": min(length_counts) if length_counts else None,
                          "max": max(length_counts) if length_counts else None,
                          "mean": sum(length_counts) / len(length_counts) if length_counts else None},
        "quality_evidence": {"answer_agreement_method": "exact normalized answer string match",
                             "answer_agreement_selected": len(selected), "reasoning_review": "not_measured",
                             "semantic_verification": "not_measured"},
        "distributions": {"topic": dict(sorted(topic_counts.items())),
                          "difficulty": dict(sorted(difficulty_counts.items()))},
        "rejection_reasons": dict(sorted(counts.items())),
        "candidate_denominator": {"unique_problems": unique_count, "training_problems": train_assigned,
                                  "post_contamination_training_problems": train_assigned - counts["contamination"]},
        "source_snapshot_sha256": snapshot_digest,
        "license_evidence": license_evidence,
        "source_files": [{k: v for k, v in item.items() if k != "absolute_path"} for item in inventory],
        "contamination_sources": contamination_sources or [],
        "contamination_coverage": "partial" if benchmark_index is not None else "not_measured",
    }
    source = SourceSpec(repo="zwhe99/DeepMath-103K", config="default", revision=PINNED_DEEPMATH_REVISION)
    tokenizer_spec = TokenizerSpec(repo=tokenizer_repo, revision=tokenizer_revision)
    policy = DataPolicy(
        stage="sft", usage="exploratory", prompt_template_id="qwen3-math-v1",
        rules=PolicyRules(correctness="filter", completeness="filter", answer_parse="filter"),
        resources=ResourceLimits(max_length=max_length, target_count=target_examples),
        tokenizer=PolicyTokenizer(repo=tokenizer_repo, revision=tokenizer_revision),
    )
    policy_integrity = (
        "passed" if benchmark_index is not None and all(item["status"] == "completed" for item in processing_steps)
        else "incomplete"
    )
    evidence = _write_evidence(
        output_dir=output, name="deepmath-mini-sft", report=build_report, config=config,
        datasets={"sft": (canonical, selected)}, source=source, tokenizer=tokenizer_spec,
        tokenizer_name=getattr(tokenizer_backend, "name", type(tokenizer_backend).__name__),
        benchmark_index=benchmark_index, benchmark_metadata=contamination_sources or [],
        overlap=counts["contamination"],
        source_acquisition={"mode": "local_snapshot_only", "revision": PINNED_DEEPMATH_REVISION,
                            "license": license_id, "attribution": "zwhe99/DeepMath-103K",
                            "license_evidence": license_evidence,
                            "source_url": "https://huggingface.co/datasets/zwhe99/DeepMath-103K"},
    )
    quality_report_path = output / "reports" / "deepmath_sft_quality.json"
    quality_payload = {
        "schema_version": "1.0", "dataset_id": evidence["sft"]["dataset_id"],
        "content_sha256": content_sha, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "exploratory", "record_count": len(selected), "unique_record_ids": len({r["id"] for r in selected}),
        "duplicate_record_ids": len(selected) - len({r["id"] for r in selected}),
        "rendered_length_tokens": {"count": len(selected), "min": min(r["metadata"]["rendered_tokens"] for r in selected),
                                   "max": max(r["metadata"]["rendered_tokens"] for r in selected),
                                   "over_limit": sum(r["metadata"]["rendered_tokens"] > max_length for r in selected)},
        "answer_check": {"method": "exact normalized string agreement", "numerator": len(selected),
                         "denominator": len(selected), "semantic_verification": "not_measured"},
        "reasoning_review": {"status": "not_measured", "coverage": 0},
        "topic_distribution": dict(sorted(topic_counts.items())),
        "difficulty_distribution": dict(sorted(difficulty_counts.items())),
        "contamination": {"status": build_report["contamination_coverage"], "sources": contamination_sources or [],
                          "matched_training_problems": counts["contamination"]},
        "limitations": ["Exact final-answer agreement is not semantic verification or proof of correct reasoning.",
                        "Human reasoning review has not been performed.",
                        "Only the listed local benchmark/holdout indexes were checked."],
    }
    write_report(quality_report_path, quality_payload)
    write_markdown_report(quality_report_path.with_suffix(".md"), "DeepMath Mini SFT Quality Evidence", quality_payload)
    processing_path = output / "reports" / "processing_report.json"
    processing_payload = json.loads(processing_path.read_text(encoding="utf-8"))
    processing_payload["source"]["license"] = license_id
    processing_payload["source"]["license_evidence"] = license_evidence
    processing_payload["source"]["snapshot_sha256"] = snapshot_digest
    processing_payload["source"]["source_file_hashes"] = [
        {k: v for k, v in item.items() if k != "absolute_path"} for item in inventory
    ]
    processing_payload["processing_steps"][-1]["details"]["candidate_denominator"] = build_report["candidate_denominator"]
    processing_payload["processing_steps"][-1]["details"]["rejection_reasons"] = build_report["rejection_reasons"]
    processing_payload["benchmark_screening"]["coverage"] = build_report["contamination_coverage"]
    write_report(processing_path, processing_payload)
    # Refresh the processing digest embedded by _write_evidence after adding source-level evidence.
    evidence["sft"]["processing_report"] = ReportReference(
        path="../reports/processing_report.json", sha256=file_sha(processing_path),
        report_id=processing_payload["build_id"],
    )
    processing_sha = file_sha(processing_path)
    build_report["processing_report"]["sha256"] = processing_sha
    build_run_path = output / "runs" / build_report["data_build_run_id"] / "lineage.json"
    if build_run_path.is_file():
        build_run = json.loads(build_run_path.read_text(encoding="utf-8"))
        for artifact in build_run.get("outputs", []):
            if artifact.get("path") == "reports/processing_report.json":
                artifact.setdefault("digest", {})["value"] = file_sha(processing_path)
        write_report(build_run_path, build_run)
    manifest = DatasetManifest(
        **evidence["sft"], stage="sft", source=source, tokenizer=tokenizer_spec,
        prompt_template_id="qwen3-math-v1",
        protocol_versions={"normalization": "v1", "problem_split": "v1", "answer_gate": "exact-normalized-v1",
                           "rendered_length_gate": "chat-template-v1", "stable_identity": "source-row-trace-content-v1"},
        split_seed=seed, sample_count=len(selected),
        hashes={"source_row_ids_sha256": _sha256_bytes("\n".join(sorted(r["metadata"]["source_row_id"] for r in selected)).encode()),
                "canonical_records_sha256": content_sha, "native_records_sha256": native_sha},
        metadata={"usage": "exploratory", "max_rendered_tokens": max_length, "license": license_id,
                  "data_policy": policy.artifact(), "policy_integrity": policy_integrity,
                  "policy_integrity_scope": "configured processing gates and local benchmark indexes; partial source coverage is reported separately",
                  "attribution": "zwhe99/DeepMath-103K", "license_url": "https://huggingface.co/datasets/zwhe99/DeepMath-103K",
                  "license_evidence": license_evidence,
                  "snapshot_sha256": snapshot_digest, "tokenizer_chat_template_sha256": template_sha256,
                  "canonical_path": "canonical/sft.jsonl",
                  "quality_evidence_path": "../reports/deepmath_sft_quality.json",
                  "native_path": "native/sft.jsonl", "native_sha256": native_sha},
    )
    manifest_path = output / "manifests" / "sft-train-v1.yaml"
    save_manifest(manifest_path, manifest)
    write_report(output / "reports" / "deepmath_sft_build.json", build_report)
    return {"build_report": build_report, "dataset_id": manifest.dataset_id,
            "manifest": str(manifest_path), "canonical": str(canonical), "native": str(native),
            "count": len(selected), "shortfall": max(0, target_examples - len(selected)),
            "canonical_sha256": content_sha, "native_sha256": native_sha,
            "quality_report": str(quality_report_path)}
