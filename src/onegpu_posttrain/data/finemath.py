from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..schemas import CPTRecord
from .decontaminate import BenchmarkIndex
from .evidence import ratio, step
from .tokenization import TokenizerBackend


def _doc_id(row: dict[str, Any]) -> str:
    """Stable source-document/split identity, intentionally independent of extracted text."""
    stable = str(row.get("url") or row.get("warc_filename") or row.get("text") or "")
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _record_id(stage: str, document_id: str, chunk_index: int, chunk_text: str) -> str:
    """Identify emitted chunk content without changing URL-keyed split assignment."""
    content_digest = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
    return f"cpt_{stage}_{document_id[:12]}_{chunk_index:04d}_{content_digest}"


@dataclass
class FineMathBuildConfig:
    train_token_budget: int = 500_000
    holdout_token_budget: int = 50_000
    max_seq_tokens: int = 1024
    holdout_permille: int = 50
    fuzzy_threshold: float = 0.85
    remove_exact_contamination: bool = True
    max_raw_rows: int | None = None


@dataclass
class FineMathBuildResult:
    train: list[dict[str, Any]] = field(default_factory=list)
    holdout: list[dict[str, Any]] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    contamination: list[dict[str, Any]] = field(default_factory=list)


class FineMathBuilder:
    def __init__(self, tokenizer: TokenizerBackend, config: FineMathBuildConfig, benchmark_index: BenchmarkIndex | None = None):
        self.tokenizer = tokenizer
        self.config = config
        self.benchmark_index = benchmark_index
        self.counters: Counter[str] = Counter()

    def _is_holdout(self, doc_id: str) -> bool:
        return int(doc_id[:8], 16) % 1000 < self.config.holdout_permille

    def build(self, rows: Iterable[dict[str, Any]]) -> FineMathBuildResult:
        out = FineMathBuildResult()
        train_tokens = 0
        holdout_tokens = 0
        chunk_token_lengths: list[int] = []
        train_docs: set[str] = set()
        holdout_docs: set[str] = set()
        contamination: list[dict[str, Any]] = []
        stop_reason = "source_exhausted"

        for raw_index, row in enumerate(rows):
            if self.config.max_raw_rows is not None and raw_index >= self.config.max_raw_rows:
                self.counters["max_raw_rows_reached"] += 1
                stop_reason = "max_raw_rows"
                break
            if train_tokens >= self.config.train_token_budget and holdout_tokens >= self.config.holdout_token_budget:
                stop_reason = "targets_met"
                break
            self.counters["raw_rows"] += 1
            text = str(row.get("text") or "").strip()
            if not text:
                self.counters["empty_text"] += 1
                continue
            self.counters["nonempty_documents"] += 1
            did = _doc_id(row)
            target_holdout = self._is_holdout(did)
            self.counters["assigned_holdout" if target_holdout else "assigned_train"] += 1

            # FineMath upstream is already decontaminated; this is a secondary verification scan.
            if self.benchmark_index is not None:
                hits = self.benchmark_index.check(text, fuzzy_threshold=self.config.fuzzy_threshold, metric="target_coverage")
                if hits:
                    contamination.append({"document_id": did, "url": row.get("url"), "hits": [h.to_dict() for h in hits]})
                    self.counters["contamination_candidates"] += 1
                    if self.config.remove_exact_contamination and any(h.kind == "exact" for h in hits):
                        self.counters["exact_contamination_removed"] += 1
                        continue

            self.counters["screened_documents"] += 1
            chunks = self.tokenizer.chunk_text(text, self.config.max_seq_tokens)
            if not chunks:
                self.counters["no_chunks"] += 1
                continue
            self.counters["chunked_documents"] += 1
            self.counters["chunks_generated"] += len(chunks)
            for chunk_idx, chunk in enumerate(chunks):
                tokens = self.tokenizer.count_text(chunk)
                if tokens <= 0:
                    self.counters["zero_token_chunks"] += 1
                    continue
                if target_holdout:
                    if holdout_tokens >= self.config.holdout_token_budget:
                        break
                    rec = CPTRecord(
                        id=_record_id("holdout", did, chunk_idx, chunk),
                        text=chunk,
                        source="finemath-4plus",
                        metadata={
                            "data_role": "final_holdout",
                            "document_id": did,
                            "url": row.get("url"),
                            "chunk_index": chunk_idx,
                            "tokens": tokens,
                            "score": row.get("score"),
                            "int_score": row.get("int_score"),
                        },
                    ).model_dump(mode="json")
                    out.holdout.append(rec)
                    holdout_docs.add(did)
                    holdout_tokens += tokens
                else:
                    if train_tokens >= self.config.train_token_budget:
                        break
                    rec = CPTRecord(
                        id=_record_id("train", did, chunk_idx, chunk),
                        text=chunk,
                        source="finemath-4plus",
                        metadata={
                            "data_role": "train_seed",
                            "document_id": did,
                            "url": row.get("url"),
                            "chunk_index": chunk_idx,
                            "tokens": tokens,
                            "score": row.get("score"),
                            "int_score": row.get("int_score"),
                        },
                    ).model_dump(mode="json")
                    out.train.append(rec)
                    train_docs.add(did)
                    train_tokens += tokens
                    chunk_token_lengths.append(tokens)

        if train_tokens >= self.config.train_token_budget and holdout_tokens >= self.config.holdout_token_budget:
            stop_reason = "targets_met"
        out.contamination = contamination
        overlap = len(train_docs & holdout_docs)
        out.report = {
            "protocol": "finemath-build-v3",
            "data_role_protocol": "cpt-data-role-v1",
            "data_role_counts": {
                "train_seed": len(out.train),
                "final_holdout": len(out.holdout),
            },
            "stop_reason": stop_reason,
            "processing_steps": self._processing_steps(len(out.train) + len(out.holdout)),
            "tokenizer": self.tokenizer.name,
            "train": {
                "token_budget": self.config.train_token_budget,
                "tokens": train_tokens,
                "records": len(out.train),
                "documents": len(train_docs),
                "avg_fill_ratio": (sum(chunk_token_lengths) / (len(chunk_token_lengths) * self.config.max_seq_tokens)) if chunk_token_lengths else 0.0,
            },
            "holdout": {
                "token_budget": self.config.holdout_token_budget,
                "tokens": holdout_tokens,
                "records": len(out.holdout),
                "documents": len(holdout_docs),
            },
            "document_overlap": overlap,
            "counters": dict(sorted(self.counters.items())),
            "contamination_candidates": len(contamination),
        }
        return out

    def _processing_steps(self, selected_chunks: int) -> list[dict[str, Any]]:
        c = self.counters
        raw, nonempty = c["raw_rows"], c["nonempty_documents"]
        screened, generated = c["screened_documents"], c["chunks_generated"]
        return [
            step(1, "validate-text", "filter", "source:processed_rows", "pool:nonempty", raw, nonempty,
                 "raw_rows", rejected={"empty_text": c["empty_text"]},
                 before={"nonempty_rate": ratio(nonempty, raw, "strip_nonempty_text:v1")},
                 after={"nonempty_rate": ratio(nonempty, nonempty, "strip_nonempty_text:v1")}),
            step(2, "document-split", "split", "pool:nonempty", "pool:assigned", nonempty, nonempty,
                 "documents", details={"holdout_permille": self.config.holdout_permille,
                                        "train_occurrences": c["assigned_train"], "holdout_occurrences": c["assigned_holdout"]},
                 limitations=["Counts are source document occurrences, not deduplicated document IDs"]),
            step(3, "decontaminate", "filter", "pool:assigned", "pool:screened", nonempty, screened,
                 "documents", rejected={"exact_contamination": c["exact_contamination_removed"]},
                 skipped=self.benchmark_index is None or not self.benchmark_index.normalized_to_ids,
                 details={"fuzzy_candidates": c["contamination_candidates"],
                          "metric": "target_coverage",
                          "remove_exact_contamination": self.config.remove_exact_contamination,
                          "fuzzy_threshold": self.config.fuzzy_threshold},
                 limitations=["No nonempty benchmark index available; no contamination conclusion is available"]
                 if self.benchmark_index is None or not self.benchmark_index.normalized_to_ids
                 else ["Fuzzy candidates are reported, not necessarily removed; only configured exact-hit removal changes output"]),
            step(4, "tokenizer-chunking", "tokenize", "pool:screened", "pool:chunks", screened, generated,
                 "documents", output_unit="chunks", rejected={"no_chunks": c["no_chunks"]},
                 details={"chunked_documents": c["chunked_documents"], "max_seq_tokens": self.config.max_seq_tokens,
                          "tokenizer": self.tokenizer.name, "record_identity": "document_id+chunk_index+chunk_sha256:v2",
                          "split_identity": "url_derived_document_id:v1"}),
            step(5, "token-budget-selection", "sample", "pool:chunks", "canonical:cpt", generated, selected_chunks,
                 "chunks", rejected={"zero_token_chunks": c["zero_token_chunks"]},
                 unselected={"token_budget": generated - selected_chunks - c["zero_token_chunks"]},
                 details={"train_token_budget": self.config.train_token_budget, "holdout_token_budget": self.config.holdout_token_budget},
                 limitations=["Complete chunks may overshoot token targets; quota exclusion is not a quality rejection",
                              "Only chunks inspected by the existing selection loop are checked for zero tokens"]),
        ]
