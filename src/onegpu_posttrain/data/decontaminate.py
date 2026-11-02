from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from .normalize import normalize_problem

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|\\[A-Za-z]+|[^\s]")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize_problem(text).lower())


def _shingles(text: str, n: int) -> set[tuple[str, ...]]:
    toks = _tokens(text)
    if not toks:
        return set()
    if len(toks) < n:
        return {tuple(toks)}
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


@dataclass(frozen=True)
class ContaminationHit:
    kind: str
    benchmark_id: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


class BenchmarkIndex:
    """Exact + shingle-Jaccard benchmark contamination index.

    The inverted shingle index avoids an O(dataset * benchmark) fuzzy scan.
    """

    def __init__(self, rows: Iterable[tuple[str, str]], *, shingle_n: int = 5):
        self.shingle_n = shingle_n
        self.normalized_to_ids: dict[str, list[str]] = defaultdict(list)
        self.shingles_by_id: dict[str, set[tuple[str, ...]]] = {}
        self.inverted: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for benchmark_id, text in rows:
            normalized = normalize_problem(text)
            if not normalized:
                continue
            self.normalized_to_ids[normalized].append(benchmark_id)
            ss = _shingles(normalized, shingle_n)
            self.shingles_by_id[benchmark_id] = ss
            for s in ss:
                self.inverted[s].add(benchmark_id)

    def check(self, text: str, *, fuzzy_threshold: float = 0.85, metric: str = "jaccard") -> list[ContaminationHit]:
        normalized = normalize_problem(text)
        exact_ids = self.normalized_to_ids.get(normalized, [])
        if exact_ids:
            return [ContaminationHit("exact", bid, 1.0) for bid in exact_ids]
        query = _shingles(normalized, self.shingle_n)
        if not query:
            return []
        counts: Counter[str] = Counter()
        for shingle in query:
            for benchmark_id in self.inverted.get(shingle, ()):  # candidates only
                counts[benchmark_id] += 1
        hits: list[ContaminationHit] = []
        for benchmark_id, _ in counts.most_common():
            target = self.shingles_by_id[benchmark_id]
            shared = len(query & target)
            if metric == "jaccard":
                union = len(query | target)
                score = shared / union if union else 0.0
            elif metric == "target_coverage":
                score = shared / len(target) if target else 0.0
            else:
                raise ValueError(f"Unknown contamination metric: {metric}")
            if score >= fuzzy_threshold:
                hits.append(ContaminationHit("fuzzy", benchmark_id, score))
        return hits
