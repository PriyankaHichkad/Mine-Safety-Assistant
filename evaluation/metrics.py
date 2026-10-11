"""Retrieval metrics, computed at document level.

A retriever returns a ranked list of chunks; several chunks can come from the
same document, so the ranking is de-duplicated to document IDs (first
occurrence wins) before any metric is computed.
"""
import math
from typing import Iterable, List, Sequence, Set, Tuple


def dedupe(ranked: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for d in ranked:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def hit_at_k(relevant: Set[str], ranked: Sequence[str], k: int) -> float:
    """1.0 if at least one relevant document is in the top-k, else 0.0."""
    return 1.0 if relevant & set(ranked[:k]) else 0.0


def recall_at_k(relevant: Set[str], ranked: Sequence[str], k: int) -> float:
    """Fraction of relevant documents found in the top-k."""
    if not relevant:
        raise ValueError("recall is undefined with no relevant documents")
    return len(relevant & set(ranked[:k])) / len(relevant)


def precision_at_k(relevant: Set[str], ranked: Sequence[str], k: int) -> float:
    """Relevant documents in the top-k divided by k."""
    return len(relevant & set(ranked[:k])) / k


def precision_ceiling(relevant: Set[str], k: int) -> float:
    """Best achievable precision@k given how many relevant docs exist."""
    return min(len(relevant), k) / k


def reciprocal_rank(relevant: Set[str], ranked: Sequence[str], k: int) -> float:
    for i, d in enumerate(ranked[:k], 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
