"""Retrieval quality metrics, implemented from first principles.

Retrieval is judged by the ranking it returns, not a single accuracy number. This module
provides the standard information-retrieval metrics, each taking the ranked list of
retrieved chunk ids and the set (or graded map) of relevant ids:

- ``recall_at_k`` / ``precision_at_k`` - coverage and purity of the top k.
- ``reciprocal_rank`` - 1 / rank of the first relevant result (averaged to MRR).
- ``average_precision`` - area under the precision-recall curve (averaged to MAP).
- ``ndcg_at_k`` - normalized discounted cumulative gain, supporting graded relevance.

All are pure functions over ids, so they are independent of any retriever.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items that appear in the top ``k``."""
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top ``k`` that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive")
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for item in top if item in relevant) / len(top)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant item, or 0 if none is retrieved."""
    for index, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def average_precision(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Average precision: mean of precision values at each relevant hit."""
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for index, item in enumerate(retrieved, start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / index
    return precision_sum / len(relevant)


def dcg_at_k(retrieved: Sequence[str], gains: Mapping[str, float], k: int) -> float:
    """Discounted cumulative gain of the top ``k`` under graded relevance ``gains``."""
    total = 0.0
    for index, item in enumerate(retrieved[:k], start=1):
        gain = gains.get(item, 0.0)
        total += gain / math.log2(index + 1)
    return total


def ndcg_at_k(retrieved: Sequence[str], gains: Mapping[str, float], k: int) -> float:
    """Normalized DCG: DCG divided by the ideal DCG for the given gains."""
    actual = dcg_at_k(retrieved, gains, k)
    ideal_order = sorted(gains, key=lambda item: gains[item], reverse=True)
    ideal = dcg_at_k(ideal_order, gains, k)
    if ideal == 0.0:
        return 0.0
    return actual / ideal
